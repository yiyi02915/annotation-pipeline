"""Incremental audio-repaired reannotation with independent GLM judgments."""
import json,copy,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import requests
from pipeline_core import *
from llm_client import Client,BOUNDARY_SCHEMA,EVENT_SCHEMA
from run_pipeline import export_results
root=Path(__file__).resolve().parent;base=root/'baseline'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda n,v:(root/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def state(stage,**kw):
    save('run_state.json',dict(stage=stage,unix_time=time.time(),**kw));print(stage,kw,flush=True)
def unit_payload(u):return {k:v for k,v in u.items() if k!='unit_id'}
def event_payload(c,units):
    p={k:c[k] for k in ['candidate_kind','floor','incoming','overlap_seconds','gap_seconds','context']}
    p['context']=[dict(context_index=i,**u) for i,u in enumerate(c['context'])]
    allowed=[-1]+[i for i,u in enumerate(c['context']) if u['speaker']==c['floor']['speaker'] and u['start']>=c['incoming']['start']]
    p['allowed_response_context_indices']=allowed
    p['floor_semantic_unit']=unit_payload(units[c['floor']['utterance_id']])
    p['incoming_semantic_unit']=unit_payload(units[c['incoming']['utterance_id']])
    schema=copy.deepcopy(EVENT_SCHEMA);schema['properties']['response_context_index']={'type':'integer','enum':allowed}
    return p,schema
def boundary_payload(c):return {k:c[k] for k in ['left','right','gap_seconds','context']}

def main():
    records=load(root/'records.json')
    bc=[c for r in records for c in make_boundaries(r)];ec=[c for r in records for c in make_events(r)]
    save('boundary_inputs.json',bc);save('event_inputs.json',ec)
    old_bc={x['candidate_id']:x for x in load(base/'boundary_inputs.json')}
    old_ec={x['candidate_id']:x for x in load(base/'event_inputs.json')}
    old_b={x['candidate_id']:x for x in load(base/'boundary_results.json')}
    old_e={x['event_id']:x for x in load(base/'interaction_results.json')}
    old_units={sid:u for u in load(base/'provisional_units.json') for sid in u['source_utterance_ids']}
    bp=(root/'boundary_prompt.txt').read_text(encoding='utf8')
    bp+='\nFor code E, the LEFT contribution must have been interrupted and yielded before the RIGHT contribution resumes. An interruption of the RIGHT contribution later does not justify the boundary before it. Cite the relevant participant and times in the reason. A pause or ellipsis alone is not proof of an interruption. Do not replace missing audio evidence with invented prosody.'
    ep=(root/'event_prompt.txt').read_text(encoding='utf8')
    ep+='\nUnknown outcome fields are allowed when the interaction type is supported. If no eligible subsequent floor-speaker response is visible, set content_uptake=UNKNOWN and response_context_index=-1; do not invent uptake. A completed prior contribution is incompatible with INTERRUPT. A listener asking for substantive repair or redirection is not BACKCHANNEL under this definition. Distinguish cooperative completion from an attempt to take the floor; overlap and cessation alone do not establish that attempt. Main-speaker continuation following listener feedback is OTHER, not a fresh interruption of the listener.'
    regions=load(root/'unresolved_audio_regions.json')
    save('effective_prompts.json',dict(boundary=bp,event=ep))
    state('waiting_for_strong_model')
    for _ in range(720):
        try:
            rr=requests.get('http://127.0.0.1:18351/v1/models',timeout=3)
            if rr.ok and any(m['id']=='glm-judge' for m in rr.json()['data']):break
        except requests.RequestException:pass
        time.sleep(5)
    else:raise TimeoutError('GLM service unavailable')
    client=Client(root,'http://127.0.0.1:18351/v1','glm-judge')
    def judged(kind,c,payload,schema,prompt,dual):
        trace=[]
        def one(suffix,instruction,p=payload):
            rr=client.call(prompt+'\n'+instruction,p,schema,c['candidate_id']+suffix)
            raw=rr['parsed'];err=validate_boundary(raw)[1] if kind=='boundary' else validate_event(c,raw)
            if err:
                rr=client.call(prompt+'\nThe previous output failed: '+err+'. Correct the label using supplied evidence. Quotes must be verbatim substrings of floor.text and incoming.text, never their semantic units. Unsupported outcomes must be UNKNOWN.',p,schema,c['candidate_id']+suffix+':validation_retry')
                raw=rr['parsed'];err=validate_boundary(raw)[1] if kind=='boundary' else validate_event(c,raw)
            trace.append(dict(role=suffix,result=raw,error=err,request_hash=rr['request_hash']))
            return raw,err
        a,err=one(':judge','Independently identify the action, response target, and chronology. Use only the supplied observations.')
        if dual:
            b,err_b=one(':audit','Independently audit the interpretation. Examine legitimate turn transitions, listener feedback, missing evidence, and whose contribution was ongoing.')
            keys=['code'] if kind=='boundary' else ['interaction_type','function','floor_completion','floor_response','content_uptake']
            agree=not err and not err_b and all(a.get(k)==b.get(k) for k in keys)
            if not agree:
                p=copy.deepcopy(payload);p['conflicting_analyses']=[a,b]
                a,err=one(':arbiter','Resolve differences using specific evidence, not majority voting. If unresolved, preserve uncertainty.',p)
        return a,err,trace
    def annotate_boundary(c):
        cid=c['candidate_id'];old=old_b.get(cid);rule=boundary_rule(c)
        unchanged=cid in old_bc and boundary_payload(c)==boundary_payload(old_bc[cid]) and c['quality_flags']==old_bc[cid]['quality_flags']
        if rule:
            decision,origin=rule;raw=None;error=None;trace=[]
        elif old and old['decision']!='PENDING_AUTOMATIC' and unchanged:
            out=copy.deepcopy(old);out['v6_origin']='reused_unchanged_input';return out
        else:
            raw,error,trace=judged('boundary',c,boundary_payload(c),BOUNDARY_SCHEMA,bp,bool(old and old['decision']=='PENDING_AUTOMATIC'))
            decision=validate_boundary(raw)[0] if not error else None;decision=decision or 'PENDING_AUTOMATIC';origin='glm_v6_boundary'
        return dict(candidate_id=cid,conversation_id=c['conversation_id'],left_id=c['left']['utterance_id'],right_id=c['right']['utterance_id'],decision=decision,origin=origin,v6_origin=origin,error=error,model_result=raw,judgments=trace,human_required=False)
    state('strong_model_boundaries',total=len(bc))
    boundaries=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,b in enumerate(pool.map(annotate_boundary,bc),1):
            boundaries.append(b)
            if i%20==0:save('boundary_results.json',boundaries);print('boundaries',i,'/',len(bc),flush=True)
    save('boundary_results.json',boundaries)
    units={}
    for r in records:
        us,_,_=assemble_units(r,[b for b in boundaries if b['conversation_id']==r['conversation_id']],[])
        for u in us:
            for sid in u['source_utterance_ids']:units[sid]=u
    save('provisional_units.json',list({u['unit_id']:u for u in units.values()}.values()))
    def annotate_event(c):
        cid=c['candidate_id'];old=old_e.get(cid);payload,schema=event_payload(c,units)
        uncertain=[a for a in regions if a['conversation_id']==c['conversation_id'] and a['start']<c['incoming']['end']+15 and a['end']>c['incoming']['start']-20]
        if uncertain:payload['unresolved_audio_regions']=uncertain
        unchanged=cid in old_ec and payload==event_payload(old_ec[cid],old_units)[0] and c['quality_flags']==old_ec[cid]['quality_flags']
        if old and unchanged and (old['status']=='accepted' or c['quality_flags']):
            out=copy.deepcopy(old);out['v6_origin']='reused_unchanged_input';return out
        raw,error,trace=judged('event',c,payload,schema,ep,bool(old and old['status']!='accepted' and not c['quality_flags']))
        safe=raw if not error else dict(interaction_type='UNRESOLVED',function='UNKNOWN',secondary_function='NONE',floor_response='UNKNOWN',content_uptake='UNKNOWN',floor_completion='UNKNOWN',response_context_index=-1,reason='Automatic output validation failed: '+str(error))
        status='accepted' if not error and raw['interaction_type']!='UNRESOLVED' and not c['quality_flags'] else 'PENDING_AUTOMATIC'
        result=dict(event_id=cid,conversation_id=c['conversation_id'],floor_id=c['floor']['utterance_id'],incoming_id=c['incoming']['utterance_id'],start=c['incoming']['start'],end=c['incoming']['end'],candidate_kind=c['candidate_kind'],overlap_seconds=c['overlap_seconds'],status=status,validation_error=error,request_hash=trace[-1]['request_hash'],judgments=trace,human_required=False,v6_origin='glm_v6_event',**safe)
        index=result['response_context_index'];result['response_utterance_id']=c['context'][index]['utterance_id'] if index>=0 else None
        if c['quality_flags']:result['input_quality_flags']=c['quality_flags']
        result['incoming_control_action']={'BACKCHANNEL':'BRIEF_FEEDBACK','INTERRUPT':'TAKE_FLOOR_ATTEMPT','TURN':'TAKE_TURN','OTHER':'OTHER','UNRESOLVED':'UNKNOWN'}[result['interaction_type']]
        result['floor_control_action']={'CONTINUE':'CONTINUE_SPEAKING','YIELD':'YIELD_FLOOR','UNKNOWN':'UNKNOWN'}[result['floor_response']]
        result['label_kind']='model_inferred_observation';result['policy_target']=False
        if uncertain:result['audio_context_uncertainty']=[a['region_id'] for a in uncertain]
        return result
    state('strong_model_interactions',total=len(ec))
    events=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,e in enumerate(pool.map(annotate_event,ec),1):
            events.append(e)
            if i%20==0:save('interaction_results.json',events);print('events',i,'/',len(ec),flush=True)
    save('interaction_results.json',events)
    attempts=[r for x in boundaries+events for r in x.get('judgments',[])]
    if attempts and all(r['error'] for r in attempts):raise RuntimeError('No valid strong-model judgments')
    independent={b['candidate_id']:b for b in load(base/'adjudicated_boundaries.json')}
    export_results(root,records,boundaries,events,independent,'glm-judge')
    # Missing audio-derived text is preserved as uncertainty, never silently dropped.
    training=[json.loads(x) for x in (root/'training_annotations.jsonl').read_text(encoding='utf8').splitlines()]
    pending=[json.loads(x) for x in (root/'pending_automatic.jsonl').read_text(encoding='utf8').splitlines()]
    safe=[];excluded=[]
    for row in training:
        if row['annotation'].get('audio_context_uncertainty'):
            row.update(pending_reason='unresolved_audio_in_input_or_outcome_context',human_required=False);pending.append(row);excluded.append(row['example_id'])
        else:safe.append(row)
    for a in regions:pending.append(dict(task='audio_transcription_region',annotation=a,human_required=False))
    for name,rows in [('training_annotations.jsonl',safe),('pending_automatic.jsonl',pending)]:
        (root/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf8')
    summary=load(root/'summary.json');summary.update(version='duplex-e2e-v6',baseline_training_annotations=207,boundary_origins=dict(Counter(b.get('v6_origin') for b in boundaries)),event_origins=dict(Counter(e.get('v6_origin') for e in events)),training_annotations=len(safe),pending_items=len(pending),audio_context_excluded_training=len(excluded),unresolved_audio_regions=len(regions))
    save('audio_context_exclusions.json',excluded)
    manifest=load(root/'manifest.json');manifest.update(version='duplex-e2e-v6',effective_prompt_sha256=hashlib.sha256((root/'effective_prompts.json').read_bytes()).hexdigest(),baseline_input_sha256=hashlib.sha256((base/'records.json').read_bytes()).hexdigest())
    save('manifest.json',manifest)
    save('summary.json',summary)
    state('annotation_complete',summary=summary)

if __name__=='__main__':
    try:main()
    except Exception as exc:state('failed',error=repr(exc));raise
