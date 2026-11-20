import argparse,json,hashlib,platform,time,copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pipeline_core import *
from llm_client import Client,BOUNDARY_SCHEMA,EVENT_SCHEMA
from adjudicate import adjudicate

def run(root,url,model,workers=4):
    root=Path(root);load=lambda name:json.loads((root/name).read_text(encoding='utf8'))
    save=lambda name,value:(root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    records=load('records.json');bcandidates=load('boundary_inputs.json');ecandidates=load('event_inputs.json')
    client=Client(root,url,model)
    import requests
    health=requests.get(url+'/models',timeout=30);health.raise_for_status()
    assert model in [m['id'] for m in health.json()['data']], 'Wrong model endpoint'
    save('endpoint_models.json',health.json())
    if not (root/'adjudicated_boundaries.json').exists():adjudicate(root,url,model)
    independent={b['candidate_id']:b for b in load('adjudicated_boundaries.json')}
    bprompt=(root/'boundary_prompt.txt').read_text(encoding='utf8')
    eprompt=(root/'event_prompt.txt').read_text(encoding='utf8')
    def annotate_boundary(c):
        cid=c['candidate_id'];rule=boundary_rule(c);raw=None;error=None
        if cid in independent:
            decision=independent[cid]['decision'];origin='independent_dispute_resolution'
            judged=independent[cid]
            raw=(judged.get('arbitration') or {}).get('result')
            if raw is None and judged.get('responses'):raw=judged['responses'][0].get('result')
        elif rule:decision,origin=rule
        elif c.get('previous_result',{}).get('decision') in {'MERGE','KEEP_BOUNDARY'}:
            decision=c['previous_result']['decision'];origin='accepted_v4_same_inputs'
        else:
            payload={k:c[k] for k in ['left','right','gap_seconds','context']}
            response=client.call(bprompt,payload,BOUNDARY_SCHEMA,cid)
            raw=response['parsed'];decision,error=validate_boundary(raw);origin=model+'_boundary'
            if error:
                retry=client.call(bprompt+'\nThe preceding output failed validation: '+error+'. Return a grounded complete result.',payload,BOUNDARY_SCHEMA,cid+':retry')
                raw=retry['parsed'];decision,error=validate_boundary(raw)
            decision=decision or 'PENDING_AUTOMATIC'
        result=dict(candidate_id=cid,conversation_id=c['conversation_id'],left_id=c['left']['utterance_id'],right_id=c['right']['utterance_id'],decision=decision,origin=origin,error=error,model_result=raw,human_required=False)
        return result
    boundaries=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,b in enumerate(pool.map(annotate_boundary,bcandidates),1):
            boundaries.append(b)
            if n%20==0:save('boundary_results.json',boundaries);print('boundaries',n,'/',len(bcandidates),flush=True)
    save('boundary_results.json',boundaries)
    attempted=[b for b in boundaries if b['origin']==model+'_boundary']
    if attempted and all(b['error'] for b in attempted):
        raise RuntimeError('Bulk boundary model returned no valid outputs; checkpoints preserved')
    provisional={}
    for r in records:
        bs=[b for b in boundaries if b['conversation_id']==r['conversation_id']]
        units,mapping,_=assemble_units(r,bs,[])
        unit_by_id={u['unit_id']:u for u in units}
        for sid,uid in mapping.items():provisional[sid]=unit_by_id[uid]
    save('provisional_units.json',list({u['unit_id']:u for u in provisional.values()}.values()))
    def annotate_event(c):
        payload={k:c[k] for k in ['candidate_kind','floor','incoming','overlap_seconds','gap_seconds','context']}
        # Explicit indices avoid asking the model to count a long JSON array.
        payload['context']=[dict(context_index=i,**u) for i,u in enumerate(c['context'])]
        allowed_response_indices=[-1]+[i for i,u in enumerate(c['context']) if u['speaker']==c['floor']['speaker'] and u['start']>=c['incoming']['start']]
        payload['allowed_response_context_indices']=allowed_response_indices
        schema=copy.deepcopy(EVENT_SCHEMA)
        schema['properties']['response_context_index']={'type':'integer','enum':allowed_response_indices}
        payload['floor_semantic_unit']=provisional[c['floor']['utterance_id']]
        payload['incoming_semantic_unit']=provisional[c['incoming']['utterance_id']]
        rr=client.call(eprompt,payload,schema,c['candidate_id']);raw=rr['parsed'];error=validate_event(c,raw)
        if error:
            rr=client.call(eprompt+'\nValidation correction required: '+error+'. Re-evaluate using exact quotes from floor.text and incoming.text (not semantic-unit text) and the explicit context_index values. Use -1 when no eligible subsequent response supports the label.',payload,schema,c['candidate_id']+':retry');raw=rr['parsed'];error=validate_event(c,raw)
        status='accepted' if not error and raw['interaction_type']!='UNRESOLVED' else 'PENDING_AUTOMATIC'
        safe=raw if not error else dict(interaction_type='UNRESOLVED',function='UNKNOWN',floor_response='UNKNOWN',content_uptake='UNKNOWN',floor_completion='UNKNOWN',response_context_index=-1,reason='Automatic output validation failed: '+str(error))
        result=dict(event_id=c['candidate_id'],conversation_id=c['conversation_id'],floor_id=c['floor']['utterance_id'],incoming_id=c['incoming']['utterance_id'],start=c['incoming']['start'],end=c['incoming']['end'],candidate_kind=c['candidate_kind'],overlap_seconds=c['overlap_seconds'],status=status,validation_error=error,request_hash=rr['request_hash'],human_required=False,**safe)
        response_index=result['response_context_index']
        result['response_utterance_id']=c['context'][response_index]['utterance_id'] if response_index>=0 else None
        if c.get('quality_flags'):
            result['status']='PENDING_AUTOMATIC';result['input_quality_flags']=c['quality_flags']
        result['incoming_control_action']={'BACKCHANNEL':'BRIEF_FEEDBACK','INTERRUPT':'TAKE_FLOOR_ATTEMPT','TURN':'TAKE_TURN','OTHER':'OTHER','UNRESOLVED':'UNKNOWN'}[result['interaction_type']]
        result['floor_control_action']={'CONTINUE':'CONTINUE_SPEAKING','YIELD':'YIELD_FLOOR','UNKNOWN':'UNKNOWN'}[result['floor_response']]
        result['label_kind']='model_inferred_observation';result['policy_target']=False
        return result
    events=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,e in enumerate(pool.map(annotate_event,ecandidates),1):
            events.append(e)
            if n%20==0:save('interaction_results.json',events);print('events',n,'/',len(ecandidates),flush=True)
    save('interaction_results.json',events)
    if events and all(e['validation_error'] for e in events):
        raise RuntimeError('Interaction model returned no valid outputs; checkpoints preserved')
    return export_results(root,records,boundaries,events,independent,model)

def export_results(root,records,boundaries,events,independent,model):
    root=Path(root)
    save=lambda name,value:(root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    consistency_issues=reconcile_boundary_evidence(records,boundaries,events)
    save('boundary_consistency_issues.json',consistency_issues)
    save('boundary_results.json',boundaries)
    conversations=[];training=[];pending=[];all_issues=[];overrides=[]
    for r in records:
        bs=[b for b in boundaries if b['conversation_id']==r['conversation_id']]
        es=[e for e in events if e['conversation_id']==r['conversation_id']]
        units,mapping,changes=assemble_units(r,bs,es);overrides.extend(changes)
        activity,transitions=control_timeline(r)
        for e in es:
            e['floor_unit_id']=mapping[e['floor_id']];e['incoming_unit_id']=mapping[e['incoming_id']]
            e['response_unit_id']=mapping.get(e.get('response_utterance_id'))
            if e['status']=='accepted' and e['interaction_type']=='INTERRUPT' and e['floor_response']=='YIELD':
                recovery=[u for u in units if u['speaker']==next(x['speaker'] for x in r['utterances'] if x['utterance_id']==e['floor_id']) and u['start']>=e['end'] and u['unit_id']!=e['floor_unit_id']]
                e['next_floor_unit_id']=recovery[0]['unit_id'] if recovery else None
                proven=[b for b in bs if b['left_id']==e['floor_id'] and (b.get('model_result') or {}).get('code')=='E']
                e['recovery_unit_id']=mapping[proven[0]['right_id']] if proven else None
                if proven:
                    recovered=next(u for u in units if u['unit_id']==e['recovery_unit_id'])
                    transitions.append(dict(time=recovered['start'],speaker=recovered['speaker'],action='RESUME_AFTER_INTERRUPTION',event_id=e['event_id'],kind='model_inferred_observation',policy_target=False))
        issues=validate_export(r,units,es,mapping);all_issues.extend([r['conversation_id']+':'+x for x in issues])
        conv=dict(conversation_id=r['conversation_id'],source_utterances=r['utterances'],semantic_units=units,semantic_boundaries=bs,interaction_events=es,control_activity=activity,control_transitions=sorted(transitions,key=lambda x:(x['time'],x['speaker'])),recording_end=dict(observed_end=max(u['end'] for u in r['utterances']),kind='transcript_observation_end',interruption=False),validation_issues=issues,policy_training_ready=False)
        conversations.append(conv)
        unit_by_id={u['unit_id']:u for u in units}
        for e in es:
            eligible=e['status']=='accepted' and not issues and all(unit_by_id[e[k]]['boundary_status']=='accepted' and unit_by_id[e[k]]['interaction_status']=='accepted' and not unit_by_id[e[k]]['source_quality_flags'] for k in ['floor_unit_id','incoming_unit_id'])
            example=dict(example_id=e['event_id'],conversation_id=r['conversation_id'],task='observed_duplex_annotation',input_context=context(r,e['start'],e['end'],before=15,after=0),annotation=e,future_outcome_labels=True,policy_target=False)
            # Separate future evidence used for offline outcome annotation from deployable online policy input.
            example['input_context']=causal_text_context(r,e['end'])
            if eligible:training.append(example)
            else:pending.append(dict(**example,pending_reason='event_or_boundary_uncertain',human_required=False))
        for b in bs:
            if b['decision']=='PENDING_AUTOMATIC':pending.append(dict(task='semantic_boundary',annotation=b,human_required=False))
    for name,rows in [('annotated_conversations.jsonl',conversations),('training_annotations.jsonl',training),('pending_automatic.jsonl',pending)]:
        (root/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf8')
    save('assembly_overrides.json',overrides)
    summary=dict(conversations=len(records),source_utterances=sum(len(r['utterances']) for r in records),boundaries=len(boundaries),boundary_decisions=dict(Counter(b['decision'] for b in boundaries)),semantic_units=sum(len(c['semantic_units']) for c in conversations),interaction_candidates=len(events),interaction_types=dict(Counter(e['interaction_type'] for e in events)),event_status=dict(Counter(e['status'] for e in events)),functions=dict(Counter(e['function'] for e in events)),floor_responses=dict(Counter(e['floor_response'] for e in events)),content_uptake=dict(Counter(e['content_uptake'] for e in events)),training_annotations=len(training),pending_items=len(pending),assembly_overrides=len(overrides),validation_issues=all_issues,independent_model=model,human_review_used=False,accuracy_evaluated=False,policy_training_ready=False)
    save('summary.json',summary)
    summary['independent_model']='GLM-5.3-Flash';summary['bulk_annotation_model']=model
    summary['accepted_interaction_types']=dict(Counter(e['interaction_type'] for e in events if e['status']=='accepted'))
    summary['control_transition_counts']=dict(Counter(t['action'] for c in conversations for t in c['control_transitions']))
    summary['independent_dispute_results']={x['previous_candidate_id']:x['decision'] for x in independent.values()}
    summary['boundary_consistency_pending']=len(consistency_issues)
    save('summary.json',summary)
    save('manifest.json',dict(version='duplex-e2e-v5',model=model,python=platform.python_version(),input_sha256=hashlib.sha256((root/'records.json').read_bytes()).hexdigest(),prompt_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['boundary_prompt.txt','event_prompt.txt']},stages=['independent_adjudication','semantic_boundaries','provisional_units','interaction_and_control','safe_assembly','audit','export']))
    if all_issues:raise RuntimeError('Export integrity failed: '+str(all_issues))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True);print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18351/v1');p.add_argument('--model',default='glm-judge');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    run(Path(__file__).resolve().parent,a.url,a.model,a.workers)
