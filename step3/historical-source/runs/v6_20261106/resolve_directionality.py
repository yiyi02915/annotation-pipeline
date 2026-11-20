"""Resolve actor/target reversal without inventing missing future responses."""
import json,copy,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from run_v6 import event_payload
from pipeline_core import validate_event
from llm_client import Client
root=Path(__file__).resolve().parent
load=lambda n:json.loads((root/n).read_text(encoding='utf8'))
events=load('interaction_results.json');candidates={c['candidate_id']:c for c in load('event_inputs.json')}
units={sid:u for u in load('provisional_units.json') for sid in u['source_utterance_ids']}
client=Client(root,'http://127.0.0.1:18351/v1','glm-judge')
prompt=load('effective_prompts.json')['event']
selection=[e for e in events if e.get('validation_error')=='missing_uptake_evidence' and not e.get('input_quality_flags')]
if not selection:
    print('No directionality failures remain; preserving prior recheck evidence.');sys.exit(0)
before=copy.deepcopy(selection)
def one(e):
    c=candidates[e['event_id']];p,schema=event_payload(c,units)
    actor=c['floor']['speaker'];target=c['incoming']['speaker']
    p['directional_roles']=dict(original_speaker=actor,incoming_speaker=target,content_uptake_question='Did '+actor+' subsequently handle the content of '+target+"'s incoming contribution?",reverse_direction_is_not_uptake=target+' answering '+actor+"'s earlier question is the incoming action, not content_uptake.")
    absent=p['allowed_response_context_indices']==[-1] and c['floor']['end']<=c['incoming']['start']
    if absent:
        p['observable_constraint']='The original speaker has no speech at or after incoming onset in the supplied context. Their reaction and content uptake are unobserved.'
        schema['properties']['content_uptake']={'type':'string','enum':['UNKNOWN']}
        schema['properties']['floor_response']={'type':'string','enum':['UNKNOWN']}
    system=prompt+'\nKeep speaker roles fixed. The incoming speaker answering the original speaker is NOT evidence that the original speaker responded to the incoming speaker. An ordinary turn following a completed question is not evidence of a reactive pause caused by the answer. Missing original-speaker response means UNKNOWN outcome fields; still classify the incoming interaction if supported.'
    trace=[]
    def call(role,extra,payload=p):
        rr=client.call(system+'\n'+extra,payload,schema,e['event_id']+':direction:'+role)
        raw=rr['parsed'];error=validate_event(c,raw)
        trace.append(dict(role='direction_'+role,result=raw,error=error,request_hash=rr['request_hash']))
        return raw,error
    a,ea=call('judge','Independently classify the incoming action and the original speaker\'s subsequent reaction.')
    b,eb=call('audit','Independently check actor direction and whether any original-speaker response is actually observable.')
    keys=['interaction_type','function','floor_completion','floor_response','content_uptake']
    if ea or eb or any(a[k]!=b[k] for k in keys):
        pp=copy.deepcopy(p);pp['conflicting_analyses']=[a,b]
        a,ea=call('arbiter','Resolve the different interpretations using the fixed speaker roles and observable evidence.',pp)
    e.setdefault('judgments',[]).extend(trace)
    e['directionality_recheck']=dict(no_original_speaker_response=absent,roles=p['directional_roles'])
    if not ea:
        e.update(a);e['status']='accepted' if a['interaction_type']!='UNRESOLVED' else 'PENDING_AUTOMATIC';e['validation_error']=None
        ix=a['response_context_index'];e['response_utterance_id']=c['context'][ix]['utterance_id'] if ix>=0 else None
        e['request_hash']=trace[-1]['request_hash'];e['v6_origin']='glm_v6_direction_recheck'
        e['incoming_control_action']={'BACKCHANNEL':'BRIEF_FEEDBACK','INTERRUPT':'TAKE_FLOOR_ATTEMPT','TURN':'TAKE_TURN','OTHER':'OTHER','UNRESOLVED':'UNKNOWN'}[e['interaction_type']]
        e['floor_control_action']={'CONTINUE':'CONTINUE_SPEAKING','YIELD':'YIELD_FLOOR','UNKNOWN':'UNKNOWN'}[e['floor_response']]
    print(e['event_id'],e['status'],e.get('floor_response'),e.get('content_uptake'),flush=True)
    return e
with ThreadPoolExecutor(max_workers=2) as pool:updated=list(pool.map(one,selection))
by_id={e['event_id']:e for e in updated}
events=[by_id.get(e['event_id'],e) for e in events]
for n,value in [('interaction_results.json',events),('directionality_recheck.json',dict(before=before,after=updated))]:
    (root/n).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
