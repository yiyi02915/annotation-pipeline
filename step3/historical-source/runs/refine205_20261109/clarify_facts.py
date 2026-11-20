import json,time,urllib.request,concurrent.futures
from refine import D,read,RULES,LABELS,ROLES
enums={'label':LABELS,'focus_before_role':['contributor','listener_feedback','unclear'],'left_state':['unfinished','completed_or_yielded','unclear'],'other_action':ROLES,'focus_continuation':['same_contribution','new_turn','listener_feedback','unclear'],'pause_function':['planning_or_repair','turn_exchange','listening_between_feedback','breath_or_rhetorical','interruption_suspension','unclear']}
props={k:{'type':'string','enum':v} for k,v in enums.items()};props.update({'evidence_ids':{'type':'array','items':{'type':'string'}},'target_explanation':{'type':'string'},'other_function_explanation':{'type':'string'}})
props['left_state']['description']='Status of the FOCUS conversational contribution, NOT grammatical completeness of its last sentence. unfinished means the same contribution remains open across the gap; completed_or_yielded means that contribution ended or floor was yielded.'
props['focus_continuation']['description']='listener_feedback if RIGHT begins another backchannel while OTHER holds the floor; new_turn requires a real floor exchange; same_contribution can cross grammatical sentence boundaries.'
props['pause_function']['description']='Function of this exact gap. Listening between two backchannels is listening_between_feedback, not planning_or_repair. An unfinished conversational contribution can contain several complete sentences.'
schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
def consistent(v):
 label=v.get('label')
 if label=='HESITATION':return v.get('focus_before_role')=='contributor' and v.get('left_state')=='unfinished' and v.get('focus_continuation')=='same_contribution' and v.get('pause_function')=='planning_or_repair' and v.get('other_action') in ['feedback_only','none']
 if label=='TURN_TRANSITION':return v.get('focus_before_role')=='contributor' and v.get('left_state')=='completed_or_yielded' and v.get('focus_continuation')=='new_turn' and v.get('pause_function')=='turn_exchange' and v.get('other_action')=='substantive_turn'
 if label=='OTHER':return (v.get('pause_function')=='listening_between_feedback' and v.get('focus_before_role')=='listener_feedback' and v.get('focus_continuation')=='listener_feedback' and v.get('other_action')=='substantive_turn') or (v.get('pause_function')=='breath_or_rhetorical' and v.get('focus_continuation')=='same_contribution') or (v.get('pause_function')=='interruption_suspension' and v.get('other_action')=='interrupting')
 return True
events={e['id']:e for e in read(D/'events.jsonl')};done={x['id'] for x in read(D/'qwen_fact_clarification.jsonl')}
def call(item):
 shard,old=item;e=events[old['id']];t=time.time();body={'model':'qwen38-27b','messages':[{'role':'system','content':RULES+'\nIMPORTANT FIELD DEFINITIONS: left_state is whether the conversational contribution remains open, not whether the last sentence is grammatical. Multiple complete sentences can be part of ONE unfinished contribution. An isolated backchannel is listener_feedback, not a substantive contributor turn. For two listener acknowledgments during OTHER continuous talk, use listener_feedback continuation and listening_between_feedback function. Evaluate facts independently from the raw event; do not infer planning solely from silence or topic continuity. Cite actual segment_id evidence and explain the exact target and OTHER function. Return consistent JSON.'},{'role':'user','content':json.dumps(e['payload'])}],'temperature':0,'max_tokens':2048,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'target_gap_precise_fields','strict':True,'schema':schema}}}
 try:
  req=urllib.request.Request(f'http://127.0.0.1:{18452+shard}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=900) as response:a=json.load(response)
  c=a['choices'][0];v=json.loads(c['message']['content']);ov=old['judgment'];return {'id':e['id'],'status':'ok','judgment':v,'finish_reason':c['finish_reason'],'seconds':time.time()-t,'usage':a.get('usage'),'repair':'fresh_blind_same_model_with_disambiguated_field_definitions','prior_judgment':ov,'label_changed':v['label']!=ov['label'],'other_role_changed':v['other_action']!=ov['other_action']}
 except Exception as ex:return {'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t}
with (D/'qwen_fact_clarification.jsonl').open('a') as f:
 while True:
  originals={x['id']:(k,x) for k in range(2) for x in read(D/f'qwen_{k}.jsonl')};originals.update({x['id']:(0,x) for x in read(D/'qwen_repair.jsonl') if x['status']=='ok'})
  jobs=[(k,x) for k,x in originals.values() if x['status']=='ok' and x.get('finish_reason')=='stop' and not consistent(x['judgment']) and x['id'] not in done]
  if jobs:
   with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for z in pool.map(call,jobs):f.write(json.dumps(z)+'\n');f.flush();done.add(z['id']);print('CLARIFY',len(done),z['status'],flush=True)
  finished=all('QWEN_COMPLETE' in (D/f'qwen_{k}.log').read_text() for k in range(2)) and 'RECOVERY_COMPLETE' in (D/'repair_outputs.log').read_text()
  if finished and not jobs:break
  time.sleep(10)
print('CLARIFICATION_COMPLETE',flush=True)
