import json,time,urllib.request,concurrent.futures
from refine import D,read,RULES,LABELS,ROLES
enums={'label':LABELS,'focus_before_role':['contributor','listener_feedback','unclear'],'left_state':['unfinished','completed_or_yielded','unclear'],'other_action':ROLES,'focus_continuation':['same_contribution','new_turn','listener_feedback','unclear'],'pause_function':['planning_or_repair','turn_exchange','listening_between_feedback','breath_or_rhetorical','interruption_suspension','unclear']}
props={k:{'type':'string','enum':v} for k,v in enums.items()};props.update({'evidence_ids':{'type':'array','items':{'type':'string'}},'target_explanation':{'type':'string'},'other_function_explanation':{'type':'string'}});schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
events={e['id']:e for e in read(D/'events.jsonl')};done={x['id'] for x in read(D/'qwen_repair.jsonl') if x['status']=='ok'}
def call(item):
 shard,old=item;e=events[old['id']];t=time.time();body={'model':'qwen38-27b','messages':[{'role':'system','content':RULES+'\nReturn structured facts and label directly. Cite actual context segment_id evidence. Explain the specific LEFT-to-RIGHT pause and OTHER function. Ensure facts support label.'},{'role':'user','content':json.dumps(e['payload'])}],'temperature':0,'max_tokens':1536,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'target_gap','strict':True,'schema':schema}}}
 try:
  req=urllib.request.Request(f'http://127.0.0.1:{18452+shard}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=900) as response:a=json.load(response)
  c=a['choices'][0];v=json.loads(c['message']['content']);return {'id':e['id'],'status':'ok','judgment':v,'finish_reason':c['finish_reason'],'seconds':time.time()-t,'usage':a.get('usage'),'repair':'schema_output_recovery_same_model_not_independent_vote','original_output':old}
 except Exception as ex:return {'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t,'original_output':old}
with (D/'qwen_repair.jsonl').open('a') as f:
 while True:
  jobs=[(k,x) for k in range(2) for x in read(D/f'qwen_{k}.jsonl') if (x['status']!='ok' or x.get('finish_reason')!='stop') and x['id'] not in done]
  if jobs:
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for z in pool.map(call,jobs):f.write(json.dumps(z)+'\n');f.flush();done.add(z['id']);print('RECOVER',len(done),z['status'],flush=True)
  if all('QWEN_COMPLETE' in (D/f'qwen_{k}.log').read_text() for k in range(2)) and not jobs:break
  time.sleep(10)
print('RECOVERY_COMPLETE',flush=True)
