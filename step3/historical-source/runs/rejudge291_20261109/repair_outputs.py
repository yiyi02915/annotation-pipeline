import time,json,urllib.request,concurrent.futures,hashlib
from evidence import *
props={'label':{'type':'string','enum':LABELS},'left_state':{'type':'string','enum':['unfinished','completed_or_yielded','unclear']},'other_action':{'type':'string','enum':['substantive_turn','feedback_only','none','interrupting','unclear']},'focus_continuation':{'type':'string','enum':['same_contribution','new_turn','unclear']},'evidence_ids':{'type':'array','items':{'type':'string'}},'reason':{'type':'string'}}
schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
while not all('QWEN_COMPLETE' in (D/f'qwen_shard{k}_v2.log').read_text() for k in range(4)):time.sleep(10)
q={e['id']:e for e in read(D/'qwen_rejudge.jsonl')+[x for k in range(4) for x in read(D/f'qwen_rejudge_shard{k}.jsonl')]}
def conflict(r):
 v=r.get('judgment',{});l=v.get('label')
 return r.get('status')!='ok' or r.get('finish_reason')!='stop' or not v.get('evidence_ids_valid') or (l=='HESITATION' and (v.get('left_state')!='unfinished' or v.get('focus_continuation')!='same_contribution')) or (l=='TURN_TRANSITION' and (v.get('left_state')!='completed_or_yielded' or v.get('focus_continuation')!='new_turn' or v.get('other_action')!='substantive_turn'))
rows=[(i,e,p,q[e['id']]) for i,(e,p) in enumerate(ready()) if conflict(q[e['id']])];out=D/'qwen_output_repair.jsonl';done={e['id'] for e in read(out)};rows=[x for x in rows if x[1]['id'] not in done]
def call(item):
 i,e,p,previous=item;port=18353+i%3;t=time.time()
 instructions='''Reassess the event from its evidence and repair any contradiction in a previous structured answer. LEFT_STATE refers to whether the FOCUS conversational contribution is still in progress, NOT merely whether a clause is grammatical. HESITATION requires unfinished + same_contribution. TURN_TRANSITION requires completed_or_yielded + new_turn and evidence of actual turn progression. Brief listener feedback alone is insufficient. If an earlier answer describes a within-turn continuation yet marks completed_or_yielded, determine which is supported by the transcript and fix it. You may change the label; never repair fields solely to make them fit a label. Cite real segment IDs covering the before and after context. If insufficient, use UNCERTAIN. Return concise JSON, reason within 120 words.'''
 body={'model':'qwen38-27b','messages':[{'role':'system','content':DEFINITIONS+'\n'+EXTRA+'\n'+instructions},{'role':'user','content':json.dumps({'event':p,'previous_answer':previous.get('judgment'),'previous_error':previous.get('error')})}],'temperature':0,'max_tokens':1200,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'repair','strict':True,'schema':schema}}}
 try:
  req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=600) as res:ans=json.load(res)
  choice=ans['choices'][0];v=json.loads(choice['message']['content']);refs={x['segment_id'] for x in p['context']};v['evidence_ids_valid']=bool(v['evidence_ids']) and all(x in refs for x in v['evidence_ids'])
  return {'id':e['id'],'status':'ok','judgment':v,'evidence_version':2,'finish_reason':choice['finish_reason'],'repair_of':previous.get('judgment'),'repair_mode':'nonthinking_consistency_check_after_thinking','seconds':time.time()-t,'usage':ans.get('usage'),'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
 except Exception as ex:return {'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t}
with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for r in pool.map(call,rows):f.write(json.dumps(r)+'\n');f.flush();print('REPAIRED',r['id'],r['status'],flush=True)
print('OUTPUT_REPAIR_COMPLETE',flush=True)
