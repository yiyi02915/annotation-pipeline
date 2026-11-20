import time,json,urllib.request,urllib.error,concurrent.futures,hashlib,argparse
from evidence import *
pa=argparse.ArgumentParser();pa.add_argument('--shard',type=int,default=-1);args=pa.parse_args();port=18352+max(args.shard,0)
schema={'type':'object','properties':{'label':{'type':'string','enum':LABELS},'left_state':{'type':'string','enum':['unfinished','completed_or_yielded','unclear']},'other_action':{'type':'string','enum':['substantive_turn','feedback_only','none','interrupting','unclear']},'focus_continuation':{'type':'string','enum':['same_contribution','new_turn','unclear']},'evidence_ids':{'type':'array','items':{'type':'string'}},'reason':{'type':'string'}},'required':['label','left_state','other_action','focus_continuation','evidence_ids','reason'],'additionalProperties':False}
def call(item):
 e,p=item;t=time.time();body={'model':'qwen38-27b','messages':[{'role':'system','content':DEFINITIONS+'\n'+EXTRA+'\nReturn JSON. Cite actual segment_id values supporting the conclusion; do not cite silence alone. Explain the pause using left_state, other_action and focus_continuation.'},{'role':'user','content':json.dumps(p)}],'temperature':0,'max_tokens':4096,'chat_template_kwargs':{'enable_thinking':True},'reasoning_effort':'medium','response_format':{'type':'json_schema','json_schema':{'name':'rejudge','strict':True,'schema':schema}}}
 try:
  req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=900) as res:ans=json.load(res)
  msg=ans['choices'][0];v=json.loads(msg['message']['content']);assert v['label'] in LABELS
  valid={s['segment_id'] for s in p['context']};refs=v['evidence_ids'];v['evidence_ids_valid']=bool(refs) and all(s in valid for s in refs)
  return {'id':e['id'],'status':'ok','evidence_version':2,'judgment':v,'seconds':time.time()-t,'finish_reason':msg['finish_reason'],'usage':ans.get('usage'),'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
 except Exception as ex:return {'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t}
out=D/('qwen_rejudge.jsonl' if args.shard<0 else f'qwen_rejudge_shard{args.shard}.jsonl');started=time.time()
selected=[e['id'] for i,e in enumerate(read(D/'inputs.jsonl')) if args.shard<0 or i%4==args.shard];selected=set(selected)
affected={e['id'] for e in read(D/'inputs.jsonl') if any(not norm(tr['text']) for tr in e['tracks'].values())}
while True:
 done={e['id'] for e in read(out)+read(D/'qwen_rejudge.jsonl') if e.get('status')=='ok' and (e['id'] not in affected or e.get('evidence_version')==2)};batch=[x for x in ready() if x[0]['id'] in selected and x[0]['id'] not in done]
 try:urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3).close()
 except Exception:time.sleep(10);continue
 if batch:
  with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
   for r in pool.map(call,batch):f.write(json.dumps(r)+'\n');f.flush();print('QWEN',len(done)+1,r['status'],flush=True);done.add(r['id'])
 if selected<=done:break
 if time.time()-started>14400:raise TimeoutError('Incomplete inputs')
 time.sleep(10)
print('QWEN_COMPLETE',flush=True)
