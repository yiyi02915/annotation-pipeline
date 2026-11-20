import concurrent.futures,hashlib,json,time,sys,urllib.request,urllib.error
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent/'finegrained_pilot_20261109';sys.path.insert(0,str(B))
from benchmark_common import DEFINITIONS,LABELS,payload
URL='http://127.0.0.1:18352'
CONFIG=json.loads((D/'annotation_config.candidate.json').read_text()) if (D/'annotation_config.candidate.json').exists() else None
def call(e,thinking):
 schema={'type':'object','properties':{'label':{'type':'string','enum':LABELS[e['task']]}},'required':['label'],'additionalProperties':False}
 body=dict(model='qwen38-27b',messages=[dict(role='system',content=DEFINITIONS+'\nReturn only a JSON object with one key: label.'),dict(role='user',content=json.dumps(payload(e),ensure_ascii=False,separators=(',',':')))],temperature=0,max_tokens=3072 if thinking else 64,chat_template_kwargs={'enable_thinking':thinking},response_format={'type':'json_schema','json_schema':{'name':'event_label','strict':True,'schema':schema}})
 if thinking:body['reasoning_effort']='medium'
 if CONFIG:body.update(CONFIG['qwen']['thinking_candidate' if thinking else 'baseline'])
 t=time.perf_counter()
 try:
  req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=600) as res:a=json.load(res)
  msg=a['choices'][0]['message'];result=json.loads(msg['content']);assert result['label'] in LABELS[e['task']]
  return dict(id=e['id'],task=e['task'],split=e['study_split'],label=result['label'],status='ok',seconds=time.perf_counter()-t,usage=a.get('usage'),finish_reason=a['choices'][0]['finish_reason'],reasoning_chars=len(msg.get('reasoning') or msg.get('reasoning_content') or ''),request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest())
 except Exception as ex:
  detail=ex.read().decode()[:1000] if isinstance(ex,urllib.error.HTTPError) else str(ex)
  return dict(id=e['id'],task=e['task'],status='error',error=detail,seconds=time.perf_counter()-t)
def main():
 rows=[json.loads(s) for s in (D/'study_inputs.jsonl').read_text().splitlines()]
 t=time.time()
 while True:
  try:
   urllib.request.urlopen(URL+'/v1/models',timeout=3).close();break
  except Exception:
   if time.time()-t>2400:raise TimeoutError('Server unavailable')
   time.sleep(5)
 runtime=[]
 for thinking in (False,True):
  name='thinking' if thinking else 'baseline';out=D/f'qwen_{name}.jsonl';done=set()
  if out.exists():done={x['id'] for x in map(json.loads,out.read_text().splitlines()) if x['status']=='ok'}
  warm=call(rows[0],thinking);(D/f'qwen_{name}_warmup.json').write_text(json.dumps(warm,indent=2));print('WARMUP',name,warm,flush=True)
  if warm['status']!='ok':raise RuntimeError(warm)
  with out.open('a') as f:
   for task in LABELS:
    subset=[e for e in rows if e['task']==task and e['id'] not in done];t=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
     for i,res in enumerate(pool.map(lambda e:call(e,thinking),subset)):
      f.write(json.dumps(res)+'\n');f.flush()
      if (i+1)%20==0:print(name,task,i+1,flush=True)
    runtime.append(dict(variant=name,task=task,n=len(subset),seconds=time.perf_counter()-t))
   (D/'qwen_study_runtime.json').write_text(json.dumps(runtime,indent=2))
 print('QWEN_STUDY_COMPLETE',flush=True)
if __name__=='__main__':main()
