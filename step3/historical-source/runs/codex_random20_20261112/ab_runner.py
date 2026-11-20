import json,time,urllib.request,hashlib,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent
base=json.loads((D/'baseline_config.json').read_text())
def go(e,version,cohort):
 rules=base['rules']+'\n'+base['extra']+' Return concise JSON.'
 if version=='optimized':rules+='\n'+(D/'prompt_addendum.txt').read_text()
 body={'model':'qwen38-27b','messages':[{'role':'system','content':rules},{'role':'user','content':json.dumps(e['payload'])}],'temperature':0,'max_tokens':1536,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'semantic_gap_binary_v2','strict':True,'schema':base['schema']}}}
 t=time.time();errors=[]
 for _ in range(3):
  try:
   req=urllib.request.Request('http://127.0.0.1:18770/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=300) as f:r=json.load(f)
   c=r['choices'][0];v=json.loads(c['message']['content']);assert c['finish_reason']=='stop'
   return dict(id=e['id'],version=version,cohort=cohort,judgment=v,response=r,seconds=time.time()-t,status='ok',request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest())
  except Exception as ex:errors.append(str(ex))
 return dict(id=e['id'],version=version,cohort=cohort,status='error',errors=errors)
deadline=time.time()+1800
while True:
 try:urllib.request.urlopen('http://127.0.0.1:18770/v1/models',timeout=3).close();break
 except Exception:
  if time.time()>deadline:raise TimeoutError('server not ready')
  time.sleep(5)
jobs=[(e,v,c) for c,fn in [('development','blind_inputs.json'),('holdout','holdout_inputs.json')] for e in json.loads((D/fn).read_text()) for v in ['baseline','optimized']]
with (D/'ab_results.jsonl').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(lambda args:go(*args),jobs):f.write(json.dumps(r)+'\n');f.flush();print(r['cohort'],r['version'],r['status'],flush=True)
print('COMPLETE',flush=True)
