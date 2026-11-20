import json,time,urllib.request,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent
es={e['id']:e for e in json.loads((D/'inputs.json').read_text())};cfg=json.loads((D/'baseline_config.json').read_text())
deadline=time.time()+1200
while 'COMPLETE' not in (D/'thinking_fresh.log').read_text():
 if time.time()>deadline:raise TimeoutError('fresh wait')
 time.sleep(5)
source=[json.loads(s) for s in (D/'thinking_fresh.jsonl').read_text().splitlines()]
jobs=[r for r in source if r['status']!='ok']
def repair(item):
 e=es[item['id']];start=time.time();body={'model':'qwen38-27b','messages':[{'role':'system','content':cfg['rules']+'\n'+cfg['extra']+' Evaluate the marked gap only. Keep reasoning focused; return the required JSON.'},{'role':'user','content':json.dumps(e['payload'])}],'temperature':1.0,'top_p':.95,'top_k':20,'presence_penalty':1.5,'seed':42,'max_tokens':6144,'chat_template_kwargs':{'enable_thinking':True},'response_format':{'type':'json_schema','json_schema':{'name':'gap_decision','strict':True,'schema':cfg['schema']}}}
 try:
  req=urllib.request.Request('http://127.0.0.1:18770/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=600) as f:r=json.load(f)
  c=r['choices'][0];v=json.loads(c['message']['content']);assert c['finish_reason']=='stop'
  return dict(id=e['id'],method='thinking',status='ok',label=v['label'],analysis=v,seconds=item['seconds']+time.time()-start,repair_seconds=time.time()-start,attempts=item['attempts']+[r],budget_repair=6144)
 except Exception as ex:return dict(item,budget_repair_error=str(ex))
with (D/'budget_repairs.jsonl').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(repair,jobs):f.write(json.dumps(r)+'\n');f.flush();print(r['id'],r['status'],flush=True)
print('COMPLETE',len(jobs),flush=True)
