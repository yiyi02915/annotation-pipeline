import concurrent.futures,json,time,urllib.request,urllib.error,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
C=json.loads((R/'baseline_config.json').read_text())
E=json.loads((R/'inputs.json').read_text())
assert E and len({e['id'] for e in E})==len(E)
assert {e['id'] for e in E}==set(json.loads((R/'references.json').read_text())['labels'])
def evaluate(e,thinking):
    body={'model':'qwen38-27b','messages':[{'role':'system','content':C['rules']+'\n'+C['extra']+' Evaluate the marked gap only. Keep reasoning focused; return the required JSON.'},{'role':'user','content':json.dumps(e['payload'])}], 'temperature':1.0 if thinking else 0,'chat_template_kwargs':{'enable_thinking':thinking},'response_format':{'type':'json_schema','json_schema':{'name':'gap_decision','strict':True,'schema':C['schema']}}}
    if thinking:body.update(top_p=.95,top_k=20,presence_penalty=1.5,seed=42)
    attempts=[]
    start=time.monotonic()
    for budget in ((3072,3072,6144) if thinking else (1024,1024)):
        body['max_tokens']=budget
        t=time.monotonic()
        record={'budget':budget,'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
        attempts.append(record)
        try:
            req=urllib.request.Request('http://127.0.0.1:18770/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=600) as f:ans=json.load(f)
            ch=ans['choices'][0]; record.update(response=ans,seconds=time.monotonic()-t)
            if ch['finish_reason']!='stop':raise ValueError('finish_reason='+str(ch['finish_reason']))
            v=json.loads(ch['message']['content']);label=v['label']
            if label not in ('PAUSE','TURN_TRANSITION'):raise ValueError('invalid label')
            return dict(id=e['id'],status='ok',label=label,analysis=v,seconds=time.monotonic()-start,attempts=attempts,budget_repair=budget==6144)
        except Exception as exc:
            record.update(error=str(exc),seconds=time.monotonic()-t)
    return dict(id=e['id'],status='error',seconds=time.monotonic()-start,attempts=attempts)
if __name__=='__main__':
    for method in ('baseline','thinking'):
        t=time.monotonic()
        with (R/(method+'.jsonl')).open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(evaluate,e,method=='thinking') for e in E]
            for future in concurrent.futures.as_completed(jobs):
                row=future.result();f.write(json.dumps(row)+'\n');f.flush()
                print(method,row['id'],row['status'],row.get('budget_repair'),flush=True)
        print(method,'wall_seconds',round(time.monotonic()-t,3),flush=True)
    print('COMPLETE',flush=True)
