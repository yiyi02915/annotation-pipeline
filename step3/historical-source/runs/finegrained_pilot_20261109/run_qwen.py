"""Uncached measured predictions. Repeated runs resume only completed IDs."""
import argparse,concurrent.futures,collections,hashlib,json,time,urllib.request,urllib.error
from pathlib import Path
from benchmark_common import ROOT,DEFINITIONS,LABELS,payload,read_inputs
URL='http://127.0.0.1:18352'
def call(e):
    labels=LABELS[e['task']]
    schema={'type':'object','properties':{'label':{'type':'string','enum':labels}},'required':['label'],'additionalProperties':False}
    body={'model':'qwen38-27b','messages':[{'role':'system','content':DEFINITIONS+'\nReturn only a JSON object with one key: label.'},
        {'role':'user','content':json.dumps(payload(e),ensure_ascii=False,separators=(',',':'))}],
        'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False},
        'response_format':{'type':'json_schema','json_schema':{'name':'event_label','strict':True,'schema':schema}}}
    started=time.perf_counter();attempts=[]
    for attempt in range(2):
        t=time.perf_counter()
        try:
            req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=240) as response:answer=json.load(response)
            raw=answer['choices'][0]['message']['content'];result=json.loads(raw)
            assert result['label'] in labels
            return {'id':e['id'],'task':e['task'],'label':result['label'],'status':'ok',
                    'elapsed_seconds':time.perf_counter()-started,'attempts':attempts,'raw':raw,'usage':answer.get('usage'),
                    'finish_reason':answer['choices'][0]['finish_reason'],'model':'Qwen3.8-27B',
                    'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
        except Exception as ex:
            detail=ex.read().decode()[:800] if isinstance(ex,urllib.error.HTTPError) else str(ex)
            attempts.append({'error':detail,'seconds':time.perf_counter()-t})
    return {'id':e['id'],'task':e['task'],'label':None,'status':'error','attempts':attempts,'elapsed_seconds':time.perf_counter()-started}
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--concurrency',type=int,default=8);a=p.parse_args()
    rows=read_inputs()
    if a.limit:
        rows=[e for task in LABELS for e in [x for x in rows if x['task']==task][:a.limit]]
    prefix='qwen_smoke' if a.limit else 'qwen'
    out=ROOT/(prefix+'_predictions.jsonl');done={}
    if out.exists():done={x['id']:x for x in map(json.loads,out.read_text().splitlines()) if x['status']=='ok'}
    waiting=time.time()
    while True:
        try:
            with urllib.request.urlopen(URL+'/v1/models',timeout=4) as r:assert r.status==200
            break
        except Exception:
            if time.time()-waiting>3600:raise TimeoutError('Model service unavailable after 1 hour')
            time.sleep(5)
    # Warm-up is separately measured and excluded from steady-state throughput.
    warm_t=time.perf_counter();warm=call(rows[0]);warm_seconds=time.perf_counter()-warm_t
    assert warm['status']=='ok',warm
    (ROOT/(prefix+'_warmup.json')).write_text(json.dumps(warm,indent=2))
    runs=[]
    with out.open('a') as f:
        for task in LABELS:
            task_rows=[e for e in rows if e['task']==task and e['id'] not in done]
            start=time.perf_counter();results=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                for result in pool.map(call,task_rows):
                    f.write(json.dumps(result)+'\n');f.flush();results.append(result)
                    if len(results)%50==0:print(json.dumps({'task':task,'completed_this_run':len(results),'seconds':time.perf_counter()-start}),flush=True)
            elapsed=time.perf_counter()-start
            runt=dict(task=task,count=len(results),wall_seconds=elapsed,events_per_second=len(results)/elapsed if elapsed else 0,
                statuses=dict(collections.Counter(x['status'] for x in results)),labels=dict(collections.Counter(x['label'] for x in results)),
                latencies_seconds=[x['elapsed_seconds'] for x in results])
            runs.append(runt)
            (ROOT/(prefix+'_runtime.json')).write_text(json.dumps({'model':'Qwen3.8-27B','gpus':1,'concurrency':a.concurrency,
                'warmup_seconds':warm_seconds,'reused_prediction_count':len(done),'tasks':runs,
                'scope':'HTTP wall time including input handling, tokenization, model prefill/generation, structured decoding; excludes startup and warm-up',
                'accuracy_evaluated':False},indent=2))
    print('QWEN_BENCHMARK_COMPLETE',flush=True)
if __name__=='__main__':main()
