"""Speaker-conditioned audio + same text context, using DuplexJev's packed readout."""
import argparse,collections,hashlib,json,os,time
from benchmark_common import ROOT,LABELS,context_text,read_inputs
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=4);a=p.parse_args()
    import torch
    from duplexjev import Decider,Question
    rows=read_inputs()
    if a.limit:rows=[e for t in LABELS for e in [x for x in rows if x['task']==t][:a.limit]]
    prefix='jev_smoke' if a.limit else 'jev'
    out=ROOT/(prefix+'_predictions.jsonl');done={}
    if out.exists():done={x['id']:x for x in map(json.loads,out.read_text().splitlines()) if x['status']=='ok'}
    model_path=ROOT.parents[1]/'models/DuplexJev-4B-Para'
    t=time.perf_counter();engine=Decider.from_pretrained(str(model_path),device='cuda:0',attn_implementation='sdpa')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-t
    print('JEV_MODEL_LOADED',load_seconds,flush=True)
    def qs(task):
        text=('For the identified FOCUS entry into the OTHER speaker\'s speech, which observed behavior applies?' if task=='overlap'
              else 'For the specified pause between LEFT and RIGHT of FOCUS, which observed behavior applies?')
        return [Question('label',text,LABELS[task],audio='*'),
                Question('turn',"What is the speaker's turn state?",['finished, ready for the other speaker','not finished, still talking','just a backchannel, not taking the turn','hesitating or asking to wait'],audio='*'),
                Question('barge_in','Is the speaker trying to interrupt or take over the conversation?',['yes','no'],audio='*')]
    def infer(batch,task):
        audios={e['id']:e['audio_clip'] for e in batch}
        contexts={e['id']:context_text(e)+'\nThe audio is the isolated FOCUS channel. Other-speaker speech is provided in the context; do not attribute it to FOCUS.' for e in batch}
        torch.cuda.synchronize();start=time.perf_counter()
        answer=engine.decide_batch(audios,qs(task),context=contexts,lang='en',mode='packed',max_items=a.batch_size)
        torch.cuda.synchronize();return answer,time.perf_counter()-start
    warm,warm_seconds=infer(rows[:1],rows[0]['task'])
    (ROOT/(prefix+'_warmup.json')).write_text(json.dumps({'seconds':warm_seconds,'answer':warm},indent=2))
    tasks=[]
    with out.open('a') as f:
        for task in LABELS:
            task_rows=[e for e in rows if e['task']==task and e['id'] not in done]
            started=time.perf_counter();results=[];batches=[]
            for offset in range(0,len(task_rows),a.batch_size):
                batch=task_rows[offset:offset+a.batch_size]
                try:
                    answers,elapsed=infer(batch,task)
                    for e in batch:
                        answer=answers[e['id']];label=answer['label']['answer'];assert label in LABELS[task]
                        probs=answer['label']['probs'];ordered=sorted(probs.values(),reverse=True)
                        result={'id':e['id'],'task':task,'label':label,'status':'ok','model':'DuplexJev-4B-Para',
                                'confidence':answer['label']['confidence'],'margin':ordered[0]-ordered[1],
                                'model_answers':answer,'batch_wall_seconds':elapsed,'batch_size':len(batch),
                                'context_sha256':hashlib.sha256(context_text(e).encode()).hexdigest()}
                        results.append(result);f.write(json.dumps(result)+'\n')
                    batches.append({'count':len(batch),'wall_seconds':elapsed})
                except Exception as ex:
                    for e in batch:
                        result={'id':e['id'],'task':task,'label':None,'status':'error','error':repr(ex)}
                        results.append(result);f.write(json.dumps(result)+'\n')
                    print('BATCH_ERROR',repr(ex),flush=True)
                    if sum(x['status']=='error' for x in results)>=12:raise
                f.flush()
                if offset%100==0:print(json.dumps({'task':task,'completed_this_run':len(results),'seconds':time.perf_counter()-started}),flush=True)
            elapsed=time.perf_counter()-started
            tasks.append({'task':task,'count':len(results),'wall_seconds':elapsed,'events_per_second':len(results)/elapsed if elapsed else 0,
                          'statuses':dict(collections.Counter(x['status'] for x in results)),
                          'labels':dict(collections.Counter(x['label'] for x in results)),'batches':batches})
            (ROOT/(prefix+'_runtime.json')).write_text(json.dumps({'model':'DuplexJev-4B-Para','gpus':1,'batch_size':a.batch_size,
                'model_load_seconds':load_seconds,'warmup_seconds':warm_seconds,'reused_prediction_count':len(done),'tasks':tasks,
                'scope':'audio loading/resampling + text context + 3 packed questions + CPU output; excludes model load and warmup',
                'accuracy_evaluated':False,'question_count':3,'input_mode':'speaker-isolated audio plus offline text/timing context'},indent=2))
    print('JEV_BENCHMARK_COMPLETE',flush=True)
if __name__=='__main__':main()
