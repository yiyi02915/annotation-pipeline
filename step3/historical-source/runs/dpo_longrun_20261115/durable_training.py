"""Detached, resumable DPO experiment queue. No Git/network publication."""
import argparse, collections, contextlib, datetime, fcntl, hashlib, json, math, os
from pathlib import Path
import random, signal, subprocess, sys, threading, time, traceback
ROOT=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    temp.replace(path)
def journal(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(dict(time=now(),**value),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def alive(pid,job=None):
    try:
        data=Path('/proc')/str(pid)/'cmdline'
        return data.exists() and (job is None or job.encode() in data.read_bytes())
    except OSError:return False
def lock(path):
    f=Path(path).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def load_rows(split):return [json.loads(l) for l in (ROOT/'data'/(split+'.jsonl')).read_text().splitlines()]
def dataset_signature():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*.json*'))}
def evaluate_model(model,fmt,rows):
    predictions=[]
    for row in rows:
        labels=fmt.OPTIONS[row['task']]
        code,scores=fmt.choose(model,fmt.prompt(row),[fmt.CODES[l] for l in labels])
        predicted=next(l for l in labels if fmt.CODES[l]==code)
        predictions.append(dict(id=row['id'],task=row['task'],reference=row['label'],predicted=predicted,scores=scores))
    recalls={l:sum(r['predicted']==l for r in predictions if r['reference']==l)/sum(r['reference']==l for r in predictions) for l in fmt.CODES}
    return dict(correct=sum(r['reference']==r['predicted'] for r in predictions),total=len(predictions),recalls=recalls,
                balanced_accuracy=sum(recalls.values())/len(recalls),predictions=predictions)
def train(job_id,pause_after=0):
    import gc,torch
    import torch.nn.functional as F
    from duplexjev import Decider
    from peft import PeftModel
    import model_format as fmt
    plan=read(ROOT/'plan.json');cfg=next(j for j in plan['jobs'] if j['id']==job_id)
    out=ROOT/'jobs'/job_id;out.mkdir(parents=True,exist_ok=True)
    guard=lock(out/'worker.lock');begin=time.monotonic();stop=[False]
    atomic(out/'status.json',dict(status='loading',pid=os.getpid(),time=now()))
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signature=dataset_signature();cfg_hash=hashlib.sha256(json.dumps(dict(config=cfg,data=signature),sort_keys=True).encode()).hexdigest()
    atomic(out/'config.json',dict(config=cfg,data_hashes=signature,config_hash=cfg_hash))
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    train_rows=load_rows('train');val_rows=load_rows('validation')
    assert not {r['conversation_id'] for r in train_rows}&{r['conversation_id'] for r in val_rows}
    baseline=read(ROOT/'data/baseline_metrics.json')
    floor=baseline['sft']['validation']['recalls']
    init=plan['adapters'][cfg['init']]
    cache_path=ROOT/'cache'/('reference_'+cfg['init']+'.json')
    latest=read(out/'latest.json') if (out/'latest.json').exists() else None
    if latest:
        state_path=out/latest['path']/'state.pt';assert cache_path.exists(),'Resume requires fixed-reference cache'
        adapter=str(out/latest['path']/'adapter')
    else:adapter=init
    engine=Decider.from_pretrained(plan['base'],device='cuda:0',attn_implementation='sdpa')
    fmt.tokenizer=engine.tok;base=engine.model.language_model
    engine.model.language_model=None;del engine;gc.collect();torch.cuda.empty_cache()
    model=PeftModel.from_pretrained(base,adapter,is_trainable=True);model.config.use_cache=False
    assert all(len(fmt.tokenizer(c,add_special_tokens=False)['input_ids'])==1 for c in fmt.CODES.values())
    cache_path.parent.mkdir(exist_ok=True)
    cache_guard=cache_path.with_suffix('.lock').open('a');fcntl.flock(cache_guard,fcntl.LOCK_EX)
    if cache_path.exists():
        cache=read(cache_path);assert cache['data_hashes']==signature
    else:
        assert latest is None
        entries=[]
        for row in train_rows:
            labels=fmt.OPTIONS[row['task']];prefix=fmt.prompt(row)
            _,scores=fmt.choose(model,prefix,[fmt.CODES[l] for l in labels])
            scores=dict(zip(labels,scores))
            for wrong in labels:
                if wrong!=row['label']:entries.append(dict(id=row['id'],chosen=row['label'],rejected=wrong,prompt=prefix,
                    reference_margin=scores[row['label']]-scores[wrong]))
        cache=dict(data_hashes=signature,pairs=entries);atomic(cache_path,cache)
    fcntl.flock(cache_guard,fcntl.LOCK_UN);cache_guard.close()
    pairs=cache['pairs'];assert len(pairs)==len({(p['id'],p['chosen'],p['rejected']) for p in pairs})
    teacher=read(ROOT/'data/teacher_targeted.json')['pairs'];hard=read(ROOT/'data/hard_pairs.json')
    checked={(r['id'],r['rejected']) for r in hard}
    for r in pairs:
        r['weight']=1.;r['teacher_status']='not_used'
        if cfg['teacher'] and (r['id'],r['rejected']) in checked:
            status=teacher[r['id']]['status'];r['teacher_status']=status
            r['weight']={'verified':1.,'contradicted':.5,'unstable':.75}[status]
    mass={l:sum(r['weight'] for r in pairs if r['chosen']==l) for l in fmt.CODES}
    for r in pairs:r['weight']*=len(pairs)/(len(mass)*mass[r['chosen']])
    atomic(out/'pair_audit.json',dict(unique_inputs=len(train_rows),unique_pairs=len(pairs),
        class_weight_mass={l:sum(r['weight'] for r in pairs if r['chosen']==l) for l in fmt.CODES},
        teacher_status=dict(collections.Counter(r['teacher_status'] for r in pairs))))
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg['lr'])
    if latest:
        state=torch.load(state_path,map_location='cpu',weights_only=False);assert state['config_hash']==cfg_hash
        optimizer.load_state_dict(state['optimizer']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);cursor=state['cursor']
        journal(out/'events.jsonl',dict(event='resumed',step=cursor['step'],checkpoint=latest['path']))
    else:
        v=evaluate_model(model,fmt,val_rows)
        expected=baseline[cfg['init']]['validation']['predictions']
        assert [(r['id'],r['predicted']) for r in v['predictions']]==[(r['id'],r['predicted']) for r in expected]
        cursor=dict(step=0,epoch=1,offset=0,best_score=v['balanced_accuracy'],best_path=init,
                    best_validation=v,stale_epochs=0,epoch_improved=False,elapsed_seconds=0.)
        journal(out/'events.jsonl',dict(event='started',initial_validation=v['correct'],unique_pairs=len(pairs)))
    resumed_step=cursor['step'];elapsed_before=cursor['elapsed_seconds'];last_saved=latest['step'] if latest else -1
    total_updates=math.ceil(len(pairs)/cfg['batch'])*cfg['max_epochs'];loss_buffer=[]
    model.gradient_checkpointing_enable();model.enable_input_require_grads()
    def checkpoint(reason):
        nonlocal last_saved
        val=evaluate_model(model,fmt,val_rows)
        eligible=all(val['recalls'][l]+1e-12>=r for l,r in floor.items())
        name=f"checkpoints/step_{cursor['step']:06d}"
        target=out/name
        if eligible and val['balanced_accuracy']>cursor['best_score']+1e-12:
            cursor['best_score']=val['balanced_accuracy'];cursor['best_path']=str(target/'adapter')
            cursor['best_validation']=val;cursor['epoch_improved']=True
        if cursor['offset']>=len(pairs):
            cursor['stale_epochs']=0 if cursor['epoch_improved'] else cursor['stale_epochs']+1
            cursor['epoch']+=1;cursor['offset']=0;cursor['epoch_improved']=False
        cursor['elapsed_seconds']=elapsed_before+time.monotonic()-begin
        if not target.exists():
            temp=target.with_name(target.name+'.writing');temp.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(temp/'adapter')
            torch.save(dict(config_hash=cfg_hash,cursor=cursor,optimizer=optimizer.state_dict(),
                python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp/'state.pt')
            meta=dict(time=now(),reason=reason,step=cursor['step'],next_epoch=cursor['epoch'],next_offset=cursor['offset'],
                eligible=eligible,validation=val,config_hash=cfg_hash,mean_recent_loss=sum(loss_buffer)/max(1,len(loss_buffer)),
                best_score=cursor['best_score'],best_path=cursor['best_path'])
            atomic(temp/'metadata.json',meta);temp.rename(target)
            journal(out/'checkpoints.jsonl',dict(step=cursor['step'],path=name,reason=reason,
                validation_correct=val['correct'],validation_total=val['total'],macro_recall=val['balanced_accuracy'],
                recalls=val['recalls'],eligible=eligible,best_score=cursor['best_score']))
        atomic(out/'latest.json',dict(step=cursor['step'],path=name,config_hash=cfg_hash))
        atomic(out/'status.json',dict(status='running',pid=os.getpid(),time=now(),**cursor))
        last_saved=cursor['step'];loss_buffer.clear();model.train()
        print(job_id,'CHECKPOINT',cursor['step'],val['correct'],val['balanced_accuracy'],flush=True)
    if latest is None:checkpoint('initial')
    while cursor['epoch']<=cfg['max_epochs']:
        if cursor['stale_epochs']>=cfg['patience']:
            reason='early_stopping';break
        order=list(pairs);random.Random(cfg['seed']+cursor['epoch']).shuffle(order);model.train()
        while cursor['offset']<len(order):
            if stop[0] or (ROOT/'STOP').exists():
                if last_saved!=cursor['step']:checkpoint('stop_requested')
                atomic(out/'status.json',dict(status='paused',time=now(),**cursor));return 75
            batch=order[cursor['offset']:cursor['offset']+cfg['batch']];optimizer.zero_grad()
            for r in batch:
                chosen=fmt.logp(model,r['prompt'],fmt.CODES[r['chosen']]);rejected=fmt.logp(model,r['prompt'],fmt.CODES[r['rejected']])
                z=cfg['beta']*(chosen-rejected-r['reference_margin']);eps=cfg['smoothing']
                loss=r['weight']*(-(1-eps)*F.logsigmoid(z)-eps*F.logsigmoid(-z)-cfg['nll']*chosen/2)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                (loss/len(batch)).backward();loss_buffer.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            cursor['step']+=1;cursor['offset']+=len(batch)
            rate=cfg['lr']*min(1.,cursor['step']/4)*max(.1,1-(cursor['step']-1)/total_updates)
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.step()
            end_epoch=cursor['offset']>=len(order)
            if cursor['step']%cfg['checkpoint_every']==0 or end_epoch or (pause_after and cursor['step']>=pause_after):
                checkpoint('epoch_end' if end_epoch else 'periodic')
            if pause_after and cursor['step']>=pause_after:
                atomic(out/'status.json',dict(status='paused_for_resume_test',resumed_from=resumed_step,time=now(),**cursor));return 75
            if end_epoch:break
        if cursor['stale_epochs']>=cfg['patience']:
            reason='early_stopping';break
    else:reason='max_epochs'
    if last_saved!=cursor['step']:checkpoint('final')
    atomic(out/'complete.json',dict(status='complete',reason=reason,time=now(),resumed_from_step=resumed_step,**cursor))
    atomic(out/'status.json',dict(status='complete',reason=reason,time=now(),**cursor));return 0

def gpu_free(index):
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(index)],text=True)
        return int(output.strip())<3000
    except Exception:return False

def supervise():
    guard=lock(ROOT/'supervisor.lock');plan=read(ROOT/'plan.json');start=time.time()
    atomic(ROOT/'supervisor.json',dict(pid=os.getpid(),status='running',started=now(),deadline_unix=start+plan['max_wall_hours']*3600))
    queue=list(plan['jobs']);mutex=threading.Lock();results={};stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    def worker(gpu):
        while True:
            with mutex:
                if not queue or stop[0]:return
                cfg=queue.pop(0)
            job=cfg['id'];out=ROOT/'jobs'/job;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():
                results[job]='already_complete';continue
            attempt=0
            while attempt<=2 and not stop[0]:
                running=read(out/'process.json') if (out/'process.json').exists() else {}
                pid=running.get('pid',0);proc=None
                if not alive(pid,job):
                    while not gpu_free(gpu) and not stop[0]:time.sleep(5)
                    if stop[0]:break
                    with (out/'train.log').open('a') as log:
                        proc=subprocess.Popen([sys.executable,str(ROOT/'durable_training.py'),'train','--job',job],
                            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                            stdout=log,stderr=log,start_new_session=True)
                    pid=proc.pid;atomic(out/'process.json',dict(pid=pid,gpu=gpu,time=now(),attempt=attempt))
                    journal(ROOT/'events.jsonl',dict(event='job_launched',job=job,pid=pid,gpu=gpu,attempt=attempt))
                while alive(pid,job):
                    if stop[0] and not (ROOT/'STOP').exists():(ROOT/'STOP').write_text('Supervisor stop or time limit\n')
                    if proc is not None and proc.poll() is not None:break
                    time.sleep(5)
                if (out/'complete.json').exists():results[job]='complete';break
                if stop[0] or (ROOT/'STOP').exists():results[job]='paused';break
                attempt+=1;journal(ROOT/'events.jsonl',dict(event='retry',job=job,attempt=attempt))
            else:results[job]='failed'
    threads=[threading.Thread(target=worker,args=(g,),daemon=False) for g in plan['gpus']]
    for t in threads:t.start()
    while any(t.is_alive() for t in threads):
        if time.time()-start>=plan['max_wall_hours']*3600 or (ROOT/'STOP').exists():stop[0]=True
        jobs={j['id']:read(ROOT/'jobs'/j['id']/'status.json') if (ROOT/'jobs'/j['id']/'status.json').exists() else {'status':'queued'} for j in plan['jobs']}
        atomic(ROOT/'progress.json',dict(time=now(),supervisor_pid=os.getpid(),elapsed_seconds=time.time()-start,
            stopping=stop[0],jobs=jobs,results=results.copy()))
        time.sleep(5)
    for t in threads:t.join()
    baseline=read(ROOT/'data/baseline_metrics.json')
    candidates=[dict(id='previous_dpo',score=baseline['dpo']['validation']['balanced_accuracy'],adapter=plan['adapters']['dpo'])]
    for job in plan['jobs']:
        path=ROOT/'jobs'/job['id']/'complete.json'
        if path.exists():
            r=read(path);candidates.append(dict(id=job['id'],score=r['best_score'],adapter=r['best_path']))
    winner=max(candidates,key=lambda c:c['score'])
    atomic(ROOT/'selection.json',dict(selected=winner,candidates=candidates,rule='Validation macro recall only; ties preserve previous DPO.',time=now()))
    status='paused_with_pending' if stop[0] else 'complete'
    atomic(ROOT/'supervisor.json',dict(status=status,pid=os.getpid(),finished=now(),results=results))
    atomic(ROOT/'progress.json',dict(time=now(),status=status,supervisor_pid=os.getpid(),results=results,
        jobs={j['id']:read(ROOT/'jobs'/j['id']/'status.json') if (ROOT/'jobs'/j['id']/'status.json').exists() else {'status':'queued'} for j in plan['jobs']}))
    if not stop[0]:
        with (ROOT/'final_evaluation.log').open('a') as log:
            result=subprocess.run([sys.executable,str(ROOT/'durable_training.py'),'evaluate'],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(plan['gpus'][0])),stdout=log,stderr=log)
        atomic(ROOT/'final_status.json',dict(status='complete' if result.returncode==0 else 'evaluation_failed',time=now(),results=results,selected=winner))
    return 0

def final_evaluate():
    plan=read(ROOT/'plan.json');chosen=read(ROOT/'selection.json')['selected'];baseline=read(ROOT/'data/baseline_metrics.json')
    if chosen['id']=='previous_dpo':metrics=baseline['dpo']
    else:
        from duplexjev import Decider
        from peft import PeftModel
        import model_format as fmt
        engine=Decider.from_pretrained(plan['base'],device='cuda:0',attn_implementation='sdpa');fmt.tokenizer=engine.tok
        model=PeftModel.from_pretrained(engine.model.language_model,chosen['adapter']).eval()
        metrics={s:evaluate_model(model,fmt,load_rows(s)) for s in ['validation','test','regression']}
    atomic(ROOT/'final_metrics.json',dict(selected=chosen,metrics=metrics,baseline=baseline['dpo'],
        limitation='Historical Codex references; reused validation; no new blind test; no audio training.'))
    (ROOT/'final_report.txt').write_text('DPO 长程任务完成\n选择：'+chosen['id']+'\n'+
        '\n'.join(f"{s}: {metrics[s]['correct']}/{metrics[s]['total']}, 类别平均召回 {metrics[s]['balanced_accuracy']:.2%}" for s in ['validation','test','regression'])+
        '\n全部检查点见 jobs/*/checkpoints.jsonl。逐条结果见 final_metrics.json。未上传 GitHub。\n')
    return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','supervise','launch','evaluate','status'])
    parser.add_argument('--job');parser.add_argument('--pause-after',type=int,default=0);args=parser.parse_args()
    if args.mode=='train':return train(args.job,args.pause_after)
    if args.mode=='supervise':return supervise()
    if args.mode=='evaluate':return final_evaluate()
    if args.mode=='status':print(json.dumps(read(ROOT/'progress.json'),ensure_ascii=False,indent=2));return 0
    if (ROOT/'supervisor.json').exists() and alive(read(ROOT/'supervisor.json').get('pid',0),'supervise'):
        print('Already running',read(ROOT/'supervisor.json')['pid']);return 0
    with (ROOT/'supervisor.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,str(ROOT/'durable_training.py'),'supervise'],cwd=ROOT,
             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print('DETACHED_SUPERVISOR_PID',proc.pid,flush=True);return 0
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc();sys.exit(1)
