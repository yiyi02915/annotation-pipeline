import copy,hashlib,json,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
SRC=BASE/'runs/full_context_20261116';REF=BASE/'runs/step3_semantic_fix_20261115/benchmark_reference.json'
sys.path.insert(0,str(ROOT/'step3'))
from run_multi import classify
def save(name,x):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def log(event,**kw):
    x=dict(at=datetime.now(timezone.utc).isoformat(),event=event,**kw)
    with (ROOT/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    print(json.dumps(x),flush=True)
def label(w,uid):
    if w.get('status')!='ok':return None
    return next((e['label'] for e in w['events'] if e['candidate_ids'][0]==uid),'NEITHER')
def score(variants,refs):
    details=[];metrics={};patterns={}
    for variant,windows in variants.items():
        rows=[];pat={}
        for cid,entry in refs['recordings'].items():
            if 'events' not in entry:continue
            found=[w for w in windows.values() if w['conversation_id']==cid]
            assert len(found)==1
            w=found[0];ids={c['candidate_id'] for c in w.get('candidates',[])}
            for target in entry['events']:
                if target['label']=='UNSCORED':continue
                uid=target['candidate_id'];pred=label(w,uid) if uid in ids else None
                rows.append(dict(number=entry['number'],id=uid,reference=target['label'],predicted=pred))
            events={e['candidate_ids'][0]:e for e in w.get('events',[])}
            for p in entry['patterns']:
                a,b=events.get(p['first']),events.get(p['second']);kind=p['kind']
                good=bool(a and b and a['label']=='INTERRUPT' and b['label']==('INTERRUPT' if kind=='bidirectional' else 'BACKCHANNEL'))
                if kind=='nested' and good:good=a['start']<=b['start']<a['end']
                row=pat.setdefault(kind,dict(n=0,correct=0));row['n']+=1;row['correct']+=int(good)
        metrics[variant]=dict(n=len(rows),correct=sum(r['reference']==r['predicted'] for r in rows),valid=sum(r['predicted'] is not None for r in rows),
            by_label={lab:dict(n=sum(r['reference']==lab for r in rows),correct=sum(r['reference']==r['predicted']==lab for r in rows),false_positive=sum(r['reference']!=lab and r['predicted']==lab for r in rows)) for lab in ['INTERRUPT','BACKCHANNEL','NEITHER']})
        patterns[variant]=pat;details.extend(dict(variant=variant,**r) for r in rows)
    changes={};new={r['id']:r for r in details if r['variant']=='stateful'}
    for v in variants:
        if v=='stateful':continue
        old={r['id']:r for r in details if r['variant']==v}
        changes[v]=dict(fixed=[r for uid,r in new.items() if old[uid]['predicted']!=r['reference']==r['predicted']],
            regressed=[r for uid,r in new.items() if old[uid]['predicted']==r['reference']!=r['predicted']])
    return dict(metrics=metrics,patterns=patterns,changes=changes,details=details)
def main():
    inputs=json.loads((SRC/'inputs_complete.json').read_text())
    previous=json.loads((BASE/'runs/overlap_recheck_20261116/first_results.json').read_text())
    reference_hash=hashlib.sha256(REF.read_bytes()).hexdigest()
    assert reference_hash==json.loads((SRC/'config_locked.json').read_text())['reference_sha256']
    save('config_locked.json',dict(reference_sha256=reference_hash,reference_labels_in_input=False,cohort='same locked regression cohort; not new blind data',
        model='qwen38-27b',thinking=False,endpoint='http://127.0.0.1:18771/v1/chat/completions',workers=2,audio_votes='frozen original local JEV evidence',
        variant='typed function -> deterministic action; full three-exchange dialogue retained',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').glob('*') if p.is_file()}))
    for attempt in range(180):
        try:
            with urllib.request.urlopen('http://127.0.0.1:18771/health',timeout=3):break
        except Exception:
            if attempt%6==0:log('WAITING_FOR_MODEL')
            time.sleep(5)
    else:raise RuntimeError('model_not_ready')
    start=time.monotonic();out={}
    def work(case):
        case=copy.deepcopy(case);case.update(context_layout='stateful',audio_text_scope='local')
        votes=dict(clip_id=case['id'],events=previous[case['id']]['audio_votes'])
        result=classify(case,audio=votes,endpoint='http://127.0.0.1:18771/v1/chat/completions')
        save('predictions/'+hashlib.sha256(case['id'].encode()).hexdigest()[:16]+'.json',result)
        log('WINDOW_DONE',id=case['id'],status=result['status'],attempts=len(result['attempts']))
        return case['id'],result
    with ThreadPoolExecutor(2) as pool:
        for future in as_completed([pool.submit(work,c) for row in inputs for c in row['cases']]):
            uid,r=future.result();out[uid]=r
    save('results.json',out)
    historical=json.loads((SRC/'results.json').read_text())
    variants={name:historical[name]['bi'] for name in ('baseline_local','complete_context','anchored_complete_context')}
    variants['stateful']=out
    assert hashlib.sha256(REF.read_bytes()).hexdigest()==reference_hash
    evaluation=score(variants,json.loads(REF.read_text()));save('evaluation.json',evaluation)
    save('summary.json',dict(status='complete',windows=len(out),wall_seconds=time.monotonic()-start,
        errors=sum(w['status']!='ok' for w in out.values()),metrics=evaluation['metrics'],patterns=evaluation['patterns'],
        requests=sum(len(w['attempts']) for w in out.values()),timing_note='text judging with cached audio; not fresh end-to-end throughput'))
    log('COMPLETE',metrics=evaluation['metrics']['stateful'])
if __name__=='__main__':main()
