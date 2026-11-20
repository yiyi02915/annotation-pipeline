import copy,hashlib,json,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1];SRC=BASE/'runs/step3_semantic_fix_20261115'
sys.path.insert(0,str(ROOT/'step3'))
from audio_judge import MixedAudioJudge,render_mix
from event_extractor import extract_candidates
from run_multi import classify
from gap_judge import request,PROMPT,COMPLETE_CONTEXT_RULES
from dialogue_context import apply_long_gap_policy
ENDPOINT='http://127.0.0.1:18771/v1/chat/completions'

def save(name,x):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def log(event,**kw):
    row=dict(at=datetime.now(timezone.utc).isoformat(),event=event,**kw)
    with (ROOT/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def eventlabel(result,cid):
    if result.get('status')!='ok':return None
    return next((e['label'] for e in result.get('events',[]) if e['candidate_ids'][0]==cid),'NEITHER')

def evaluate(results):
    refs=json.loads((SRC/'benchmark_reference.json').read_text())
    assert hashlib.sha256((SRC/'benchmark_reference.json').read_bytes()).hexdigest()==json.loads((ROOT/'config_locked.json').read_text())['reference_sha256']
    details=[];patterns={};metrics={}
    for variant,res in results.items():
        rows=[];pat={}
        for cid,entry in refs['recordings'].items():
            cases=[r for r in res['bi'].values() if r['conversation_id']==cid]
            if 'events' in entry:
                for target in entry['events']:
                    if target['label']=='UNSCORED':continue
                    uid=target['candidate_id']
                    match=[r for r in cases if uid in {c['candidate_id'] for c in r.get('candidates',[])}]
                    pred=eventlabel(match[0],uid) if len(match)==1 else None
                    rows.append(dict(task='bi',number=entry['number'],id=uid,reference=target['label'],predicted=pred))
                events={e['candidate_ids'][0]:e for r in cases for e in r.get('events',[])}
                for p in entry['patterns']:
                    kind=p['kind'];a,b=events.get(p['first']),events.get(p['second'])
                    good=bool(a and b and a['label']=='INTERRUPT' and b['label']==('INTERRUPT' if kind=='bidirectional' else 'BACKCHANNEL'))
                    if kind=='nested' and good:good=a['start']<=b['start']<a['end']
                    row=pat.setdefault(kind,dict(total=0,correct=0));row['total']+=1;row['correct']+=int(good)
            else:
                target=entry['gap']['target']
                match=[(uid,g) for uid,g in res['gaps'].items() if uid.startswith(cid+':') and all(g['target'][k]==target[k] for k in ['left_word_id','right_word_id'])]
                assert len(match)==1,(cid,len(match))
                uid,g=match[0]
                rows.append(dict(task='gap',number=entry['number'],id=uid,reference=entry['gap']['label'],predicted=g.get('label'),seconds=target['gap_seconds'],policy_conflict=g.get('duration_semantics_conflict',False)))
        totals={}
        for task in ('bi','gap','all'):
            subset=[r for r in rows if task=='all' or r['task']==task]
            totals[task]=dict(n=len(subset),correct=sum(r['predicted']==r['reference'] for r in subset),valid=sum(r['predicted'] is not None for r in subset))
            totals[task]['agreement']=totals[task]['correct']/len(subset)
            totals[task]['by_label']={label:dict(n=sum(r['reference']==label for r in subset),correct=sum(r['reference']==r['predicted']==label for r in subset),false_positive=sum(r['reference']!=label and r['predicted']==label for r in subset)) for label in sorted({r['reference'] for r in subset})}
        metrics[variant]=totals;patterns[variant]=pat
        details.extend(dict(variant=variant,**r) for r in rows)
    base={r['id']:r for r in details if r['variant']=='baseline_local'}
    changes={}
    for v in results:
        selected=[r for r in details if r['variant']==v]
        changes[v]=dict(fixed=[r for r in selected if base[r['id']]['predicted']!=r['reference']==r['predicted']],regressed=[r for r in selected if base[r['id']]['predicted']==r['reference']!=r['predicted']])
    save('evaluation.json',dict(metrics=metrics,patterns=patterns,changes=changes,details=details))

def main():
    original=json.loads((SRC/'inputs.json').read_text());complete=json.loads((ROOT/'inputs_complete.json').read_text())
    previous=json.loads((BASE/'runs/overlap_recheck_20261116/first_results.json').read_text())
    t=time.monotonic();judge=MixedAudioJudge(BASE/'models/DuplexJev-32B-Para','cuda:0');log('JEV_READY',load_seconds=time.monotonic()-t)
    for attempt in range(180):
        try:
            with urllib.request.urlopen('http://127.0.0.1:18771/health',timeout=3):break
        except Exception:
            if attempt%6==0:log('WAITING_FOR_MODEL')
            time.sleep(5)
    else:raise RuntimeError('qwen_model_not_ready')
    log('QWEN_READY')
    audio={};(ROOT/'audio').mkdir(exist_ok=True);at=time.monotonic()
    for inp in complete:
        for case in inp['cases']:
            key=hashlib.sha256(case['id'].encode()).hexdigest()[:16]
            tracks={k:(Path(v[0]),v[1]) for k,v in inp['tracks'].items()}
            mix=render_mix(tracks,case['interval'],ROOT/'audio'/f'{key}.wav')
            votes=dict(clip_id=case['id'],events=[])
            for c in extract_candidates(case,refined=True,floor_checks=True):
                if c['other_active_run_at_onset'] and c['overlap_seconds']>0:
                    votes['events'].append(dict(candidate_id=c['candidate_id'],mix=judge.vote(case,c,mix,'en')))
            audio[case['id']]=votes;save('audio_votes_complete.json',audio);log('AUDIO_DONE',id=case['id'],votes=len(votes['events']))
    audio_wall=time.monotonic()-at
    results={};timing={}
    def run_one(kind,payload,votes,variant):
        if kind=='bi':
            r=classify(payload,audio=votes,concise=True,refined=True,floor_checks=True,recover_invalid_merge=True,endpoint=ENDPOINT)
            r['conversation_id']=payload['conversation_id']
            r['attempts']=[{k:v for k,v in a.items() if k!='response'} for a in r['attempts']]
        else:
            r=request(payload,PROMPT+(COMPLETE_CONTEXT_RULES if variant=='complete_context' else ''),endpoint=ENDPOINT)
            if variant=='complete_context': r=apply_long_gap_policy(r,payload,0)
            r['target']=payload['target'];r['context_policy']=payload.get('context_policy')
            r['gap_activity']=payload.get('gap_activity')
        name=f"{variant}/{kind}_{hashlib.sha256(payload['id'].encode()).hexdigest()[:16]}.json"
        save(name,r);log('JUDGED',variant=variant,kind=kind,id=payload['id'],status=r['status'])
        return kind,payload['id'],r
    for variant,inputs in [('baseline_local',original),('complete_context',complete)]:
        t=time.monotonic();res=dict(bi={},gaps={})
        with ThreadPoolExecutor(2) as pool:
            jobs=[]
            for inp in inputs:
                for case in inp['cases']:
                    votes=audio[case['id']] if variant=='complete_context' else dict(clip_id=case['id'],events=previous[case['id']]['audio_votes'])
                    jobs.append(pool.submit(run_one,'bi',case,votes,variant))
                for g in inp['gaps']:jobs.append(pool.submit(run_one,'gaps',g,None,variant))
            for f in as_completed(jobs):
                kind,uid,r=f.result();res[kind][uid]=r
        timing[variant]=time.monotonic()-t;results[variant]=res;save('results.json',results);log('VARIANT_DONE',variant=variant,seconds=timing[variant])
    for limit in (3,5,10):
        res=copy.deepcopy(results['complete_context'])
        for uid,g in res['gaps'].items():
            res['gaps'][uid]=apply_long_gap_policy(g,dict(target=g['target']),limit)
        results[f'complete_context_cap{limit}']=res
    save('results.json',results);evaluate(results)
    save('summary.json',dict(status='complete',recordings=30,gap_predictions=61,overlap_windows=15,
        audio_complete_seconds=audio_wall,text_phase_wall_seconds=timing,
        timing_note='Baseline audio reused for matched comparison; timings are NOT fresh end-to-end throughput.',
        errors={v:sum(r.get('status')!='ok' for part in ('bi','gaps') for r in x[part].values()) for v,x in results.items()}))
    log('COMPLETE')

if __name__=='__main__':main()
