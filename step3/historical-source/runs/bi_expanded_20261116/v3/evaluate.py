import hashlib,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def event_label(w,uid):
 if w.get('status')!='ok':return None
 return next((e['label'] for e in w.get('events',[]) if e['candidate_ids'][0]==uid),'NEITHER')
def counts(rows):
 return dict(n=len(rows),valid=sum(r['predicted'] is not None for r in rows),correct=sum(r['predicted']==r['reference'] for r in rows),
   by_label={l:dict(n=sum(r['reference']==l for r in rows),correct=sum(r['predicted']==r['reference']==l for r in rows),
    false_positive=sum(r['predicted']==l and r['reference']!=l for r in rows)) for l in ('INTERRUPT','BACKCHANNEL','NEITHER')})
def main():
 refs=json.loads((ROOT/'reference_locked.json').read_text());lock=json.loads((ROOT/'reference_lock.json').read_text())
 assert hashlib.sha256((ROOT/'reference_locked.json').read_bytes()).hexdigest()==lock['sha256']
 results={k:json.loads((ROOT/f'{k}.json').read_text()) for k in ('baseline','optimized')}
 assert all(set(r)==set(refs['cases']) for r in results.values())
 metrics={};details=[];structures=[]
 for name,windows in results.items():
  rows=[];checks=[]
  for number,ref in refs['cases'].items():
   w=windows[number]
   for e in ref['events']:
    rows.append(dict(variant=name,number=number,id=e['candidate_id'],reference=e['label'],predicted=event_label(w,e['candidate_id']),
      stratum=ref['stratum'],function=e['function'],reference_quality_flag=ref['reference_quality_flag']))
   for c in ref['structure_checks']:
    hits={u:[e for e in w.get('events',[]) if u in e['candidate_ids'] and e['label']==c['label']] for u in (c['parent'],c['child'])}
    a,b=hits[c['parent']],hits[c['child']]
    ok=bool(len(a)==len(b)==1 and ((a[0]['candidate_ids'][0]!=b[0]['candidate_ids'][0]) if c['kind']=='separate_feedback' else (a[0]['candidate_ids'][0]==b[0]['candidate_ids'][0])))
    checks.append(dict(variant=name,number=number,correct=ok,**c))
  ordinary=[r for r in rows if r['function']=='ordinary_response']
  metrics[name]=dict(all=counts(rows),higher_reference_quality=counts([r for r in rows if not r['reference_quality_flag']]),
    by_stratum={s:counts([r for r in rows if r['stratum']==s]) for s in sorted({r['stratum'] for r in rows})},
    normal_answer=counts(ordinary),
    structure={s:dict(n=sum(c['kind']==s for c in checks),correct=sum(c['kind']==s and c['correct'] for c in checks)) for s in ('same_feedback','same_interruption','separate_feedback')},
    failed_windows=sum(w['status']!='ok' for w in windows.values()))
  details+=rows;structures+=checks
 base={r['id']:r for r in details if r['variant']=='baseline'};new={r['id']:r for r in details if r['variant']=='optimized'}
 fixed=[r for uid,r in new.items() if base[uid]['predicted']!=r['reference']==r['predicted']]
 regressed=[r for uid,r in new.items() if base[uid]['predicted']==r['reference']!=r['predicted']]
 reviews=[dict(number=n,**r) for n,w in results['optimized'].items() for r in w.get('context_reviews',[])]
 core=[r for r in reviews if r['candidate_id'] in base]
 baseline_errors={uid for uid,r in base.items() if r['predicted']!=r['reference']}
 review_stats=dict(all_candidate_reviews=len(reviews),core_candidate_reviews=len(core),core_review_rate=len(core)/len(base),
    requests=sum(len(r['attempts']) for r in reviews),failures=sum(r['status']!='ok' for r in reviews),
    changes=sum(r['changed'] for r in reviews),core_changes=sum(r['changed'] for r in core),
    baseline_errors=len(baseline_errors),errors_routed=sum(r['candidate_id'] in baseline_errors for r in core),
    correct_routed=sum(r['candidate_id'] not in baseline_errors for r in core))
 rng=random.Random(2026111603);numbers=list(refs['cases']);sample=[]
 per={n:([r for r in base.values() if r['number']==n],[r for r in new.values() if r['number']==n]) for n in numbers}
 for _ in range(5000):
  selected=[per[rng.choice(numbers)] for _ in numbers];den=sum(len(a) for a,b in selected)
  sample.append(sum(sum(r['predicted']==r['reference'] for r in b)-sum(r['predicted']==r['reference'] for r in a) for a,b in selected)/den)
 sample.sort();discord=len(fixed)+len(regressed);small=min(len(fixed),len(regressed))
 significance=dict(paired_discordant=discord,fixed=len(fixed),regressed=len(regressed),
  mcnemar_exact_p=min(1,2*sum(math.comb(discord,k) for k in range(small+1))/2**discord) if discord else 1,
  conversation_cluster_bootstrap_delta_ci95=[sample[125],sample[4874]],bootstrap_repetitions=5000,
  note='Small targeted sample and Codex references limit generalization; cluster resampling accounts for positions within conversations')
 out=dict(metrics=metrics,fixed=fixed,regressed=regressed,review=review_stats,significance=significance,details=details,structures=structures)
 (ROOT/'evaluation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(json.dumps(dict(metrics=metrics,review=review_stats,significance=significance),indent=2))
if __name__=='__main__':main()
