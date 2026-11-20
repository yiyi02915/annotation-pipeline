import copy,hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from run import ROOT,BASE,SRC,REF,save,log,score
sys.path.insert(0,str(ROOT/'step3'))
from context_review_bi import review
from stateful_bi import classify

def main():
    inputs=json.loads((SRC/'inputs_complete.json').read_text());cases={c['id']:c for row in inputs for c in row['cases']}
    historical=json.loads((SRC/'results.json').read_text());baseline=historical['baseline_local']['bi']
    original=json.loads((ROOT/'results.json').read_text());save('initial_results.json',original)
    save('review_config_locked.json',dict(reference_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),
        baseline='cached exact baseline_local; no reference labels passed to review',workers=2,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').glob('*') if p.is_file()}))
    start=time.monotonic();out={}
    def work(uid):
        r=review(cases[uid],baseline[uid],endpoint='http://127.0.0.1:18771/v1/chat/completions')
        save('review_predictions/'+hashlib.sha256(uid.encode()).hexdigest()[:16]+'.json',r)
        log('REVIEW_DONE',id=uid,reviewed=len(r['context_reviews']),changed=sum(q['changed'] for q in r['context_reviews']))
        return uid,r
    with ThreadPoolExecutor(2) as pool:
        for f in as_completed([pool.submit(work,uid) for uid in baseline]):
            uid,r=f.result();out[uid]=r
    save('review_results.json',out)
    review_seconds=time.monotonic()-start
    previous=json.loads((BASE/'runs/overlap_recheck_20261116/first_results.json').read_text())
    for uid,w in original.items():
        if w['status']=='ok':continue
        case=copy.deepcopy(cases[uid]);case.update(context_layout='stateful',audio_text_scope='local')
        original[uid]=classify(case,dict(clip_id=uid,events=previous[uid]['audio_votes']),endpoint='http://127.0.0.1:18771/v1/chat/completions')
        log('REPAIR_DONE',id=uid,status=original[uid]['status'])
    save('repaired_stateful_results.json',original)
    variants={name:historical[name]['bi'] for name in ('baseline_local','complete_context','anchored_complete_context')}
    variants['typed_initial']=json.loads((ROOT/'initial_results.json').read_text());variants['typed_repaired']=original
    variants['stateful']=out
    evaluation=score(variants,json.loads(REF.read_text()));save('review_evaluation.json',evaluation)
    reviews=[r for w in out.values() for r in w['context_reviews']]
    save('review_summary.json',dict(status='complete',metrics=evaluation['metrics'],patterns=evaluation['patterns'],
        review_wall_seconds=review_seconds,reviewed_candidates=len(reviews),changed=sum(r['changed'] for r in reviews),
        failed_reviews=sum(r['status']!='ok' for r in reviews),requests=sum(len(r['attempts']) for r in reviews),
        timing_note='cached baseline and local audio; review-only time, not fresh end-to-end throughput'))
    log('REVIEW_COMPLETE',metrics=evaluation['metrics']['stateful'])
if __name__=='__main__':main()
