import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from run import ROOT,SRC,REF,save,log,score
sys.path.insert(0,str(ROOT/'step3'))
from context_review_bi import review

def main():
    inputs=json.loads((SRC/'inputs_complete.json').read_text());cases={c['id']:c for row in inputs for c in row['cases']}
    historical=json.loads((SRC/'results.json').read_text());baseline=historical['baseline_local']['bi']
    refhash=hashlib.sha256(REF.read_bytes()).hexdigest()
    save('final_config_locked.json',dict(reference_sha256=refhash,reference_labels_passed_to_model=False,
        cohort='same regression cohort used for diagnosis; not a fresh blind test',workers=2,
        baseline='unchanged cached baseline_local; same frozen mixed-audio JEV votes',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').glob('*') if p.is_file()}))
    start=time.monotonic();out={}
    def work(uid):
        r=review(cases[uid],baseline[uid],endpoint='http://127.0.0.1:18771/v1/chat/completions')
        save('final_predictions/'+hashlib.sha256(uid.encode()).hexdigest()[:16]+'.json',r)
        log('FINAL_WINDOW_DONE',id=uid,reviewed=len(r['context_reviews']),changed=sum(q['changed'] for q in r['context_reviews']))
        return uid,r
    with ThreadPoolExecutor(2) as pool:
        for f in as_completed([pool.submit(work,uid) for uid in baseline]):
            uid,r=f.result();out[uid]=r
    save('final_results.json',out)
    variants={name:historical[name]['bi'] for name in ('baseline_local','complete_context','anchored_complete_context')}
    variants['typed_repaired']=json.loads((ROOT/'repaired_stateful_results.json').read_text())
    variants['review_v1']=json.loads((ROOT/'review_results.json').read_text());variants['stateful']=out
    assert hashlib.sha256(REF.read_bytes()).hexdigest()==refhash
    evaluation=score(variants,json.loads(REF.read_text()));save('final_evaluation.json',evaluation)
    reviews=[r for w in out.values() for r in w['context_reviews']]
    save('final_summary.json',dict(status='complete',metrics=evaluation['metrics'],patterns=evaluation['patterns'],
        review_wall_seconds=time.monotonic()-start,reviewed_candidates=len(reviews),changed=sum(r['changed'] for r in reviews),
        failed_reviews=sum(r['status']!='ok' for r in reviews),requests=sum(len(r['attempts']) for r in reviews),
        timing_note='review-only wall time; cached initial text/audio judgments; not end-to-end throughput'))
    log('FINAL_COMPLETE',metrics=evaluation['metrics']['stateful'])
if __name__=='__main__':main()
