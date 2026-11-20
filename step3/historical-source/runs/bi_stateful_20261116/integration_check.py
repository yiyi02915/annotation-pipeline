import copy,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from run import ROOT,BASE,SRC,save,log
sys.path.insert(0,str(ROOT/'step3'))
from run_multi import classify

def main():
    rows=json.loads((SRC/'inputs_complete.json').read_text())
    cases=[c for row in rows for c in row['cases'] if c['conversation_id'] in ('V00_S1517_I00000694','V00_S1697_I00001046')]
    frozen=json.loads((BASE/'runs/overlap_recheck_20261116/first_results.json').read_text())
    previous=json.loads((ROOT/'guarded_results.json').read_text());start=time.monotonic()
    def work(c):
        c=copy.deepcopy(c);c.update(context_layout='review',audio_text_scope='local')
        result=classify(c,audio=dict(clip_id=c['id'],events=frozen[c['id']]['audio_votes']),endpoint='http://127.0.0.1:18771/v1/chat/completions')
        before=[(e['candidate_ids'],e['label'],e['start'],e['end']) for e in previous[c['id']]['events']]
        after=[(e['candidate_ids'],e['label'],e['start'],e['end']) for e in result.get('events',[])]
        log('INTEGRATION_DONE',id=c['id'],status=result['status'],same_as_cached=before==after)
        return dict(id=c['id'],same_as_cached=before==after,result=result)
    with ThreadPoolExecutor(2) as pool:out=list(pool.map(work,cases))
    save('integration_check.json',dict(windows=out,wall_seconds=time.monotonic()-start,
        note='Fresh initial Qwen + context review, frozen JEV audio; two affected windows only'))
if __name__=='__main__':main()
