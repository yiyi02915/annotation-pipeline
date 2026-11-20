import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'bi_blind_20261116';OUT=ROOT/'v3';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'step3'))
from guarded_expanded_bi import review
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
inputs=json.loads((SRC/'inputs.json').read_text());old=json.loads((SRC/'optimized.json').read_text());proposal=json.loads((ROOT/'optimized.json').read_text())
for n in ('reference_locked.json','reference_lock.json','inputs.json'):(OUT/n).write_bytes((SRC/n).read_bytes())
save('baseline.json',old);save('config.json',dict(source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').iterdir() if p.is_file()},cached_graph_proposals=True,reference_labels_passed_to_model=False))
start=time.monotonic();out={}
def work(r):return r['number'],review(r['case'],old[r['number']],proposal[r['number']],endpoint='http://127.0.0.1:18771/v1/chat/completions')
with ThreadPoolExecutor(2) as pool:
 for f in as_completed([pool.submit(work,r) for r in inputs]):
  n,res=f.result();out[n]=res;save('optimized.json',out);print('DONE',n,len(res['response_audits']),res['context_review_failed'],flush=True)
save('timing.json',dict(wall_seconds=time.monotonic()-start,windows=len(out)));print('COMPLETE',flush=True)

