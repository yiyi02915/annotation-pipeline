import hashlib,json,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'bi_blind_20261116'
sys.path.insert(0,str(ROOT/'step3'))
from expanded_review_bi import review
def save(name,x):(ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 inputs=json.loads((SRC/'inputs.json').read_text());old=json.loads((SRC/'optimized.json').read_text())
 for n in ('reference_locked.json','reference_lock.json','inputs.json'):(ROOT/n).write_bytes((SRC/n).read_bytes())
 save('baseline.json',old)
 save('dev_config.json',dict(source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').iterdir() if p.is_file()},
   cohort='previous blind set now development regression; not a new blind test',reference_sha256=hashlib.sha256((ROOT/'reference_locked.json').read_bytes()).hexdigest(),
   started=datetime.now(timezone.utc).isoformat(),labels_in_model_input=False,workers=2))
 for _ in range(180):
  try:
   with urllib.request.urlopen('http://127.0.0.1:18771/health',timeout=3):break
  except Exception:time.sleep(5)
 else:raise RuntimeError('model_not_ready')
 out={};t=time.monotonic()
 def work(r):return r['number'],review(r['case'],old[r['number']],endpoint='http://127.0.0.1:18771/v1/chat/completions')
 with ThreadPoolExecutor(2) as pool:
  for f in as_completed([pool.submit(work,r) for r in inputs]):
   n,res=f.result();out[n]=res;save('optimized.json',out)
   print('DONE',n,res['status'],res.get('context_review_failed'),len(res.get('expanded_review_attempts',[])),flush=True)
 save('dev_timing.json',dict(wall_seconds=time.monotonic()-t,windows=len(out),failed_reviews=sum(r.get('context_review_failed',False) for r in out.values())))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
