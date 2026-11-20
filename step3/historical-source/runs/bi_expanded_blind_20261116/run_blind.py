import copy,hashlib,json,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'step3'))
from audio_judge import MixedAudioJudge
from run_multi import classify
from context_review_bi import review
from expanded_review_bi import review as graph_review
from guarded_expanded_bi import review as guarded_review
def save(name,x):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def log(event,**kw):
 x=dict(at=datetime.now(timezone.utc).isoformat(),event=event,**kw)
 with (ROOT/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 print(json.dumps(x),flush=True)
def main():
 lock=json.loads((ROOT/'source_lock.json').read_text())
 assert all(hashlib.sha256((ROOT/'step3'/n).read_bytes()).hexdigest()==h for n,h in lock['source_hashes'].items())
 t=time.monotonic();judge=MixedAudioJudge(BASE/'models/DuplexJev-32B-Para','cuda:0');log('AUDIO_MODEL_LOADED',seconds=time.monotonic()-t)
 log('WAITING_FOR_LOCKED_REFERENCE')
 while not (ROOT/'reference_lock.json').exists():time.sleep(5)
 refs=json.loads((ROOT/'reference_locked.json').read_text());refhash=hashlib.sha256((ROOT/'reference_locked.json').read_bytes()).hexdigest()
 assert refhash==json.loads((ROOT/'reference_lock.json').read_text())['sha256']
 inputs=json.loads((ROOT/'inputs.json').read_text())
 assert {x['number'] for x in inputs}==set(refs['cases'])
 for _ in range(180):
  try:
   with urllib.request.urlopen('http://127.0.0.1:18771/health',timeout=3):break
  except Exception:time.sleep(5)
 else:raise RuntimeError('qwen_not_ready')
 save('inference_lock.json',dict(reference_sha256=refhash,input_sha256=hashlib.sha256((ROOT/'inputs.json').read_bytes()).hexdigest(),
   source_hashes=lock['source_hashes'],prediction_start=datetime.now(timezone.utc).isoformat(),workers=2,temperature=0,
   model='qwen38-27b',qwen_context=32768,tensor_parallel=1,jev='DuplexJev-32B-Para',audio_mode='fresh mixed dual tracks, four option permutations',
   label_source='Codex before predictions; not independent human ground truth',labels_in_model_input=False))
 start=time.monotonic();audio={}
 for r in inputs:
  c=r['case'];votes=dict(clip_id=c['id'],events=[])
  for x in r['candidates']:
   if x['other_active_run_at_onset'] and x['overlap_seconds']>0:
    votes['events'].append(dict(candidate_id=x['candidate_id'],mix=judge.vote(c,x,ROOT/'audio'/f"{r['number']}.wav",'en')))
  audio[c['id']]=votes;save('audio_votes.json',audio);log('AUDIO_DONE',number=r['number'],votes=len(votes['events']))
 audio_seconds=time.monotonic()-start
 baseline={};optimized={};original={};text_start=time.monotonic()
 def work(r):
  c=r['case'];local={k:v for k,v in c.items() if k not in ('context_layout','dialogue_context','context_policy','audio_text_scope')}
  result=classify(local,audio=audio[c['id']],concise=True,refined=True,floor_checks=True,recover_invalid_merge=True,endpoint='http://127.0.0.1:18771/v1/chat/completions')
  result['conversation_id']=c['conversation_id']
  previous=review(c,result,endpoint='http://127.0.0.1:18771/v1/chat/completions')
  proposal=graph_review(c,previous,endpoint='http://127.0.0.1:18771/v1/chat/completions')
  enhanced=guarded_review(c,previous,proposal,endpoint='http://127.0.0.1:18771/v1/chat/completions')
  log('TEXT_DONE',number=r['number'],status=enhanced['status'],reviews=len(enhanced.get('context_reviews',[])))
  return r['number'],result,previous,enhanced
 with ThreadPoolExecutor(2) as pool:
  for f in as_completed([pool.submit(work,r) for r in inputs]):
   n,l,b,o=f.result();original[n]=l;baseline[n]=b;optimized[n]=o;save('original_local.json',original);save('baseline.json',baseline);save('optimized.json',optimized)
 assert hashlib.sha256((ROOT/'reference_locked.json').read_bytes()).hexdigest()==refhash
 save('timing.json',dict(audio_seconds=audio_seconds,text_seconds=time.monotonic()-text_start,wall_seconds=time.monotonic()-start,
   note='shared fresh audio then concurrent local+previous review+expanded graph+guarded positive audits; model loading excluded; paired experiment not independent production benchmarks'))
 log('INFERENCE_COMPLETE',windows=len(optimized),failed=sum(r['status']!='ok' for r in optimized.values()))
if __name__=='__main__':main()
