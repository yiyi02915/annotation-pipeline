import copy,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'step3'))
from run_multi import classify
def main():
 plan=dict(cases=['B04','B05','B14'],reason='One preselected reference case for continuous feedback, interruption continuation and normal answer; selected before viewing model predictions',additional_runs=1)
 (ROOT/'repeat_plan.json').write_text(json.dumps(plan,indent=2))
 while not (ROOT/'timing.json').exists():time.sleep(5)
 inputs={r['number']:r for r in json.loads((ROOT/'inputs.json').read_text())}
 audio=json.loads((ROOT/'audio_votes.json').read_text());first=json.loads((ROOT/'optimized.json').read_text())
 def work(n):
  c=copy.deepcopy(inputs[n]['case']);c['context_layout']='review'
  r=classify(c,audio=audio[c['id']],endpoint='http://127.0.0.1:18771/v1/chat/completions')
  signature=lambda w:[(e['candidate_ids'],e['label'],e['start'],e['end']) for e in w.get('events',[])]
  return dict(number=n,status=r['status'],identical_structure=signature(first[n])==signature(r),result=r)
 with ThreadPoolExecutor(2) as pool:out=list(pool.map(work,plan['cases']))
 (ROOT/'repeat_results.json').write_text(json.dumps(out,indent=2))
 print(json.dumps([{k:v for k,v in r.items() if k!='result'} for r in out]),flush=True)
if __name__=='__main__':main()
