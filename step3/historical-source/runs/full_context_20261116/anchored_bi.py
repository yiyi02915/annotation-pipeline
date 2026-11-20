import copy,hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'step3'))
from run_multi import classify
from run import save,log,evaluate,ENDPOINT
inputs=json.loads((ROOT/'inputs_complete.json').read_text())
previous=json.loads((BASE/'runs/overlap_recheck_20261116/first_results.json').read_text())
save('anchored_config_locked.json',dict(variant='anchored_complete_context',
    change='Local onset view + complete three-exchange text with block timestamps; original local JEV votes; no duration override.',
    reference_labels_in_prompt=False,development='Exploratory refinement after aggregate full-context regression; not new blind data.',
    reference_sha256=json.loads((ROOT/'config_locked.json').read_text())['reference_sha256'],
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').glob('*') if p.is_file()}))
start=time.monotonic();out={}
def run(case):
    case=copy.deepcopy(case);case.update(context_layout='anchored',audio_text_scope='local')
    votes=dict(clip_id=case['id'],events=previous[case['id']]['audio_votes'])
    result=classify(case,audio=votes,concise=True,refined=True,floor_checks=True,recover_invalid_merge=True,endpoint=ENDPOINT)
    result['conversation_id']=case['conversation_id']
    result['attempts']=[{k:v for k,v in a.items() if k!='response'} for a in result['attempts']]
    save('anchored_bi/'+hashlib.sha256(case['id'].encode()).hexdigest()[:16]+'.json',result)
    log('ANCHORED_BI_DONE',id=case['id'],status=result['status'])
    return case['id'],result
with ThreadPoolExecutor(2) as pool:
    for f in as_completed([pool.submit(run,c) for row in inputs for c in row['cases']]):
        uid,r=f.result();out[uid]=r
save('anchored_bi_results.json',out)
# Single final writer: wait for gap repair to finish before combining and scoring.
while not (ROOT/'context_repair_complete.json').exists():time.sleep(5)
results=json.loads((ROOT/'results.json').read_text())
res=copy.deepcopy(results['complete_context']);res['bi']=out
results['anchored_complete_context']=res
cap=copy.deepcopy(results['complete_context_cap3']);cap['bi']=out
results['anchored_complete_context_cap3']=cap
save('results.json',results);evaluate(results)
save('anchored_complete.json',dict(status='complete',windows=len(out),wall_seconds=time.monotonic()-start,errors=sum(r['status']!='ok' for r in out.values())))
log('ANCHORED_COMPLETE')
