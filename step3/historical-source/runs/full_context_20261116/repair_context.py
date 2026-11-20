"""Deduplicate history timing, verify model token budget, then re-run gap branch."""
import copy,hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1];SRC=BASE/'runs/step3_semantic_fix_20261115'
sys.path.insert(0,str(ROOT/'step3'))
from dialogue_context import DialogueContext,apply_long_gap_policy
from input_seamless import load_published
from gap_judge import request,PROMPT,COMPLETE_CONTEXT_RULES
from context_budget import fit_gap_payload
from run import save,log,evaluate,ENDPOINT
old=json.loads((SRC/'inputs.json').read_text());byid={r['row']['conversation_id']:r for r in old}
base={p['id']:p for p in json.loads((ROOT/'inputs_baseline_current.json').read_text())}
fresh=[];audits=[]
for row,streams,tracks,error in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),[r['row'] for r in old]):
    assert not error,error
    ctx=DialogueContext(streams)
    for oldp in byid[row['conversation_id']]['gaps']:
        p=ctx.gap(base[oldp['id']])
        fitted,audit=fit_gap_payload(p,PROMPT+COMPLETE_CONTEXT_RULES,ENDPOINT,'qwen38-27b')
        audits.append(dict(id=p['id'],**audit));fresh.append(p)
save('inputs_complete_dedup.json',fresh);save('token_budget_audit.json',audits)
log('TOKEN_PREFLIGHT_COMPLETE',n=len(fresh),fallbacks=sum(a['whole_block_fallback'] for a in audits),maximum_tokens=max(a['checks'][-1]['input_tokens'] for a in audits))
if '--prepare-only' in sys.argv:sys.exit(0)
while not (ROOT/'baseline_correction_complete.json').exists():time.sleep(5)
results=json.loads((ROOT/'results.json').read_text())
save('results_before_context_repair.json',results)
start=time.monotonic();pred={}
def classify(p):
    r=request(p,PROMPT+COMPLETE_CONTEXT_RULES,endpoint=ENDPOINT)
    r=apply_long_gap_policy(r,p,0);r['target']=p['target'];r['gap_activity']=p['gap_activity'];r['context_policy']=p['context_policy']
    save(f'complete_context_dedup/{hashlib.sha256(p["id"].encode()).hexdigest()[:16]}.json',r)
    log('REPAIRED_CONTEXT_GAP',id=p['id'],status=r['status'])
    return p['id'],r
with ThreadPoolExecutor(2) as pool:
    for f in as_completed([pool.submit(classify,p) for p in fresh]):
        uid,r=f.result();pred[uid]=r
results['complete_context']['gaps']=pred
for limit in (3,5,10):
    res=copy.deepcopy(results['complete_context'])
    for uid,g in res['gaps'].items():res['gaps'][uid]=apply_long_gap_policy(g,dict(target=g['target']),limit)
    results[f'complete_context_cap{limit}']=res
save('results.json',results);evaluate(results)
save('context_repair_complete.json',dict(status='complete',n=len(pred),wall_seconds=time.monotonic()-start,
    errors=sum(r['status']!='ok' for r in pred.values()),fallbacks=sum(r.get('input_budget',{}).get('whole_block_fallback',False) for r in pred.values()),
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').glob('*') if p.is_file()}))
log('CONTEXT_REPAIR_COMPLETE')
