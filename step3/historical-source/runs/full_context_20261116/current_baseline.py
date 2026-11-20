"""Regenerate uploaded-version gap payloads: old cached inputs omit newer fields."""
import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1];SRC=BASE/'runs/step3_semantic_fix_20261115'
sys.path.insert(0,str(ROOT/'step3'))
from pipeline import WordTimeline,gap_candidates
from input_seamless import load_published
from dialogue_context import DialogueContext
from gap_judge import request,PROMPT
from run import save,log,evaluate,ENDPOINT
inputs=json.loads((SRC/'inputs.json').read_text());byid={r['row']['conversation_id']:r for r in inputs}
complete={g['id']:g for r in json.loads((ROOT/'inputs_complete.json').read_text()) for g in r['gaps']}
payloads=[];audit=[]
for row,streams,tracks,error in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),[r['row'] for r in inputs]):
    assert not error,error
    timeline=WordTimeline(streams)
    fresh=list(gap_candidates(row['conversation_id'],timeline,row.get('core'),row.get('interval')))
    old={g['id']:g for g in byid[row['conversation_id']]['gaps']}
    assert set(old)=={g['id'] for g in fresh}
    ctx=DialogueContext(streams)
    for p in fresh:
        rebuilt=ctx.gap(p)
        assert rebuilt==complete[p['id']],('complete_context_payload_changed',p['id'])
        audit.append(dict(id=p['id'],added_fields=sorted(set(p)-set(old[p['id']])),target_unchanged=p['target']==old[p['id']]['target']))
        payloads.append(p)
save('inputs_baseline_current.json',payloads);save('baseline_input_audit.json',audit)
log('CURRENT_BASELINE_PREPARED',gaps=len(payloads),complete_payloads_identical=True)
if '--prepare-only' in sys.argv:sys.exit(0)
scored_cids={r['row']['conversation_id'] for r in inputs if r['row']['number'].startswith('H')}
payloads=[p for p in payloads if p['id'].split(':')[0] in scored_cids]
assert len(payloads)==15
while not (ROOT/'summary.json').exists():time.sleep(5)
folder=ROOT/'baseline_current';folder.mkdir(exist_ok=True)
start=time.monotonic();pred={}
def classify(p):
    r=request(p,PROMPT,endpoint=ENDPOINT);r['target']=p['target']
    save(f'baseline_current/{hashlib.sha256(p["id"].encode()).hexdigest()[:16]}.json',r)
    log('CURRENT_BASELINE_GAP',id=p['id'],status=r['status'])
    return p['id'],r
with ThreadPoolExecutor(2) as pool:
    for future in as_completed([pool.submit(classify,p) for p in payloads]):
        uid,r=future.result();pred[uid]=r
results=json.loads((ROOT/'results.json').read_text())
save('results_before_baseline_correction.json',results)
save('evaluation_before_baseline_correction.json',json.loads((ROOT/'evaluation.json').read_text()))
results['baseline_local']['gaps']=pred
save('results.json',results);evaluate(results)
save('baseline_correction_complete.json',dict(status='complete',gaps=len(pred),wall_seconds=time.monotonic()-start,
    reason='Baseline regenerated from uploaded gap_candidates; historical inputs lacked target_right_unit/other_before_right. Complete-context payloads verified identical.',
    errors=sum(r['status']!='ok' for r in pred.values())))
log('CURRENT_BASELINE_COMPLETE')
