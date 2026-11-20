import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
SRC=BASE/'runs/step3_semantic_fix_20261115'
sys.path.insert(0,str(ROOT/'step3'))
from input_seamless import load_published
from dialogue_context import DialogueContext
def save(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
inputs=json.loads((SRC/'inputs.json').read_text())
rows=[x['row'] for x in inputs]
full={row['conversation_id']:(streams,tracks,error) for row,streams,tracks,error in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),rows)}
updated=[];stats=[]
for inp in inputs:
    streams,tracks,error=full[inp['row']['conversation_id']]
    assert not error,error
    ctx=DialogueContext(streams)
    new=copy.deepcopy(inp)
    new['cases']=[ctx.bi(c) for c in inp['cases']]
    new['gaps']=[ctx.gap(g) for g in inp['gaps']]
    updated.append(new)
    stats.append(dict(id=inp['row']['conversation_id'],full_words=sum(map(len,streams.values())),
        blocks=len(ctx.blocks),bi=[dict(chars=sum(len(b['text']) for b in c['dialogue_context']),**c['context_policy']) for c in new['cases']],
        gaps=[dict(id=g['id'],chars=sum(len(b['text']) for b in g['context']),**g['context_policy']) for g in new['gaps']]))
save('inputs_complete.json',updated);save('context_audit.json',stats)
save('config_locked.json',dict(reference_sha256=hashlib.sha256((SRC/'benchmark_reference.json').read_bytes()).hexdigest(),
    source_commit='a62c5a28bfbcf2824904f868ca947e1ae9d936b7',threshold=3.0,
    threshold_rationale='Engineering cap requested by user, consistent with prior 3s convention; not human hesitation maximum.',
    sensitivity_thresholds=[0,3,5,10],variants=['baseline_local','complete_context','complete_context_cap3'],
    scoring='locked 61 overlap candidates and 15 gap targets; no reference labels enter inference',
    qwen_model='qwen38-27b',qwen_workers=2,sft_used=False,
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').glob('*') if p.is_file()}))
print(json.dumps(dict(recordings=len(updated),bi=sum(len(x['cases']) for x in updated),gaps=sum(len(x['gaps']) for x in updated),
    max_bi_chars=max(b['chars'] for x in stats for b in x['bi']),max_gap_chars=max(b['chars'] for x in stats for b in x['gaps']))))
