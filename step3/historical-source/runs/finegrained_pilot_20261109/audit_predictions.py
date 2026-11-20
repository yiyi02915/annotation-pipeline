import collections,json,hashlib,math
from pathlib import Path
from benchmark_common import ROOT,LABELS,read_inputs
rows=read_inputs();ids={e['id']:e for e in rows};report={}
for name in ['qwen','jev']:
    path=ROOT/(name+'_predictions.jsonl')
    preds=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(preds)==3000 and len({p['id'] for p in preds})==3000
    assert {p['id'] for p in preds}==set(ids)
    assert collections.Counter(p['task'] for p in preds)=={'overlap':1500,'pause':1500}
    for p in preds:
        assert p['status']=='ok' and p['label'] in LABELS[p['task']]
        assert p['task']==ids[p['id']]['task']
        if name=='qwen':
            assert p['finish_reason']=='stop' and not p['attempts']
            assert p['usage']['completion_tokens']>0
        else:
            probabilities=p['model_answers']['label']['probs']
            assert set(probabilities)==set(LABELS[p['task']])
            assert abs(sum(probabilities.values())-1)<1e-4
            assert all(math.isfinite(x) and 0<=x<=1 for x in probabilities.values())
            assert abs(probabilities[p['label']]-max(probabilities.values()))<1e-6
    runtime=json.loads((ROOT/(name+'_runtime.json')).read_text())
    assert runtime['reused_prediction_count']==0
    assert all(t['count']==1500 and t['statuses']=={'ok':1500} for t in runtime['tasks'])
    report[name]={'count':3000,'unique_ids':3000,'status':'passed','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'cached_answers_used':False,'task_counts':dict(collections.Counter(p['task'] for p in preds))}
comparison=json.loads((ROOT/'comparison_summary.json').read_text())
assert comparison['complete'] and comparison['accuracy'] is None
result={'passed':True,'models':report,'comparison_complete':True,'behavioral_accuracy_verified':False}
(ROOT/'prediction_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
