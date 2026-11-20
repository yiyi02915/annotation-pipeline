import json,collections
from benchmark_common import ROOT,LABELS
def read(name):return {p['id']:p for p in map(json.loads,(ROOT/name).read_text().splitlines())}
new=read('jev_option_check.jsonl');old=read('jev_predictions.jsonl');qwen=read('qwen_predictions.jsonl');summary={}
for task in LABELS:
    ids=[k for k,v in new.items() if v['task']==task]
    paired=[k for k in ids if k in qwen and qwen[k]['status']=='ok']
    summary[task]={'n':len(ids),'changed_labels':sum(old[k]['label']!=new[k]['label'] for k in ids),
        'original_labels':dict(collections.Counter(old[k]['label'] for k in ids)),
        'natural_option_labels':dict(collections.Counter(new[k]['label'] for k in ids)),
        'paired_with_qwen':len(paired),'original_agree_qwen':sum(old[k]['label']==qwen[k]['label'] for k in paired),
        'natural_options_agree_qwen':sum(new[k]['label']==qwen[k]['label'] for k in paired)}
result={'task_results':summary,'accuracy_evaluated':False,'scope':'Diagnostic, same 100 samples/task, only primary option wording changed; no model retraining.'}
(ROOT/'option_sensitivity_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
