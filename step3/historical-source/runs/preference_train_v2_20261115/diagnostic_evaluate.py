"""After all tuning is frozen, evaluate the strongest final candidate once.

This does not override the validation safety gate or choose by held-out scores.
"""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
folder=root/'teacher_targeted_strong'
history=json.loads((folder/'validation_history.json').read_text())
chosen=max((h for h in history if h['epoch']>0),key=lambda h:h['metrics']['balanced_accuracy'])
selection=json.loads((folder/'selection.json').read_text())
meta=dict(epoch=chosen['epoch'],eligible=chosen['eligible'],selected_for_use=selection['epoch']==chosen['epoch'],
          reason='Best trained epoch by validation macro recall only; diagnostic if rejected.')
(root/'diagnostic_selection.json').write_text(json.dumps(meta,indent=2))
if meta['selected_for_use']:
    (root/'diagnostic_metrics.json').write_text((folder/'metrics.json').read_text())
else:
    from duplexjev import Decider
    from peft import PeftModel
    import model_format as fmt
    engine=Decider.from_pretrained('/data/L202500493/duplex_postprocess/models/DuplexJev-4B-Para',device='cuda:0',attn_implementation='sdpa')
    fmt.tokenizer=engine.tok
    model=PeftModel.from_pretrained(engine.model.language_model,str(folder/('epoch_'+str(chosen['epoch'])+'_lora'))).eval()
    source=root.parent/'preference_train_v1b_20261115';metrics={}
    for split in ['validation','test','regression']:
        predictions=[]
        for row in map(json.loads,(source/(split+'.jsonl')).read_text().splitlines()):
            labels=fmt.OPTIONS[row['task']]
            code,scores=fmt.choose(model,fmt.prompt(row),[fmt.CODES[l] for l in labels])
            predicted=next(l for l in labels if fmt.CODES[l]==code)
            predictions.append(dict(id=row['id'],task=row['task'],reference=row['label'],predicted=predicted,scores=scores))
        recalls={l:sum(r['predicted']==l for r in predictions if r['reference']==l)/sum(r['reference']==l for r in predictions) for l in fmt.CODES}
        metrics[split]=dict(predictions=predictions,correct=sum(r['reference']==r['predicted'] for r in predictions),total=len(predictions),recalls=recalls,balanced_accuracy=sum(recalls.values())/len(recalls))
    assert [(r['id'],r['predicted']) for r in metrics['validation']['predictions']]==[(r['id'],r['predicted']) for r in chosen['metrics']['predictions']]
    (root/'diagnostic_metrics.json').write_text(json.dumps(metrics,indent=2))
print('DIAGNOSTIC_COMPLETE',meta,flush=True)
