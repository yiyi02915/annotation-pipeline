"""Matched A/B/C experiment. Frozen references, resumable checkpoints, separate risk flags."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from multimodal_model import Model, OPTIONS

ROOT=Path(__file__).resolve().parent


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(path)


def metrics(rows):
    classes=sorted({r['reference'] for r in rows})
    detail={}
    for c in classes:
        tp=sum(r['reference']==c and r['predicted']==c for r in rows)
        ref=sum(r['reference']==c for r in rows);pred=sum(r['predicted']==c for r in rows)
        detail[c]=dict(correct=tp,total=ref,predicted=pred,recall=tp/ref,precision=tp/pred if pred else 0,f1=2*tp/(ref+pred))
    wrong=[r for r in rows if r['reference']!=r['predicted']]
    flagged=[r for r in rows if r['risk_flag']]
    detected=sum(r['risk_flag'] for r in wrong)
    return dict(total=len(rows),correct=len(rows)-len(wrong),reference_agreement=(len(rows)-len(wrong))/len(rows) if rows else None,macro_f1=sum(r['f1'] for r in detail.values())/len(detail) if detail else 0,classes=detail,risk=dict(flagged=len(flagged),errors_detected=detected,error_recall=detected/len(wrong) if wrong else None,flag_precision=detected/len(flagged) if flagged else None,false_flags=len(flagged)-detected,unflagged=len(rows)-len(flagged)),reference_type='automatic references; not human accuracy')


@torch.no_grad()
def evaluate(model,rows,audio):
    model.model.language_model.eval();result=[];start=time.monotonic()
    for row in rows:
        begin=time.monotonic();scores=model.scores(row,audio);probs=scores.softmax(-1)
        labels=OPTIONS[row['task']];prediction=labels[int(scores.argmax())]
        ordered=probs.sort(descending=True).values
        margin=float(ordered[0]-ordered[1]);flags=list(row.get('risk_reasons',[]))
        if margin<.2:flags.append('small_option_score_margin')
        if audio:
            text_scores=model.scores(row,False)
            if labels[int(text_scores.argmax())]!=prediction:flags.append('audio_text_ablation_disagreement')
        result.append(dict(id=row['id'],conversation_id=row['conversation_id'],task=row['task'],reference=row['label'],predicted=prediction,risk_flag=bool(flags),risk_reasons=flags,option_margin=margin,confidence_calibrated=False,seconds=time.monotonic()-begin,audio_seconds=row['audio_interval'][1]-row['audio_interval'][0]))
    return dict(metrics=metrics(result),predictions=result,wall_seconds=time.monotonic()-start,throughput_targets_per_minute=len(result)*60/max(.001,time.monotonic()-start))


def fit_task(model,arm,task,train,val):
    out=ROOT/'abc'/arm/task;out.mkdir(parents=True,exist_ok=True)
    done=out/'complete.json'
    if done.exists():return json.loads(done.read_text())
    audio=arm!='A';is_dpo=arm=='C'
    initial=None
    if is_dpo:initial=json.loads((ROOT/'abc/B'/task/'complete.json').read_text())['best_adapter']
    data_hash=hashlib.sha256(json.dumps([(r['id'],r['label']) for r in train]).encode()).hexdigest()
    lr=2e-5 if is_dpo else 1e-4
    model.adapter(initial);model.model.language_model.eval()
    margins={}
    if is_dpo:
        margin_file=out/'fixed_reference_margins.json'
        if margin_file.exists():margins=json.loads(margin_file.read_text())
        else:
            with torch.no_grad():
                for row in train:
                    scores=model.scores(row,True);labels=OPTIONS[task];chosen=labels.index(row['label'])
                    rejected=max((i for i in range(len(labels)) if i!=chosen),key=lambda i:float(scores[i]))
                    margins[row['id']]=dict(chosen=chosen,rejected=rejected,margin=float(scores[chosen]-scores[rejected]))
            save(margin_file,margins)
    latest=out/'latest.json';start_epoch=0;start_offset=0;updates=0
    best=dict(score=-1,adapter=None);history=[]
    if latest.exists():
        pointer=json.loads(latest.read_text());checkpoint=Path(pointer['checkpoint'])
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
        assert state['data_hash']==data_hash
        model.adapter(checkpoint/'adapter');start_epoch=state['epoch'];start_offset=state['offset'];updates=state['updates'];best=state['best'];history=state['history']
    optimizer=torch.optim.AdamW(model.trainable(),lr=lr)
    if latest.exists():
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        baseline=evaluate(model,val,audio);base=out/'initial_adapter';model.model.language_model.save_pretrained(base)
        best=dict(score=baseline['metrics']['macro_f1'],adapter=str(base));history.append(dict(epoch=0,validation=baseline['metrics']))
        save(out/'validation_initial.json',baseline)
    begin=time.monotonic();epochs=1 if is_dpo else 2
    def checkpoint(epoch,offset):
        target=out/'checkpoints'/f'update_{updates:05d}';target.mkdir(parents=True,exist_ok=True)
        model.model.language_model.save_pretrained(target/'adapter')
        torch.save(dict(optimizer=optimizer.state_dict(),epoch=epoch,offset=offset,updates=updates,best=best,history=history,data_hash=data_hash,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),target/'state.pt')
        save(out/'latest.json',dict(checkpoint=str(target),updates=updates,epoch=epoch,offset=offset))
    for epoch in range(start_epoch,epochs):
        order=list(train);random.Random(20261116+epoch).shuffle(order)
        offset=start_offset if epoch==start_epoch else 0
        model.model.language_model.train()
        for lo in range(offset,len(order),4):
            group=order[lo:lo+4];optimizer.zero_grad(set_to_none=True);losses=[]
            for row in group:
                if is_dpo:
                    p=margins[row['id']];s=model.scores(row,True)
                    loss=-torch.nn.functional.logsigmoid(.1*(s[p['chosen']]-s[p['rejected']]-p['margin']))
                else:loss=model.loss(row,audio)
                assert torch.isfinite(loss)
                (loss/len(group)).backward();losses.append(float(loss.detach()))
            model.assert_frozen();torch.nn.utils.clip_grad_norm_(model.trainable(),1.0);optimizer.step();updates+=1
            if updates%25==0:
                checkpoint(epoch,lo+len(group));print('TRAIN',arm,task,epoch,lo+len(group),len(order),sum(losses)/len(losses),flush=True)
        evaluation=evaluate(model,val,audio)
        candidate=out/f'epoch_{epoch+1}';model.model.language_model.save_pretrained(candidate)
        history.append(dict(epoch=epoch+1,validation=evaluation['metrics']))
        save(out/f'validation_epoch_{epoch+1}.json',evaluation)
        if evaluation['metrics']['macro_f1']>best['score']:
            best=dict(score=evaluation['metrics']['macro_f1'],adapter=str(candidate))
        checkpoint(epoch+1,0)
    summary=dict(arm=arm,task=task,best_adapter=best['adapter'],best_validation_macro_f1=best['score'],history=history,train_inputs=len(train),updates=updates,data_hash=data_hash,active_seconds_this_process=time.monotonic()-begin,reference_fixed_to=initial,dpo_replaced_sft=is_dpo and best['adapter']!=str(out/'initial_adapter'))
    save(done,summary);return summary


def main(arm):
    assert json.loads((ROOT/'checkpoint_abc_engineering.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'novelty_check.json').read_text())['status']=='passed'
    lock=json.loads((ROOT/'reference_lock.json').read_text())
    for name,digest in lock['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    new_training=json.loads((ROOT/'expanded_train.json').read_text())
    train=[r for r in new_training if r['quality']['preference_eligible']]
    audited=[r for r in json.loads((ROOT/'revised_train.json').read_text()) if r['audit_required'] and r['audio_ready'] and r['quality']['preference_eligible']]
    train+=audited
    val=json.loads((ROOT/'expanded_validation.json').read_text())
    counts=Counter(r['label'] for r in train)
    assert all(counts[c]>=10 for c in ['PAUSE','TURN_TRANSITION','INTERRUPT','BACKCHANNEL','NEITHER']),f'insufficient_qualified_training_examples:{counts}'
    save(ROOT/'abc'/arm/'matched_training_manifest.json',dict(ids=[r['id'] for r in train],counts=dict(counts),reused_audited_inputs=len(audited),excluded_new_training=sum(not r['quality']['preference_eligible'] for r in new_training),criterion='same cross-modal stable inputs for all A/B/C; no seed fallback'))
    torch.manual_seed(20261116);model=Model();selected={}
    for task in ['gap','overlap']:
        torch.manual_seed(20261116)
        selected[task]=fit_task(model,arm,task,[r for r in train if r['task']==task],[r for r in val if r['task']==task])
    save(ROOT/'abc'/arm/'selection_locked.json',selected)
    # Test references are read only after both task checkpoints are selected.
    test=json.loads((ROOT/'expanded_test.json').read_text());all_predictions=[];time_total=0
    for task in ['gap','overlap']:
        model.adapter(selected[task]['best_adapter'])
        result=evaluate(model,[r for r in test if r['task']==task],arm!='A')
        save(ROOT/'abc'/arm/f'test_{task}.json',result);all_predictions+=result['predictions'];time_total+=result['wall_seconds']
    save(ROOT/'abc'/arm/'test_summary.json',dict(metrics=metrics(all_predictions),wall_seconds=time_total,targets_per_minute=len(all_predictions)*60/max(.001,time_total),prediction_risk_method='predefined score margin <0.2 or audio/text ablation disagreement; no reference label used to flag risk',reference_limit='automatic teacher references may share biases'))
    save(ROOT/'abc'/arm/'risk_subset.json',[r for r in all_predictions if r['risk_flag']])
    save(ROOT/'abc'/arm/'predictions.json',all_predictions)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['A','B','C']);a=p.parse_args();main(a.arm)
