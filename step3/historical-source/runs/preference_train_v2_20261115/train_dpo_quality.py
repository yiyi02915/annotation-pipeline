"""Controlled DPO continuation; only training labels construct pairs.

All arms start at the same SFT checkpoint and use the same fixed reference.
Evaluation labels are read only after checkpoint selection on validation.
"""
import argparse, collections, gc, hashlib, json, math, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from duplexjev import Decider
from peft import PeftModel
import model_format as fmt

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--variant',choices=['dedup','anchored'],required=True)
p.add_argument('--base',default='/data/L202500493/duplex_postprocess/models/DuplexJev-4B-Para')
args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
random.seed(20261115);torch.manual_seed(20261115);start=time.monotonic()
def save(name,value):
    path=args.output/name; temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def rows(split):return [json.loads(l) for l in (args.source/(split+'.jsonl')).read_text().splitlines()]
train=rows('train'); validation=rows('validation')
assert not {r['conversation_id'] for r in train}&{r['conversation_id'] for r in validation}
config=dict(variant=args.variant,seed=20261115,learning_rate=5e-6,beta=.2,epochs=2,
            label_smoothing=.1 if args.variant=='anchored' else 0.,
            chosen_nll_weight=.2 if args.variant=='anchored' else 0.,
            disputed_pair_weight=.25 if args.variant=='anchored' else 1.,
            sampling='one hardest SFT-negative per unique input per epoch',
            selection='validation macro recall; no class recall below initial SFT; ties prefer earlier',
            effective_batch=4,reference='fixed initial SFT; precomputed log probabilities',
            reference_labels='historical Codex adjudication, not human gold',
            modality='text decoder only',base=args.base,
            input_hashes={s:hashlib.sha256((args.source/(s+'.jsonl')).read_bytes()).hexdigest() for s in ['train','validation','test','regression']})
save('config.json',config)
engine=Decider.from_pretrained(args.base,device='cuda:0',attn_implementation='sdpa')
fmt.tokenizer=engine.tok;base=engine.model.language_model
engine.model.language_model=None;del engine;gc.collect();torch.cuda.empty_cache()
assert all(len(fmt.tokenizer(c,add_special_tokens=False)['input_ids'])==1 for c in fmt.CODES.values())
model=PeftModel.from_pretrained(base,str(args.source/'policy_sft_lora'),is_trainable=True)
model.config.use_cache=False
old_pairs=json.loads((args.source/'preference_pairs.json').read_text())
verified={(r['id'],r['chosen'],r['rejected']) for r in old_pairs if r.get('source')!='independent_seed_fallback'}
pairs=[]
for row in train:
    labels=fmt.OPTIONS[row['task']];prefix=fmt.prompt(row)
    _,scores=fmt.choose(model,prefix,[fmt.CODES[label] for label in labels])
    scores=dict(zip(labels,scores));wrong=max((l for l in labels if l!=row['label']),key=scores.get)
    key=(row['id'],row['label'],wrong)
    pairs.append(dict(id=row['id'],task=row['task'],chosen=row['label'],rejected=wrong,
                      prompt=prefix,reference_margin=scores[row['label']]-scores[wrong],
                      judge_verified=key in verified))
assert len(pairs)==len({r['id'] for r in pairs})
# Smooth importance weights, without physically repeating minority examples.
counts=collections.Counter(r['chosen'] for r in pairs)
for row in pairs:
    row['weight']=min(2.,max(.5,math.sqrt(len(pairs)/(5*counts[row['chosen']]))))
    if not row['judge_verified']:row['weight']*=config['disputed_pair_weight']
average=sum(r['weight'] for r in pairs)/len(pairs)
for row in pairs:row['weight']/=average
save('pairs.json',pairs)
save('pair_audit.json',dict(unique_inputs=len(train),unique_pairs=len(pairs),
     judge_verified=sum(r['judge_verified'] for r in pairs),class_counts=counts,
     disputed=[{k:r[k] for k in ['id','chosen','rejected','weight']} for r in pairs if not r['judge_verified']]))
def evaluate(data):
    result=[]
    for row in data:
        labels=fmt.OPTIONS[row['task']]
        code,scores=fmt.choose(model,fmt.prompt(row),[fmt.CODES[l] for l in labels])
        label=next(l for l in labels if fmt.CODES[l]==code)
        result.append(dict(id=row['id'],task=row['task'],reference=row['label'],predicted=label,scores=scores))
    recalls={l:sum(r['predicted']==l for r in result if r['reference']==l)/sum(r['reference']==l for r in result) for l in fmt.CODES if any(r['reference']==l for r in result)}
    return dict(correct=sum(r['reference']==r['predicted'] for r in result),total=len(result),
                balanced_accuracy=sum(recalls.values())/len(recalls),recalls=recalls,predictions=result)
initial=evaluate(validation)
expected=json.loads((args.source/'metrics.json').read_text())['sft']['validation']['predictions']
assert [(r['id'],r['predicted']) for r in initial['predictions']]==[(r['id'],r['predicted']) for r in expected], 'Reloaded SFT differs from control'
history=[dict(epoch=0,metrics=initial,eligible=True)]
best=initial['balanced_accuracy'];selected=0
model.save_pretrained(args.output/'selected_lora')
model.gradient_checkpointing_enable();model.enable_input_require_grads()
optim=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=config['learning_rate'])
losses=[];step=0;total_steps=math.ceil(len(pairs)/4)*config['epochs']
for epoch in range(1,config['epochs']+1):
    order=list(pairs);random.Random(20261115+epoch).shuffle(order);model.train()
    for offset in range(0,len(order),4):
        batch=order[offset:offset+4];optim.zero_grad()
        for row in batch:
            chosen=fmt.logp(model,row['prompt'],fmt.CODES[row['chosen']])
            rejected=fmt.logp(model,row['prompt'],fmt.CODES[row['rejected']])
            z=config['beta']*(chosen-rejected-row['reference_margin'])
            eps=config['label_smoothing']
            preference=-(1-eps)*F.logsigmoid(z)-eps*F.logsigmoid(-z)
            loss=row['weight']*(preference-config['chosen_nll_weight']*chosen/2)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/len(batch)).backward()
            losses.append(dict(epoch=epoch,id=row['id'],loss=float(loss.detach()),
                               preference=float(preference.detach()),chosen_logp=float(chosen.detach()),
                               margin_delta=float((z/config['beta']).detach())))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        step+=1
        # Short warm-up and linear decay, fixed before looking at validation.
        multiplier=min(1.,step/4)*max(.1,1-(step-1)/total_steps)
        for group in optim.param_groups:group['lr']=config['learning_rate']*multiplier
        optim.step()
        if step%10==0:print(args.variant,'epoch',epoch,'update',step,'of',total_steps,flush=True)
    val=evaluate(validation)
    eligible=all(val['recalls'][l]+1e-12>=r for l,r in initial['recalls'].items())
    history.append(dict(epoch=epoch,metrics=val,eligible=eligible))
    model.save_pretrained(args.output/('epoch_'+str(epoch)+'_lora'))
    if eligible and val['balanced_accuracy']>best:
        selected=epoch;best=val['balanced_accuracy'];model.save_pretrained(args.output/'selected_lora')
    save('validation_history.json',history)
    print('VALIDATION',epoch,val['correct'],val['balanced_accuracy'],'eligible',eligible,'selected',selected,flush=True)
save('selection.json',dict(epoch=selected,validation_macro_recall=best,rule=config['selection']))
# Freeze the selection before reading test/regression targets.
base=model.unload();del model;gc.collect();torch.cuda.empty_cache()
model=PeftModel.from_pretrained(base,str(args.output/'selected_lora')).eval()
metrics={s:evaluate(rows(s)) for s in ['validation','test','regression']}
save('metrics.json',metrics);save('losses.json',losses)
save('complete.json',dict(status='complete',wall_seconds=time.monotonic()-start,selected_epoch=selected))
print('COMPLETE',args.variant,flush=True)
