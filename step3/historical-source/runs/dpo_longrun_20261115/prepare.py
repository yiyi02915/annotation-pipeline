import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;source=root.parent/'preference_train_v1b_20261115';v2=root.parent/'preference_train_v2_20261115'
data=root/'data';data.mkdir(exist_ok=True)
files={s+'.jsonl':source/(s+'.jsonl') for s in ['train','validation','test','regression']}
files.update({'baseline_metrics.json':source/'metrics.json','teacher_targeted.json':v2/'teacher_targeted.json','hard_pairs.json':v2/'dedup/pairs.json'})
for name,path in files.items():
    target=data/name
    if target.exists():assert target.read_bytes()==path.read_bytes(),'Refusing to mutate snapshotted inputs'
    else:shutil.copyfile(path,target)
base=dict(seed=20261115,max_epochs=8,checkpoint_every=20,patience=3,batch=4,smoothing=.1)
specs=[
 dict(id='r01_seed_balanced',init='sft',teacher=False,lr=2e-5,beta=.1,nll=.2),
 dict(id='r02_teacher_floor',init='sft',teacher=True,lr=2e-5,beta=.1,nll=.2),
 dict(id='r03_teacher_anchor',init='sft',teacher=True,lr=2e-5,beta=.2,nll=1.),
 dict(id='r04_seed_regularized',init='sft',teacher=False,lr=1e-5,beta=.3,nll=.5),
 dict(id='r05_continue_dpo',init='dpo',teacher=True,lr=5e-6,beta=.3,nll=1.),
 dict(id='r06_continue_seed',init='dpo',teacher=False,lr=1e-5,beta=.3,nll=.5)]
plan=dict(base='/data/L202500493/duplex_postprocess/models/DuplexJev-4B-Para',
    adapters={k:'/data/L202500493/duplex_postprocess/runs/preference_train_v1b_20261115/'+v for k,v in {'sft':'policy_sft_lora','dpo':'policy_dpo_lora'}.items()},
    gpus=[2,3],max_wall_hours=4,created='2026-11-15',jobs=[dict(base,**s) for s in specs],
    description='Six bounded comparisons; 164 unique inputs/237 unique preference pairs; no reference label edits or new blind data. Keep every checkpoint. Validation-only selection; final held-out evaluation once.')
(root/'plan.final.json').write_text(json.dumps(plan,indent=2))
probe=dict(base,id='resume_probe',init='sft',teacher=False,lr=1e-5,beta=.1,nll=.2)
probe.update(max_epochs=1,checkpoint_every=1)
(root/'plan.json').write_text(json.dumps(dict(plan,jobs=[probe]),indent=2))
print('SNAPSHOT_AND_PLAN_READY')
