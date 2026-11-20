import hashlib,json,shutil,sys,tarfile
from pathlib import Path
BASE=Path('/data/L202500493/duplex_postprocess')
RUN=BASE/'runs/superface_v3_20261116'
ROOT=BASE/'releases/duplex_20261116'
ROOT.mkdir(parents=True,exist_ok=True)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): assert digest(src)==digest(dst),(str(dst),'immutable_release_changed')
    else:shutil.copy2(src,dst)

original=ROOT/'original_pipeline'
for p in (BASE/'runs/step3_semantic_fix_20261115/step3').glob('*'):
    if p.is_file() and p.suffix in {'.py','.txt','.md'}:copy(p,original/'step3'/p.name)
copy(BASE/'runs/step3_semantic_fix_20261115/run_final_comparison.json',original/'evaluation.json')
sft=ROOT/'best_sft'
for task,epoch in [('gap',2),('overlap',1)]:
    for p in (RUN/f'abc/A/{task}/epoch_{epoch}').iterdir():
        if p.is_file():copy(p,sft/task/p.name)
for name in ['multimodal_model.py','audit_multimodal.py']:
    copy(RUN/name,sft/'runtime'/name)
copy(RUN/'step3/risk_annotations.py',sft/'runtime/step3/risk_annotations.py')
copy(RUN/'abc/A/selection_locked.json',sft/'selection.json')
copy(RUN/'abc/A/test_summary.json',sft/'evaluation.json')
manifest=dict(original_pipeline=dict(kind='27B text + JEV-32B synchronized mixed audio + rules',evaluation='70/76 historical regression targets; not a fresh blind test',base_models=['Qwen serving alias qwen38-27b','DuplexJev-32B-Para']),best_sft=dict(kind='text SFT A; separate gap/overlap adapters',base_model='DuplexJev-4B-Para.language_model',evaluation='404/500 automatic reference agreement',dpo_selected=False),comparison_warning='Different evaluation cohorts; percentages cannot rank the two versions head-to-head',files={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(RUN/'preserved_versions.tgz','w:gz') as t:
    for p in ROOT.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(ROOT)))
print(json.dumps(dict(release=str(ROOT),files=len(manifest['files']),bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()))))

# Local presentation clips only; these are not included in the public source repository.
sys.path.insert(0,str(RUN/'step3'))
from audio_judge import render_mix
clips=RUN/'presentation_audio';clips.mkdir(exist_ok=True)
audio_map={}
for record in json.loads((BASE/'runs/step3_semantic_fix_20261115/inputs.json').read_text()):
    cid=record['row']['conversation_id'];interval=record['row']['interval']
    path=clips/(cid+'.wav')
    render_mix({k:(Path(p),ch) for k,(p,ch) in record['tracks'].items()},interval,path)
    audio_map[cid]=dict(file=path.name,interval=interval)
rows={r['id']:r for r in json.loads((RUN/'expanded_test.json').read_text())}
pred=json.loads((RUN/'abc/A/predictions.json').read_text())
selected=[]
for task in ['gap','overlap']:
    for correct in [False,True]:
        selected.extend([r for r in pred if r['task']==task and (r['predicted']==r['reference'])==correct][:2])
for r in selected:
    source=rows[r['id']];name=hashlib.sha256(r['id'].encode()).hexdigest()[:20]+'.wav'
    shutil.copy2(source['audio'],clips/name);audio_map[r['id']]=dict(file=name,interval=source['audio_interval'])
(clips/'index.json').write_text(json.dumps(audio_map,indent=2))
with tarfile.open(RUN/'presentation_audio.tgz','w:gz') as t:
    for p in clips.iterdir():t.add(p,arcname=p.name)
print('AUDIO_READY',len(audio_map))
