import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
outputs=root.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
locked=read(outputs/'step3_validation_20261115/benchmark_reference.json')['recordings']
blocked=set(locked);records={};sources={};excluded=0
def add(uid,cid,task,payload,label,source):
    global excluded
    if cid in blocked:excluded+=1;return
    if uid in records:
        assert records[uid]['label']==label,'Reference conflict'
        return
    records[uid]=dict(id=uid,conversation_id=cid,task=task,input=payload,label=label,source=source)
folder=outputs/'ob141_smoketest_20261112'
for prefix in ('','holdout_','blind100_','risk_blind_'):
    refs=read(folder/(prefix+'references.json'))
    inputs=read(folder/(prefix+'inputs.json'))
    sources[prefix+'ob141']=hashlib.sha256((folder/(prefix+'references.json')).read_bytes()).hexdigest()
    for item in inputs:
        r=refs['labels'].get(item['id'])
        if r and r.get('evidence_strength')=='clear':
            add(item['id'],item['conversation_id'],'gap',item['payload'],r['label'],refs['source'])
folder=outputs/'codex_random20_20261112'
for rf,inf in [('codex_reference_locked.json','blind_inputs.json'),('holdout_reference_locked.json','holdout_inputs.json')]:
    refs={r['id']:r for r in read(folder/rf)}
    for item in read(folder/inf):
        r=refs[item['id']];add(item['id'],item['id'].split(':')[0],'gap',item['payload'],r['label'],r.get('basis','Codex reference'))
folder=outputs/'event_multi_v1_20261109/paired_holdout_20261113'
refs=read(folder/'reference.locked.json')
for file in ('sample.json','supplement.json'):
    for case in read(folder/file)['cases']:
        for c in refs['clips'][case['number']]['candidates']:
            if c['label']=='UNSCORED':continue
            payload=dict(target={k:v for k,v in c.items() if k not in ('label','ordinary_entry','evidence','reason')},context=case['context'])
            add(c['candidate_id'],case['conversation_id'],'overlap',payload,c['label'],refs['source'])
splits={'train':[],'validation':[],'test':[],'regression':[]}
for record in records.values():
    n=int(hashlib.sha256(record['conversation_id'].encode()).hexdigest()[:8],16)%100
    split='train' if n<70 else 'validation' if n<85 else 'test'
    splits[split].append(record)
inputs={r['row']['conversation_id']:r for r in read(outputs/'step3_semantic_fix_20261115/inputs.json')}
for cid,r in locked.items():
    if 'gap' in r:
        g=next(g for g in inputs[cid]['gaps'] if g['target']['left_word_id']==r['gap']['target']['left_word_id'])
        splits['regression'].append(dict(id=g['id'],conversation_id=cid,task='gap',input=g,label=r['gap']['label']))
    else:
        for c in r['events']:
            if c['label']=='UNSCORED':continue
            splits['regression'].append(dict(id=c['candidate_id'],conversation_id=cid,task='overlap',input=dict(target={k:v for k,v in c.items() if k not in ('label','ordinary_entry','ordinal')},context=inputs[cid]['cases'][0]['context']),label=c['label']))
groups={k:{r['conversation_id'] for r in v} for k,v in splits.items()}
for k,a in groups.items():
    for l,b in groups.items():
        if k!=l:assert not a&b
for name,rows in splits.items():
    (root/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
manifest=dict(counts={k:len(v) for k,v in splits.items()},dialogues={k:len(v) for k,v in groups.items()},excluded_regression_records=excluded,source_hashes=sources,reference_type='Codex transcript/context adjudication, not human gold',split='SHA256(original conversation id) modulo100: train<70 validation<85 otherwise test; current30 excluded before splitting')
(root/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
