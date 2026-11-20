"""Automatic independent references, locked before A/B/C inference."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from audit_multimodal import text_judge, evidence, RULES, OPTIONS
from risk_annotations import annotate

ROOT=Path(__file__).resolve().parent


def records(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value)+'\n');f.flush()


def text_phase(rows):
    path=ROOT/'expanded_text.jsonl';done={r['key'] for r in records(path)}
    jobs=[(r,k) for r in rows for k in range(2) if f"{r['id']}|{k}" not in done]
    def one(x):
        r,k=x;return dict(key=f"{r['id']}|{k}",id=r['id'],result=text_judge(r,bool(k)))
    with ThreadPoolExecutor(4) as pool:
        for i,result in enumerate(pool.map(one,jobs),1):
            append(path,result)
            if i%25==0:print('EXPANDED_TEXT',len(done)+i,2*len(rows),flush=True)


def audio_phase(rows):
    from duplexjev import Decider,Question
    path=ROOT/'expanded_audio.jsonl';done={r['id'] for r in records(path)}
    engine=Decider.from_pretrained('/data/L202500493/duplex_postprocess/models/DuplexJev-32B-Para',device='cuda:0',attn_implementation='sdpa')
    for row in rows:
        if row['id'] in done:continue
        begin=time.monotonic()
        try:
            q=Question('target',RULES+' Judge the exact specified target.',OPTIONS[row['task']],audio='mix')
            answer=engine.decide_batch({'mix':row['audio']},[q],context={'mix':json.dumps(evidence(row))},lang='en',n_perm=4,seed=20261116,max_items=1)['mix']['target']
            result=dict(status='ok',**answer)
        except Exception as exc:result=dict(status='error',error=str(exc))
        append(path,dict(id=row['id'],result=result,seconds=time.monotonic()-begin))
        done.add(row['id'])
        if len(done)%25==0:print('EXPANDED_AUDIO',len(done),len(rows),flush=True)


def finalize(rows):
    lock=ROOT/'reference_lock.json'
    if lock.exists():
        frozen=json.loads(lock.read_text())
        for name,digest in frozen['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        return
    text={r['key']:r['result'] for r in records(ROOT/'expanded_text.jsonl')}
    audio={r['id']:r['result'] for r in records(ROOT/'expanded_audio.jsonl')}
    path=ROOT/'expanded_adjudications.jsonl';done={r['id']:r for r in records(path)}
    def adjudicate(row):
        votes=[text.get(f"{row['id']}|{k}",{}) for k in range(2)]
        aud=audio.get(row['id'],{});av=aud.get('answer');labels=[v.get('label') for v in votes]
        initial=next((x for x in labels if x in OPTIONS[row['task']]),None)
        flags=[];adjudication=None
        stable=initial is not None and labels==[initial]*2 and av==initial
        if stable:final=initial
        else:
            adjudication=text_judge(row,supplemental=dict(text_observations=votes,audio_observation=aud))
            final=adjudication.get('label')
        if final not in OPTIONS[row['task']]:
            # No artificial default label: this is a recorded technical failure, not a third business class.
            out=dict(id=row['id'],status='failed',reason='no_valid_model_output',text=votes,audio=aud)
        else:
            initial=initial or final
            quality=annotate(row['task'],initial,final,labels,av,flags,not stable)
            out=dict(id=row['id'],status='ok',label=final,quality=quality,text=votes,audio=aud,adjudication=adjudication)
        return out
    with ThreadPoolExecutor(4) as pool:
        for out in pool.map(adjudicate,[r for r in rows if r['id'] not in done]):
            append(path,out);done[out['id']]=out
            if len(done)%25==0:print('REFERENCE',len(done),len(rows),flush=True)
    sha={};summary={}
    for split in ['train','validation','test']:
        selected=[]
        for row in rows:
            if row['split']!=split:continue
            r=done[row['id']]
            if r['status']!='ok':continue
            selected.append(dict(row,label=r['label'],quality=r['quality'],reference_type='independent automatic text/audio adjudication; not human gold'))
        name=f'expanded_{split}.json'
        (ROOT/name).write_text(json.dumps(selected,indent=2))
        sha[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        summary[split]=dict(total=len(selected),labels=dict(Counter(r['label'] for r in selected)),preference_eligible=sum(r['quality']['preference_eligible'] for r in selected),risk_flagged=sum(r['quality']['risk_flag'] for r in selected))
    (ROOT/'expanded_risk_subset.json').write_text(json.dumps([r for r in done.values() if r.get('quality',{}).get('risk_flag')],indent=2))
    lock.write_text(json.dumps(dict(sha256=sha,counts=summary,locked_before_abc=True,technical_failures=sum(r['status']!='ok' for r in done.values()),reference_type='automatic; not independently human-validated'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['text','audio','finalize']);args=parser.parse_args()
    rows=json.loads((ROOT/'expanded_inputs.json').read_text())
    {'text':text_phase,'audio':audio_phase,'finalize':finalize}[args.phase](rows)
