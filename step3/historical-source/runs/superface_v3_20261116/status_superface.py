"""Compact status and a results-only snapshot, without copying audio or model weights."""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent


def main():
    checkpoints=[json.loads(x) for x in (ROOT/'checkpoints.jsonl').read_text().splitlines()]
    stages={x['stage']:x for x in checkpoints}
    counts={};errors={}
    for name in ['expanded_text.jsonl','expanded_audio.jsonl','expanded_adjudications.jsonl']:
        p=ROOT/name
        counts[name]=sum(1 for line in p.open() if line.strip()) if p.exists() else 0
        errors[name]=0
        if p.exists():
            for line in p.open():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r.get('result',r).get('status') in {'error','failed'}:errors[name]+=1
    status=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=stages,reference_progress=counts,reference_failures=errors)
    for name in ['checkpoint_01_preparation.json','checkpoint_02_audit.json','checkpoint_03_smoke32.json','checkpoint_04_expansion.json','reference_lock.json','checkpoint_06_abc.json']:
        if (ROOT/name).exists():status[name]=json.loads((ROOT/name).read_text())
    (ROOT/'status.json').write_text(json.dumps(status,indent=2))
    versions={}
    for key in ['torch','transformers','peft','duplexjev']:
        try:versions[key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:versions[key]='not_registered'
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    (ROOT/'runtime_versions.json').write_text(json.dumps(dict(packages=versions,source_sha256=sources),indent=2))
    names=['status.json','runtime_versions.json','checkpoints.jsonl','alignment21.json','audit21_results.json','audit21_inputs.json','audit_text.json','audit_audio.json','checkpoint_01_preparation.json','checkpoint_02_audit.json','checkpoint_03_smoke32.json','checkpoint_04_expansion.json','checkpoint_abc_engineering.json','reference_lock.json','checkpoint_06_abc.json','smoke32/gap/verification.json','smoke32/overlap/verification.json']
    with tarfile.open(ROOT/'progress_snapshot.tgz','w:gz') as tar:
        for name in names:
            p=ROOT/name
            if p.exists():tar.add(p,arcname=name)
    print(json.dumps(dict(stages={k:v['status'] for k,v in stages.items()},counts=counts)))


if __name__=='__main__':main()
