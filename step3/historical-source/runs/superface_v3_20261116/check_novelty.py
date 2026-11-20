"""Cross-check new conversations against other historical experiment manifests."""
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'expanded_inputs.json').read_text())
new={r['conversation_id'] for r in rows};collisions={};checked=0
for folder in ROOT.parent.iterdir():
    if not folder.is_dir() or folder==ROOT:continue
    for path in folder.rglob('*'):
        if not path.is_file() or path.suffix not in {'.json','.jsonl'}:continue
        if any(p in {'dataset','priority','models','audio','checkpoints','full_annotation'} for p in path.relative_to(folder).parts):continue
        if path.stat().st_size>100_000_000:continue
        if not any(k in path.name.lower() for k in ['input','case','reference','sample','selected','selection','pairs','prior_dialogue','annotation','manifest']):continue
        used=set(re.findall(r'V\d{2}_S\d+_I\d+',path.read_text(errors='replace')))
        checked+=1
        for cid in used&new:collisions.setdefault(cid,[]).append(str(path))
report=dict(checked_files=checked,new_conversations=len(new),collisions=collisions,status='passed' if not collisions else 'failed',scope='all prior run manifests matching experiment input/reference/case names; raw dataset/index files excluded')
(ROOT/'novelty_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
assert not collisions,'historical conversation reuse found'
