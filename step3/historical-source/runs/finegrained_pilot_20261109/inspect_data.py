import json
from pathlib import Path
r=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
with (r/'seamless_fd_data/metadata.jsonl').open() as f:d=json.loads(next(f))
print(json.dumps({k:v for k,v in d.items() if k!='participants'},indent=2))
for p in d['participants']:
 print(json.dumps({k:v for k,v in p.items() if k!='words'},indent=2))
 print('WORDS',len(p['words']))
for split in ['train','dev','test']:
 with (r/'training/index_v1'/f'{split}.jsonl').open() as f:print(split,next(f)[:2500])
