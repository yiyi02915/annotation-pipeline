import json
import random
from pathlib import Path

root = Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
out = Path(__file__).resolve().parent
ids = set()
for path in sorted((root/'training/index_v1').glob('*.jsonl')):
    for line in path.open():
        ids.add(json.loads(line)['interaction_id'])
prior = {json.loads(line)['conversation_id'] for line in (out/'pairs.jsonl').read_text().splitlines()}
selected = random.Random(20261115).sample(sorted(ids-prior), 2000)
(out/'random2000.jsonl').write_text(''.join(json.dumps({'conversation_id': cid})+'\n' for cid in selected))
print(json.dumps({'available': len(ids), 'selected': len(selected), 'seed': 20261115}))
