import json
from pathlib import Path
D=Path(__file__).resolve().parent
data=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
cases=json.loads((D/'cases.json').read_text());wanted={c['id'].split(':')[0] for c in cases}
indices={}
for line in (data/'training/index_v1/train.jsonl').open():
 r=json.loads(line)
 if r['interaction_id'] in wanted:indices[r['interaction_id']]=r
out=[]
with (data/'seamless_fd_data/metadata.jsonl').open('rb') as f:
 for c in cases:
  cid=c['id'].split(':')[0];ix=indices[cid];f.seek(ix['metadata_byte_offset']);row=json.loads(f.read(ix['metadata_bytes']));streams={}
  for p in row['participants']:
   sp=p['sample_id'].rsplit('_',1)[1];streams[sp]=[dict(id=f'{sp}:w{i}',text=w['word'],start=float(w['start_sec']),end=float(w['end_sec'])) for i,w in enumerate(p['words']) if w['end_sec']>c['interval'][0]-6 and w['start_sec']<c['interval'][1]+4]
  out.append(dict(id=c['id'],number=c['number'],conversation_id=cid,interval=c['interval'],target=c['target'],streams=streams,original_label=c['code'],context=c['context']))
(D/'word_inputs.json').write_text(json.dumps(out,indent=2));print('WORD_INPUTS',len(out))
