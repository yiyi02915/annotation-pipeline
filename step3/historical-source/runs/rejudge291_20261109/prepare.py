import json,hashlib,sys
from pathlib import Path
import soundfile as sf,numpy as np
from scipy.signal import resample_poly
from math import gcd
D=Path(__file__).resolve().parent;B=D.parent/'finegrained_pilot_20261109';S=D.parent/'model_config_study_20261109'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
q={e['id']:e for e in read(B/'qwen_predictions.jsonl')};j={e['id']:e for e in read(S/'jev32B_enum_p4_full.jsonl')}
rows=[e for e in read(B/'benchmark_inputs.jsonl') if e['task']=='pause' and q[e['id']]['label']=='HESITATION' and j[e['id']]['label']=='TURN_TRANSITION'];assert len(rows)==291
DATA=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd');idx={e['interaction_id']:e for e in read(DATA/'training/index_v1/train.jsonl')};meta=open(DATA/'seamless_fd_data/metadata.jsonl','rb');(D/'clips').mkdir(exist_ok=True)
out=[]
for e in rows:
 ix=idx[e['conversation_id']];meta.seek(ix['metadata_byte_offset']);r=json.loads(meta.read(ix['metadata_bytes']))
 a=max(0,e['anchor']-12);b=min(e['source_duration_seconds'],e['right_word']['end']+8);e['expanded_interval']=[a,b];e['tracks']={};e['old_labels']={'qwen':q[e['id']]['label'],'jev32':j[e['id']]['label']}
 key=hashlib.sha256(e['id'].encode()).hexdigest()[:20];mix=[]
 for p in r['participants']:
  sp=p['sample_id'].rsplit('_',1)[1];data,sr=sf.read(e['audio_paths'][sp],start=int(a*sf.info(e['audio_paths'][sp]).samplerate),stop=int(b*sf.info(e['audio_paths'][sp]).samplerate),dtype='float32');data=data.mean(1) if data.ndim>1 else data
  if sr!=16000:g=gcd(sr,16000);data=resample_poly(data,16000//g,sr//g)
  path=D/'clips'/f'{key}_{sp}.wav';sf.write(path,data,16000);mix.append(data)
  words=[dict(id=f'{sp}:w{i}',text=w['word'],start=w['start_sec'],end=w['end_sec'],score=w.get('score',1)) for i,w in enumerate(p['words']) if w['start_sec']>=a and w['end_sec']<=b]
  e['tracks'][sp]={'audio':str(path),'words':words,'text':' '.join(w['text'] for w in words),'rms':float(np.sqrt(np.mean(data**2))),'finite':bool(np.isfinite(data).all())}
 n=min(map(len,mix));mixed=(mix[0][:n]+mix[1][:n])*.5;path=D/'clips'/f'{key}_both.wav';sf.write(path,mixed,16000);e['mixed_audio']=str(path);out.append(e)
with (D/'inputs.jsonl').open('w') as f:
 for e in out:f.write(json.dumps(e)+'\n')
print('PREPARED',len(out),flush=True)
