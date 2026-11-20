import json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
D=Path(__file__).resolve().parent;B=D.parent/'finegrained_pilot_20261109'
rows=[e for e in map(json.loads,(B/'benchmark_inputs.jsonl').read_text().splitlines()) if e['task']=='pause']
opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
sess=ort.InferenceSession(str(D.parents[1]/'models/smart-turn-v3.2/smart-turn-v3.2-cpu.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);fe=WhisperFeatureExtractor(chunk_length=8)
start=time.perf_counter()
with (D/'smart_turn_tail02.jsonl').open('w') as f:
 for i,e in enumerate(rows):
  audio,sr=sf.read(e['audio_clip'],dtype='float32');assert sr==16000
  # Original clip interval ends at anchor+0.5; retain exactly anchor+0.2.
  desired=e['anchor']+.2-e['audio_interval'][0];audio=audio[:round(desired*sr)][-128000:]
  if len(audio)<128000:audio=np.pad(audio,(128000-len(audio),0))
  inp=fe(audio,sampling_rate=16000,return_tensors='np',padding='max_length',max_length=128000,truncation=True,do_normalize=True)
  prob=float(np.asarray(sess.run(None,{'input_features':inp.input_features.astype(np.float32)})[0]).reshape(-1)[0]);assert 0<=prob<=1
  f.write(json.dumps(dict(id=e['id'],task='pause',p_finished=prob,state='FINISHED' if prob>.5 else 'UNFINISHED',tail_seconds=.2))+'\n')
  if (i+1)%250==0:f.flush();print('TAIL02',i+1,flush=True)
print('TAIL02_COMPLETE',time.perf_counter()-start,flush=True)
