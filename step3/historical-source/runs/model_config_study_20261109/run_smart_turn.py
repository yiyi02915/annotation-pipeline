import json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
D=Path(__file__).resolve().parent;B=D.parent/'finegrained_pilot_20261109'
rows=[json.loads(s) for s in (B/'benchmark_inputs.jsonl').read_text().splitlines()];rows=[e for e in rows if e['task']=='pause']
opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
t=time.perf_counter();sess=ort.InferenceSession(str(D.parents[1]/'models/smart-turn-v3.2/smart-turn-v3.2-cpu.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);fe=WhisperFeatureExtractor(chunk_length=8);load=time.perf_counter()-t
def call(e):
 audio,sr=sf.read(e['audio_clip'],dtype='float32');assert sr==16000
 audio=audio[-128000:]
 if len(audio)<128000:audio=np.pad(audio,(128000-len(audio),0))
 inp=fe(audio,sampling_rate=16000,return_tensors='np',padding='max_length',max_length=128000,truncation=True,do_normalize=True)
 prob=float(np.asarray(sess.run(None,{'input_features':inp.input_features.astype(np.float32)})[0]).reshape(-1)[0]);assert 0<=prob<=1
 return prob
t=time.perf_counter();call(rows[0]);warm=time.perf_counter()-t;t=time.perf_counter()
with (D/'smart_turn_predictions.jsonl').open('w') as f:
 for i,e in enumerate(rows):
  st=time.perf_counter();prob=call(e)
  f.write(json.dumps(dict(id=e['id'],task='pause',p_finished=prob,state='FINISHED' if prob>.5 else 'UNFINISHED',seconds=time.perf_counter()-st))+'\n');f.flush()
  if (i+1)%250==0:print('SMART_TURN',i+1,flush=True)
elapsed=time.perf_counter()-t
(D/'smart_turn_runtime.json').write_text(json.dumps(dict(n=len(rows),seconds=elapsed,events_per_second=len(rows)/elapsed,load_seconds=load,warm_seconds=warm,device='CPU',threads=4,note='End-of-turn evidence only, not hesitation/transition labels; last8sec of previous clip ending 0.5sec after left word'),indent=2))
print('SMART_TURN_COMPLETE',elapsed,flush=True)
