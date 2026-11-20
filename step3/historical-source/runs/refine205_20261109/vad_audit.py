import sys,json,time
from pathlib import Path
import numpy as np,soundfile as sf,torch
from scipy.signal import resample_poly
D=Path(__file__).resolve().parent
sys.path.insert(0,'/data/L202500293/obed/models/silero-vad/upstream/src')
from silero_vad.utils_vad import get_speech_timestamps
model=torch.jit.load('/data/L202500293/obed/models/silero-vad/checkpoints/silero_vad.jit',map_location='cpu');torch.set_num_threads(2)
events=[json.loads(s) for s in (D/'events.jsonl').read_text().splitlines()];out=[]
for e in events:
 p=e['payload'];l,r=p['target']['clip_relative_gap'];x,sr=sf.read(e['audio_tracks'][p['focus']],dtype='float32')
 if sr!=16000:
  import math
  g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g);sr=16000
 probes=[]
 for threshold in [.35,.65]:
  spans=get_speech_timestamps(torch.from_numpy(x),model,threshold=threshold,sampling_rate=16000,min_speech_duration_ms=64,min_silence_duration_ms=64,speech_pad_ms=0,return_seconds=False)
  spans=[{'start':s['start']/sr,'end':s['end']/sr} for s in spans];left=min(spans,key=lambda s:abs(s['end']-l),default=None);right=min(spans,key=lambda s:abs(s['start']-r),default=None)
  inside=sum(max(0,min(s['end'],r-.1)-max(s['start'],l+.1)) for s in spans) if r-l>.2 else 0
  probes.append({'threshold':threshold,'speech_spans':spans,'nearest_left_end':left['end'] if left else None,'nearest_right_start':right['start'] if right else None,'speech_inside_gap_seconds':round(inside,3),'boundary_supported':bool(left and right and abs(left['end']-l)<=.2 and abs(right['start']-r)<=.2)})
 # Positive speech inside the alleged pause at both VAD thresholds is a conflict.
 conflict=r-l>.5 and all(p['speech_inside_gap_seconds']>.2 for p in probes)
 out.append({'id':e['id'],'probes':probes,'vad_boundary_support':all(p['boundary_supported'] for p in probes),'speech_inside_gap_conflict':conflict})
(D/'vad_audit.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in out));print({'count':len(out),'boundary_supported':sum(x['vad_boundary_support'] for x in out),'speech_inside_gap_conflicts':sum(x['speech_inside_gap_conflict'] for x in out)})
