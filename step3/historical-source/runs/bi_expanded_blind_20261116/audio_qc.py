import json,numpy as np,soundfile as sf
from scipy.signal import resample_poly,correlate
from math import gcd
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=[]
for r in json.loads((ROOT/'inputs.json').read_text()):
 signals=[];lo,hi=r['case']['interval'];info=[]
 for sp,(path,ch) in r['tracks'].items():
  with sf.SoundFile(path) as f:
   rate=f.samplerate;f.seek(round(lo*rate));wave=f.read(round((hi-lo)*rate),dtype='float32',always_2d=True)[:,ch]
  div=gcd(rate,8000);wave=resample_poly(wave,8000//div,rate//div);signals.append(wave)
  info.append(dict(speaker=sp,rms=float(np.sqrt(np.mean(wave**2))),peak=float(np.max(np.abs(wave)))))
 a,b=signals;n=min(len(a),len(b));a=a[:n]-np.mean(a[:n]);b=b[:n]-np.mean(b[:n]);den=np.linalg.norm(a)*np.linalg.norm(b)
 corr=correlate(a,b,mode='full',method='fft')/max(den,1e-12);mid=n-1;bound=4000;v=corr[mid-bound:mid+bound+1];offset=int(np.argmax(np.abs(v)))-bound
 out.append(dict(number=r['number'],tracks=info,zero_lag_correlation=float(corr[mid]),peak_absolute_correlation=float(np.max(np.abs(v))),lag_seconds=offset/8000,
  quality_flag='near_duplicate_audio' if np.max(np.abs(v))>.95 else None))
(ROOT/'audio_qc.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
