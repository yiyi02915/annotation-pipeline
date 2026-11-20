import json,re
from collections import Counter
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from nemo.collections.asr.models import SortformerEncLabelModel
R=Path(__file__).resolve().parent
items=json.loads((R/'inputs.json').read_text())
assert len(items)==20
model=SortformerEncLabelModel.restore_from('/data/L202500721/stevezhuang/annotation_pipeline/models/v3/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo',map_location='cuda')
model.eval()
for k,v in dict(chunk_len=340,chunk_right_context=40,fifo_len=40,spkcache_update_period=300).items():setattr(model.sortformer_modules,k,v)
folder=R/'acoustic_recheck_windows';folder.mkdir(exist_ok=True)
shards={}
for p in (R/'priority').rglob('predictions.jsonl'):
    for line in p.open():
        x=json.loads(line);shards[x['chunk_id']]=x
def word(wid):
    k,i=wid.rsplit(':w',1);x=shards[k]
    return [t+x['audio_start_sec'] for t in x['force_align']['timestamps'][int(i)]]
def overlaps(s,e,tracks):
    c=Counter()
    for a,b,p in tracks:c[p]+=max(0,min(b,e)-max(a,s))
    return c
report=[]
for i,x in enumerate(items,1):
    p=x['payload'];t=p['target'];s,e=t['absolute_gap'];a=max(0,int(s)-19)
    f=folder/f'{i:02d}.wav'
    with sf.SoundFile(R/x['source_audio']) as inp:
        sr=inp.samplerate;inp.seek(a*sr);y=inp.read(int(e-a+19)*sr,dtype='float32')
    d=gcd(sr,16000);sf.write(f,resample_poly(y,16000//d,sr//d),16000)
    pred=model.diarize(audio=[str(f)],batch_size=1)
    if isinstance(pred,tuple):pred=pred[0]
    tracks=[(float(a0)+a,float(b0)+a,p0) for a0,b0,p0 in (seg.split() for seg in pred[0])]
    vals=[overlaps(*word(t[k]),tracks) for k in ('left_word_id','right_word_id')]
    ww=[word(t[k]) for k in ('left_word_id','right_word_id')]
    anchors=[v.most_common(1)[0][0] if v and v.most_common(1)[0][1]>=.5*(w[1]-w[0]) else None for v,w in zip(vals,ww)]
    stat='same' if anchors[0] and anchors[0]==anchors[1] else 'different' if all(anchors) else 'one_or_both_unassigned'
    ov=overlaps(s+.1,e-.1,tracks)
    report.append(dict(id=x['id'],index=i,recheck=stat,anchors=anchors,other_activity_seconds=round(sum(v for k,v in ov.items() if k!=anchors[0]),3),focus_activity_seconds=round(ov[anchors[0]],3),speakers_near_gap=len({sp for lo,hi,sp in tracks if hi>=s-3 and lo<=e+3})))
(R/'acoustic_recheck.json').write_text(json.dumps(report,indent=2))
print(Counter(x['recheck'] for x in report),flush=True)
