import json, random, re, collections, argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from nemo.collections.asr.models import SortformerEncLabelModel
r=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
mode=parser.add_mutually_exclusive_group()
mode.add_argument('--holdout',action='store_true')
mode.add_argument('--blind100',action='store_true')
mode.add_argument('--risk-blind',action='store_true')
args=parser.parse_args()
holdout=args.holdout or args.blind100 or args.risk_blind
prior_files=['inputs.json','holdout_inputs.json','blind100_inputs.json'] if args.risk_blind else ['inputs.json','holdout_inputs.json'] if args.blind100 else ['inputs.json'] if holdout else []
excluded={x['conversation_id'] for name in prior_files for x in json.loads((r/name).read_bytes())}
seed=20261117 if args.risk_blind else 20261115 if args.blind100 else 20261113 if holdout else 42
target_count=25 if args.risk_blind else 100 if args.blind100 else 20
manifest={x['Path']:x['Size'] for x in json.loads((r/'manifest.json').read_text())}
paths={}
for folder in ('dataset','priority'):
    for f in (r/folder).rglob('predictions.jsonl'):
        rel=f.relative_to(r/folder).as_posix()
        if f.stat().st_size==manifest.get(rel):paths[rel]=f
data=collections.defaultdict(list)
for path in paths.values():
    for line in path.open():
        x=json.loads(line)
        if x['alignment_status']=='aligned':
            data[x['conversation_id']].append(x)
ready={}
for rel,size in manifest.items():
    if rel.startswith('audio/') and rel.endswith('.wav'):
        p=r/'dataset'/rel
        c=p.parent.parent.name
        if c not in excluded and c in data and p.is_file() and p.stat().st_size==size:
            if data[c][0]['source_audio_path'].endswith('/'+ '/'.join(rel.split('/')[-3:])):
                ready[c]=p
print('matched recording count',len(ready),flush=True)
if not ready:raise RuntimeError('No exact WAV-to-FA matches yet')
rng=random.Random(seed)
clips=[]
for c in sorted(ready):
    p=ready[c]
    info=sf.info(p)
    print(c,info.channels,info.samplerate,info.duration,flush=True)
    if info.channels!=1:raise RuntimeError('Multichannel data requires verifying FA channel mapping')
    xs=data[c]
    if max(x['audio_end_sec'] for x in xs)>info.duration+1:continue
    for i in rng.sample(range(0,int(info.duration)//48), min(18,int(info.duration)//48)):
        a=i*48
        ts=[x for x in xs if x['audio_end_sec']>a and x['audio_start_sec']<a+48]
        if len(ts)>=5:clips.append((c,a,ts))
rng.shuffle(clips)
clips=clips[:700 if args.risk_blind else 1900 if args.blind100 else 320 if holdout else 150]
print('clips',len(clips),flush=True)
wavdir=r/'diar_windows';wavdir.mkdir(exist_ok=True)
for c,a,_ in clips:
    if (wavdir/f'{c}_{a}.wav').is_file():continue
    with sf.SoundFile(ready[c]) as f:
        f.seek(a*f.samplerate); y=f.read(48*f.samplerate,dtype='float32'); rate=f.samplerate
    div=gcd(rate,16000)
    sf.write(wavdir/f'{c}_{a}.wav',resample_poly(y,16000//div,rate//div),16000)
model=SortformerEncLabelModel.restore_from('/data/L202500721/stevezhuang/annotation_pipeline/models/v3/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo',map_location='cuda')
model.eval()
for k,v in dict(chunk_len=340,chunk_right_context=40,fifo_len=40,spkcache_update_period=300).items():
    setattr(model.sortformer_modules,k,v)
def join(ws):
    return ' '.join(w['text'] for w in ws)
pool=[]
for c,a,xs in clips:
    clip=wavdir/f'{c}_{a}.wav'
    pred=model.diarize(audio=[str(clip)],batch_size=1)
    if isinstance(pred,tuple):pred=pred[0]
    tracks=[]
    for seg in pred[0]:
        lo,hi,sp=seg.split()
        tracks.append((float(lo)+a,float(hi)+a,sp))
    words=[]
    for x in sorted(xs,key=lambda z:z['audio_start_sec']):
        fa=x['force_align'];off=x['audio_start_sec']
        if len(fa['words'])!=len(fa['timestamps']):continue
        for j,(tok,(s,e)) in enumerate(zip(fa['words'],fa['timestamps'])):
            s+=off;e+=off
            if not a<=s<e<=a+48 or not re.search(r'\w',tok):continue
            hits=collections.Counter()
            for u,v,sp in tracks:
                hits[sp]+=max(0,min(v,e)-max(u,s))/(e-s)
            ranked=hits.most_common()
            sp=ranked[0][0] if ranked and ranked[0][1]>=.7 and (len(ranked)<2 or ranked[1][1]<.2) else 'UNKNOWN'
            words.append(dict(text=tok,start=s,end=e,speaker=sp,id=f'{x["chunk_id"]}:w{j}'))
    words.sort(key=lambda w:(w['start'],w['end']))
    for focus in sorted({w['speaker'] for w in words}-{'UNKNOWN'}):
        fw=[(i,w) for i,w in enumerate(words) if w['speaker']==focus]
        for j,((li,l),(ri,h)) in enumerate(zip(fw,fw[1:])):
            lo,hi=l['end'],h['start']
            if hi-lo<=.500001 or hi-lo>18 or lo<a+4 or hi>a+44:continue
            if any(w['speaker']=='UNKNOWN' for w in words[li+1:ri]):continue
            if any(sp==focus and min(v,hi-.1)-max(u,lo+.1)>.16 for u,v,sp in tracks):continue
            window=[w for w in words if w['end']>=lo-12 and w['start']<=hi+12]
            if len({w['speaker'] for w in window}-{'UNKNOWN'})!=2:continue
            runs=[]
            for w in window:
                if not runs or w['speaker']!=runs[-1][-1]['speaker'] or w['start']-runs[-1][-1]['end']>.6:runs.append([])
                runs[-1].append(w)
            segs=[dict(segment_id=f'S{k+1}',speaker=ws[0]['speaker'],start=round(ws[0]['start'],3),end=round(ws[-1]['end'],3),text=join(ws)) for k,ws in enumerate(runs)]
            left=[w for _,w in fw[max(0,j-20):j+1] if w['start']>=lo-12]
            right=[w for _,w in fw[j+1:j+22] if w['start']<=hi+12]
            if len(left)<3 or len(right)<3:continue
            other=next(iter({w['speaker'] for w in window}-{'UNKNOWN',focus}))
            eid=f'{c}:{focus}:{lo:.3f}:{hi:.3f}'
            target=dict(left_word=l['text'],right_word=h['text'],left_word_id=l['id'],right_word_id=h['id'],absolute_gap=[round(lo,3),round(hi,3)],clip_relative_gap=[round(lo-a,3),round(hi-a,3)],gap_seconds=round(hi-lo,3))
            pool.append(dict(id=eid,conversation_id=c,clip=str(clip.relative_to(r)),source_audio=str(ready[c].relative_to(r)),payload=dict(id=eid,focus=focus,other=other,target=target,focus_before_target=join(left),focus_after_target=join(right),other_near_target=[s for s in segs if s['speaker']==other and s['end']>=lo-3 and s['start']<=hi+3],context=segs)))
    print('window',c,a,'words',len(words),'pool',len(pool),flush=True)
rng.shuffle(pool)
picked=[];ids=set();per=collections.Counter()
for e in pool:
    if e['id'] not in ids and per[e['clip']]<(1 if holdout else 2) and per[e['conversation_id']]<(2 if holdout else 5):
        picked.append(e);ids.add(e['id']);per[e['clip']]+=1;per[e['conversation_id']]+=1
        if len(picked)==target_count:break
prefix='risk_blind_' if args.risk_blind else 'blind100_' if args.blind100 else 'holdout_' if holdout else ''
(r/(prefix+'inputs.json')).write_text(json.dumps(picked,ensure_ascii=False,indent=2))
(r/(prefix+'extraction.json')).write_text(json.dumps(dict(matched_recordings=len(ready),clips=len(clips),pool=len(pool),sample_size=len(picked),seed=seed,selected_recordings=len({x['conversation_id'] for x in picked}),excluded_recordings=sorted(excluded)),indent=2))
print('DONE',len(picked),flush=True)
