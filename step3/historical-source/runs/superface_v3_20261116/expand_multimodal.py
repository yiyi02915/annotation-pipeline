"""New Seamless conversations, split before labels; preserve original time/word IDs."""
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
DATA=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
sys.path.insert(0,str(ROOT/'step3'))
from input_seamless import load_published
from audio_judge import audio_duration,render_mix
from pipeline import WordTimeline,context,gap_candidates,windows


def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)


def main():
    assert json.loads((ROOT/'checkpoint_03_smoke32.json').read_text())['status']=='passed'
    if (ROOT/'expanded_inputs.json').exists():
        print('REUSE_LOCKED_EXPANSION');return
    start=time.monotonic();excluded=set();sources=[]
    # Include historical event/holdout manifests, not only current training IDs.
    files=[]
    for folder in ['preference_train_v1b_20261115','event_multi_v1_20261109','step3_validation_20261115','step3_semantic_fix_20261115','finegrained_pilot_20261109']:
        d=BASE/folder
        for p in d.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.jsonl'} and p.stat().st_size<100_000_000:
                if any(x in p.parts for x in ['models','audio','__pycache__']):continue
                files.append(p)
    for p in files:
        found=set(re.findall(r'V\d{2}_S\d+_I\d+',p.read_text(errors='replace')))
        if found: excluded.update(found);sources.append(dict(path=str(p),ids=len(found)))
    save('expansion_exclusions.json',dict(conversations=sorted(excluded),sources=sources))
    pool=[]
    for p in sorted((DATA/'training/index_v1').glob('*.jsonl')):
        for line in p.open():
            x=json.loads(line);cid=x['interaction_id']
            if cid not in excluded: pool.append(cid)
    pool=sorted(set(pool),key=lambda s:hashlib.sha256(('new20261116:'+s).encode()).hexdigest())[:6000]
    quotas={'train':1000,'validation':250,'test':500}
    counts=collections.Counter();rows=[];failures=collections.Counter();rng=random.Random(20261116)
    audio_dir=ROOT/'expansion_audio';audio_dir.mkdir(exist_ok=True)
    for src,streams,tracks,error in load_published(DATA,[dict(conversation_id=cid) for cid in pool]):
        cid=src['conversation_id']
        h=int(hashlib.sha256(('split20261116:'+cid).encode()).hexdigest(),16)%100
        split='train' if h<60 else ('validation' if h<75 else 'test')
        if counts[split]>=quotas[split]:continue
        if error: failures[error]+=1;continue
        try:
            duration=audio_duration(tracks)
            if src.get('invalid_fa_words',0): failures['invalid_fa_words']+=1;continue
            if any(w['end']>duration+.001 for ws in streams.values() for w in ws):
                failures['fa_out_of_duration']+=1;continue
            timeline=WordTimeline(streams)
            # One target per task per conversation limits within-dialogue duplication.
            gaps=[g for g in gap_candidates(cid,timeline) if g['target']['gap_seconds']<=20]
            events=[]
            for case,candidates in windows(cid,timeline,duration):
                for c in candidates:
                    if case['target'][0]<=c['start']<case['target'][1]:events.append(c)
            rng.shuffle(gaps);rng.shuffle(events)
            chosen=[]
            if gaps:chosen.append(('gap',gaps[0]['id'],gaps[0]))
            if events:
                c=events[0]
                chosen.append(('overlap',c['candidate_id'],dict(target=c)))
            for task,uid,payload in chosen:
                if counts[split]>=quotas[split] or counts[(split,task)]>=quotas[split]//2:continue
                target=payload['target'];a,b=target.get('absolute_gap',[target.get('start'),target.get('end')])
                lo,hi=max(0,a-12),min(duration,b+12)
                if hi-lo>60:failures['window_over_60s']+=1;continue
                near=timeline.between(lo,hi);payload['context']=context(near)
                audio=audio_dir/(hashlib.sha256(uid.encode()).hexdigest()[:20]+'.wav')
                render_mix(tracks,[lo,hi],audio)
                rows.append(dict(id=uid,conversation_id=cid,split=split,task=task,input=payload,audio=str(audio),audio_ready=True,audio_interval=[lo,hi],target_relative=[a-lo,b-lo],risk_reasons=[],audio_source=dict(kind='synchronized_dual_track_mix',tracks={k:[str(p),ch] for k,(p,ch) in tracks.items()})))
                counts[split]+=1;counts[(split,task)]+=1
                if len(rows)%50==0:
                    save('expansion_progress.json',dict(targets=len(rows),counts={str(k):v for k,v in counts.items()},failures=dict(failures),seconds=time.monotonic()-start))
                    print('EXPANSION',len(rows),flush=True)
        except Exception as exc: failures[f'{type(exc).__name__}:{exc}']+=1
        if all(counts[k]>=v for k,v in quotas.items()):break
    by_split={s:{r['conversation_id'] for r in rows if r['split']==s} for s in quotas}
    assert not (by_split['train']&by_split['validation'] or by_split['train']&by_split['test'] or by_split['test']&by_split['validation'])
    assert not (set.union(*by_split.values())&excluded)
    save('expanded_inputs.json',rows)
    digest=hashlib.sha256((ROOT/'expanded_inputs.json').read_bytes()).hexdigest()
    save('checkpoint_04_expansion.json',dict(status='prepared',targets=len(rows),counts={str(k):v for k,v in counts.items()},failures=dict(failures),inputs_sha256=digest,labels_not_yet_created=True,conversation_disjoint=True,excluded_conversations=len(excluded),gap_filter='0.5 < gap <= 20 seconds; long gaps outside this pilot scope',sampling='one gap and one overlap candidate per conversation when available; not claimed class-balanced',seconds=time.monotonic()-start))
    assert all(counts[k]>=v for k,v in quotas.items()),'insufficient_fresh_inputs; inspect expansion report instead of duplicating'


if __name__=='__main__':main()
