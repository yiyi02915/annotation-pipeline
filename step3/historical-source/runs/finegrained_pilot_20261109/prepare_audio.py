"""Decode real focus-channel audio, validate basic signal quality, preserve 1500 per task."""
import collections,hashlib,json,math,time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parent
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def main():
    began=time.time();dest=ROOT/'clips';dest.mkdir(exist_ok=True)
    initial=read(ROOT/'candidates.jsonl');reserve=read(ROOT/'candidate_reserve.jsonl')
    counts=collections.Counter();accepted=[];excluded=[]
    for e in initial+reserve:
        if counts[e['task']]>=1500:continue
        try:
            path=e['audio_paths'][e['focus']];a,b=e['audio_interval']
            with sf.SoundFile(path) as f:
                sr=f.samplerate
                assert f.channels==1 and f.frames/sr>=b-.02
                f.seek(round(a*sr));audio=f.read(round((b-a)*sr),dtype='float32')
            assert len(audio)>sr*.1 and np.isfinite(audio).all()
            rms=float(np.sqrt(np.mean(audio**2)));clipped=float(np.mean(np.abs(audio)>=.999))
            assert rms>1e-4,'silent_focus_track'
            assert clipped<.02,'clipped_audio'
            g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g).astype(np.float32)
            name=hashlib.sha256(e['id'].encode()).hexdigest()[:24]+'.wav'
            clip=dest/name;sf.write(clip,audio,16000,subtype='PCM_16')
            e.update(audio_clip=str(clip),audio_clip_seconds=len(audio)/16000,
                     audio_clip_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),audio_rms=rms,audio_clipped_fraction=clipped)
            accepted.append(e);counts[e['task']]+=1
        except Exception as ex:
            excluded.append({'id':e['id'],'task':e['task'],'error':str(ex)})
        if (len(accepted)+len(excluded))%100==0:
            progress={'accepted':dict(counts),'excluded':len(excluded),'elapsed_seconds':time.time()-began}
            (ROOT/'audio_progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        if all(counts[t]==1500 for t in ('overlap','pause')):break
    assert counts=={'overlap':1500,'pause':1500},counts
    accepted.sort(key=lambda e:(e['task'],e['conversation_id'],e['anchor']))
    with (ROOT/'benchmark_inputs.jsonl').open('w') as f:
        for e in accepted:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    report={'count':len(accepted),'by_task':dict(counts),'excluded_count':len(excluded),'excluded':excluded,
            'unique_conversations':len({e['conversation_id'] for e in accepted}),
            'unique_sessions':len({e['session_id'] for e in accepted}),
            'focus_clip_seconds':sum(e['audio_clip_seconds'] for e in accepted),
            'by_kind':dict(collections.Counter(e['conversation_kind'] for e in accepted)),
            'sha256':hashlib.sha256((ROOT/'benchmark_inputs.jsonl').read_bytes()).hexdigest(),
            'elapsed_seconds':time.time()-began,'audio_validation':'decodable, finite, non-silent and clipping fraction < 2%; not a semantic alignment verification'}
    (ROOT/'audio_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
