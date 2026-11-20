"""Probe all disconnected speech locations proposed by the initial alignments."""
import sys,json,gc
from pathlib import Path
sys.path.insert(0,'/data/L202500293/workspace/git/Qwen3-ASR')
import torch,soundfile as sf
from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
root=Path(__file__).resolve().parent
targets=json.loads((root/'alignment_evidence.json').read_text())
probes=[];audio_cache={}
for t in targets:
    if t['utterance'].get('transcript_status'):continue
    old=t['utterance'];low=max(old['start']-.3,t['previous']['end']+.02 if t['previous'] else 0)
    high=min(old['end']+.3,t['next']['start']-.02 if t['next'] else old['end']+2)
    spans=sorted((max(low,w['start']),min(high,w['end'])) for a in t['alignments'] for w in a.get('words',[]) if w['end']>low and w['start']<high and 0<=w['end']-w['start']<=2)
    groups=[]
    for a,b in spans:
        if groups and a-groups[-1][1]<=.8:groups[-1][1]=max(groups[-1][1],b)
        else:groups.append([a,b])
    for i,(a,b) in enumerate(groups):
        prev_end=t['previous']['end']+.01 if t['previous'] else 0
        next_start=t['next']['start']-.01 if t['next'] else high+2
        windows=[[round(max(prev_end,a-pad),3),round(min(next_start,b+pad),3)] for pad in [.45,.85]]
        probes.append(dict(probe_id=old['utterance_id']+'::p'+str(i),parent_id=old['utterance_id'],audio=t['audio'],hypothesized_span=[a,b],windows=windows,asr=[],alignments=[]))
def save(): (root/'focused_evidence.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2),encoding='utf8')
asr=Qwen3ASRModel.from_pretrained('/data/L202500293/obed/models/qwen3-asr/checkpoints/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_new_tokens=128)
for p in probes:
    if p['audio'] not in audio_cache:
        data,sr=sf.read(p['audio'],dtype='float32');audio_cache[p['audio']]=(data.mean(axis=1) if data.ndim>1 else data,sr)
    data,sr=audio_cache[p['audio']]
    for a,b in p['windows']:
        r=asr.transcribe(audio=(data[int(a*sr):int(b*sr)],sr),language='English')
        p['asr'].append(dict(window=[a,b],text=r[0].text));print(p['probe_id'],a,b,r[0].text,flush=True);save()
del asr;gc.collect();torch.cuda.empty_cache()
fa=Qwen3ForcedAligner.from_pretrained('/data/L202500293/obed/pipline_pre/qwenasr-code/checkpoint/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
for p in probes:
    data,sr=audio_cache[p['audio']]
    for text in dict.fromkeys(x['text'] for x in p['asr'] if x['text'].strip()):
        for a,b in p['windows']:
            try:
                r=fa.align(audio=(data[int(a*sr):int(b*sr)],sr),text=text,language='English')
                p['alignments'].append(dict(text=text,window=[a,b],words=[dict(word=w.text,start=round(a+w.start_time,3),end=round(a+w.end_time,3)) for w in r[0]]))
            except Exception as exc:p['alignments'].append(dict(text=text,window=[a,b],error=repr(exc)))
            save()
print('FOCUSED_EVIDENCE_COMPLETE',flush=True)
