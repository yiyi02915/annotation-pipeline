"""Experimental audio screen, not validated diarization. Kept separate from trusted gate."""
import json,time
from pathlib import Path
from duplexjev import Decider,Question
D=Path(__file__).resolve().parent;R=D.parents[1]
engine=Decider.from_pretrained(str(R/'models/DuplexJev-4B-Para'),device='cuda:0',attn_implementation='sdpa')
labels=['TWO_OR_FEWER','THREE_OR_MORE']
qs=[Question('count','How many distinct real human speakers are audible in this audio? Count voices, not channels or turns. Background people count; changes of emotion and pitch by the same person do not create a new speaker.',labels,audio='*'),Question('third','Listen for an additional person beyond the main conversational pair. Return THREE_OR_MORE only for a distinct third human voice, not leakage of the pair between channels, background noise, or quoted speech in the same voice.',labels,audio='*')]
out=[]
for c in json.loads((D/'cases.json').read_text()):
 path=R/'runs/html_extra_cases_20261109'/f"{c['number']}_both.wav";t=time.time()
 try:
  ans=engine.decide(str(path),qs,context='Assess acoustic voice count only. No assumptions about the number of recording channels. No reference answer is supplied.',lang='en',mode='packed',n_perm=4,seed=0)
  row={'number':c['number'],'id':c['id'],'answers':ans,'seconds':time.time()-t,'status':'ok','validated_speaker_detector':False}
 except Exception as ex:row={'number':c['number'],'id':c['id'],'status':'error','error':str(ex)}
 out.append(row);(D/'speaker_probe.json').write_text(json.dumps(out,indent=2));print(c['number'],row['status'],flush=True)
print('SPEAKER_PROBE_COMPLETE',flush=True)
