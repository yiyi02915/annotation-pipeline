import sys,json,time,re
from pathlib import Path
import torch,soundfile as sf
D=Path(__file__).resolve().parent
sys.path.insert(0,'/data/L202500293/workspace/git/Qwen3-ASR')
from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
selected={r['id'] for r in read(D/'vad_audit.jsonl') if r['speech_inside_gap_conflict']}
asr=Qwen3ASRModel.from_pretrained('/data/L202500293/obed/models/qwen3-asr/checkpoints/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_new_tokens=256)
fa=Qwen3ForcedAligner.from_pretrained('/data/L202500293/obed/pipline_pre/qwenasr-code/checkpoint/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
out=D/'gap_audio_probe.jsonl';done={r['id'] for r in read(out)} if out.exists() else set()
with out.open('a') as f:
 for e in read(D/'events.jsonl'):
  if e['id'] not in selected or e['id'] in done:continue
  p=e['payload'];l,r=p['target']['clip_relative_gap'];path=e['audio_tracks'][p['focus']];info=sf.info(path);a=max(0,l-1.5);b=min(info.duration,r+1.5);x,sr=sf.read(path,start=int(a*info.samplerate),stop=int(b*info.samplerate),dtype='float32');t=time.time()
  try:
   txt=asr.transcribe(audio=(x,sr),language='English')[0].text;aw=[]
   if txt.strip():
    aligned=fa.align(audio=(x,sr),text=txt,language='English')[0];aw=[{'text':w.text,'start':round(a+w.start_time,3),'end':round(a+w.end_time,3)} for w in aligned]
   inside=[w for w in aw if w['end']-w['start']>=.04 and l+.1<(w['start']+w['end'])/2<r-.1]
   z={'id':e['id'],'status':'ok','local_asr':txt,'asr_alignment':aw,'words_inside_marked_gap':inside,'note':'Diagnostic only: ASR/FA is not an independent gold label and may miss fillers or hear channel leakage.'}
  except Exception as ex:z={'id':e['id'],'status':'error','error':str(ex)}
  z['seconds']=time.time()-t;f.write(json.dumps(z)+'\n');f.flush();done.add(e['id']);print('GAP_PROBE',len(done),z['status'],flush=True)
print('GAP_PROBE_COMPLETE',flush=True)
