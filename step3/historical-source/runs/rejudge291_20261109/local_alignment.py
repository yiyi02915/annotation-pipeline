import sys,time,json
from pathlib import Path
sys.path.insert(0,'/data/L202500293/workspace/git/Qwen3-ASR')
import torch,soundfile as sf
from qwen_asr import Qwen3ForcedAligner
from evidence import *
model=Qwen3ForcedAligner.from_pretrained('/data/L202500293/obed/pipline_pre/qwenasr-code/checkpoint/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
out=D/'local_fa.jsonl';done={e['id'] for e in read(out)}
with out.open('a') as f:
 for e in read(D/'inputs.jsonl'):
  if e['id'] in done:continue
  a=max(e['expanded_interval'][0],e['anchor']-3);b=min(e['expanded_interval'][1],e['right_word']['end']+3);tr=e['tracks'][e['focus']];ws=[w for w in tr['words'] if w['start']>=a and w['end']<=b];r={'id':e['id'],'window':[a,b]}
  try:
   sr=sf.info(tr['audio']).samplerate;data,sr=sf.read(tr['audio'],start=max(0,round((a-e['expanded_interval'][0])*sr)),stop=round((b-e['expanded_interval'][0])*sr),dtype='float32')
   aligned=model.align(audio=(data,sr),text=' '.join(w['text'] for w in ws),language='English')[0];new=[{'text':w.text,'start':round(a+w.start_time,3),'end':round(a+w.end_time,3)} for w in aligned];mp,cov=aligned_map(ws,new);ix={w['id']:i for i,w in enumerate(ws)};left=mp.get(ix.get(e['left_word']['id']));right=mp.get(ix.get(e['right_word']['id']));r.update(words=new,mapped_fraction=cov,left=left,right=right,gap=round(right['start']-left['end'],3) if left and right else None)
  except Exception as ex:r['error']=str(ex)
  f.write(json.dumps(r)+'\n');f.flush();done.add(e['id']);print('LOCAL_FA',len(done),flush=True)
print('LOCAL_FA_COMPLETE',flush=True)
