import os,sys,json,time,gc,argparse,re,difflib
from pathlib import Path
sys.path.insert(0,'/data/L202500293/workspace/git/Qwen3-ASR')
import torch,soundfile as sf
from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--shard',type=int);a=p.parse_args();rows=[json.loads(s) for s in (D/'inputs.jsonl').read_text().splitlines()][a.shard::2]
def load(p):return {e['id']:e for e in map(json.loads,p.read_text().splitlines())} if p.exists() else {}
ap=D/f'asr_{a.shard}.jsonl';done=load(ap)
model=Qwen3ASRModel.from_pretrained('/data/L202500293/obed/models/qwen3-asr/checkpoints/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_new_tokens=512)
with ap.open('a') as f:
 for i,e in enumerate(rows):
  if e['id'] in done:continue
  result={'id':e['id'],'tracks':{}};t=time.time()
  for sp,tr in e['tracks'].items():
   try:
    data,sr=sf.read(tr['audio'],dtype='float32');s=model.transcribe(audio=(data,sr),language='English')[0].text
    norm=lambda x:re.findall(r"[a-z0-9]+",x.lower());ratio=difflib.SequenceMatcher(None,norm(tr['text']),norm(s),autojunk=False).ratio()
    result['tracks'][sp]={'asr_text':s,'transcript_token_similarity':ratio}
   except Exception as ex:result['tracks'][sp]={'error':str(ex)}
  result['seconds']=time.time()-t;f.write(json.dumps(result)+'\n');f.flush();print('ASR',i+1,len(rows),flush=True)
del model;gc.collect();torch.cuda.empty_cache()
fp=D/f'fa_{a.shard}.jsonl';done=load(fp)
model=Qwen3ForcedAligner.from_pretrained('/data/L202500293/obed/pipline_pre/qwenasr-code/checkpoint/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
with fp.open('a') as f:
 for i,e in enumerate(rows):
  if e['id'] in done:continue
  result={'id':e['id'],'tracks':{}};t=time.time()
  for sp,tr in e['tracks'].items():
   try:
    if not tr['text'].strip():result['tracks'][sp]={'words':[]};continue
    data,sr=sf.read(tr['audio'],dtype='float32');aligned=model.align(audio=(data,sr),text=tr['text'],language='English')[0]
    result['tracks'][sp]={'words':[{'text':w.text,'start':round(e['expanded_interval'][0]+w.start_time,3),'end':round(e['expanded_interval'][0]+w.end_time,3)} for w in aligned]}
   except Exception as ex:result['tracks'][sp]={'error':str(ex)}
  result['seconds']=time.time()-t;f.write(json.dumps(result)+'\n');f.flush();print('FA',i+1,len(rows),flush=True)
print('AUDIO_CHECK_COMPLETE',flush=True)
