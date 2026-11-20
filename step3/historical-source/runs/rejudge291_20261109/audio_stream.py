import sys,json,time,argparse,re,difflib
from pathlib import Path
sys.path.insert(0,'/data/L202500293/workspace/git/Qwen3-ASR')
import torch,soundfile as sf
from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--shard',type=int);a=p.parse_args();rows=[json.loads(s) for s in (D/'inputs.jsonl').read_text().splitlines()][a.shard::2]
def load(p):return {e['id']:e for e in map(json.loads,p.read_text().splitlines())} if p.exists() else {}
ap=D/f'asr_{a.shard}.jsonl';fp=D/f'fa_{a.shard}.jsonl';adone=load(ap);fdone=load(fp)
asr=Qwen3ASRModel.from_pretrained('/data/L202500293/obed/models/qwen3-asr/checkpoints/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_new_tokens=512)
fa=Qwen3ForcedAligner.from_pretrained('/data/L202500293/obed/pipline_pre/qwenasr-code/checkpoint/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
with ap.open('a') as af,fp.open('a') as ff:
 for i,e in enumerate(rows):
  cache={sp:sf.read(tr['audio'],dtype='float32') for sp,tr in e['tracks'].items()}
  if e['id'] not in adone:
   r={'id':e['id'],'tracks':{}};t=time.time()
   for sp,tr in e['tracks'].items():
    try:
     data,sr=cache[sp];s=asr.transcribe(audio=(data,sr),language='English')[0].text
     norm=lambda x:re.findall(r'[a-z0-9]+',x.lower());ratio=difflib.SequenceMatcher(None,norm(tr['text']),norm(s),autojunk=False).ratio();r['tracks'][sp]={'asr_text':s,'transcript_token_similarity':ratio}
    except Exception as ex:r['tracks'][sp]={'error':str(ex)}
   r['seconds']=time.time()-t;af.write(json.dumps(r)+'\n');af.flush()
  if e['id'] not in fdone:
   r={'id':e['id'],'tracks':{}};t=time.time()
   for sp,tr in e['tracks'].items():
    try:
     if not tr['text'].strip():r['tracks'][sp]={'words':[]};continue
     data,sr=cache[sp];aligned=fa.align(audio=(data,sr),text=tr['text'],language='English')[0]
     r['tracks'][sp]={'words':[{'text':w.text,'start':round(e['expanded_interval'][0]+w.start_time,3),'end':round(e['expanded_interval'][0]+w.end_time,3)} for w in aligned]}
    except Exception as ex:r['tracks'][sp]={'error':str(ex)}
   r['seconds']=time.time()-t;ff.write(json.dumps(r)+'\n');ff.flush()
  print('AUDIO_COMPLETE',i+1,len(rows),flush=True)
print('AUDIO_CHECK_COMPLETE',flush=True)
