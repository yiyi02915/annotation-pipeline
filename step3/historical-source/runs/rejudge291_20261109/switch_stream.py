import os,json,signal,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]
old=json.loads((D/'launch.json').read_text())['processes']
for r in old:
 if 'audio_check.py' in ' '.join(r['args']):
  try:
   if b'audio_check.py' in Path(f'/proc/{r["pid"]}/cmdline').read_bytes():os.kill(r['pid'],signal.SIGTERM)
  except FileNotFoundError:pass
jobs=[]
for k in range(2):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(k+1),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 f=open(D/f'audio_stream_{k}.log','w');p=subprocess.Popen([str(R/'audio_environment/bin/python'),'-u',str(D/'audio_stream.py'),'--shard',str(k)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'pid':p.pid,'shard':k})
(D/'stream_launch.json').write_text(json.dumps(jobs));print(jobs)
