import os,signal,json,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];jobs=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes()
  if b'rejudge_qwen.py' in cmd and b'--shard' not in cmd:os.kill(int(proc.name),signal.SIGTERM)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
source=(D.parent/'finegrained_pilot_20261109/start_qwen.sh').read_text()
for k in range(4):
 if k:
  script=D/f'start_qwen_{k}.sh';script.write_text(source.replace('CUDA_VISIBLE_DEVICES=0',f'CUDA_VISIBLE_DEVICES={k}').replace('--port 18352',f'--port {18352+k}'))
  f=open(D/f'qwen_server_{k}.log','w');p=subprocess.Popen(['bash',str(script)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'pid':p.pid,'role':'server','shard':k})
 f=open(D/f'qwen_shard{k}.log','w');p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(D/'rejudge_qwen.py'),'--shard',str(k)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'pid':p.pid,'role':'client','shard':k})
(D/'parallel_launch.json').write_text(json.dumps({'time':time.time(),'jobs':jobs}));print(jobs)
