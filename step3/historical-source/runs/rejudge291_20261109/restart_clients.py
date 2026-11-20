import os,signal,subprocess,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];meta=json.loads((D/'parallel_launch.json').read_text())
for job in meta['jobs']:
 if job['role']!='client':continue
 try:
  if b'rejudge_qwen.py' in Path(f'/proc/{job["pid"]}/cmdline').read_bytes():os.kill(job['pid'],signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError):pass
for k in range(4):
 f=open(D/f'qwen_shard{k}_v2.log','w');p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(D/'rejudge_qwen.py'),'--shard',str(k)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);meta['jobs'].append({'pid':p.pid,'role':'client_v2','shard':k})
(D/'parallel_launch_v2.json').write_text(json.dumps(meta));print('CLIENTS_RESTARTED')
