import time,os,signal,subprocess,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];t=time.time()
while True:
 if 'QWEN_COMPLETE' in (D/'qwen_shard0_v2.log').read_text():break
 if time.time()-t>10800:raise TimeoutError('Qwen shard0 not complete')
 time.sleep(10)
meta=json.loads((D/'launch.json').read_text())
for job in meta['processes']:
 if job['log']=='qwen_server.log':
  pid=job['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  assert b'18352' in cmd and b'vllm' in cmd;os.killpg(pid,signal.SIGTERM)
time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
f=open(D/'jev_emptyfix.log','w');p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(D/'rejudge_jev.py'),'--empty-fix'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(D/'jev_fix_launch.json').write_text(json.dumps({'pid':p.pid}));print('JEV_FIX_STARTED',p.pid,flush=True)
