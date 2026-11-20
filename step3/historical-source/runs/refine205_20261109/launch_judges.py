import os,json,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];jobs=[]
def start(cmd,name,gpu):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
 p=subprocess.Popen(cmd,env=env,cwd=D,stdout=open(D/(name+'.log'),'a'),stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'name':name,'pid':p.pid})
start([str(R/'runtime/jev/bin/python'),'-u',str(D/'refine.py'),'jev'],'jev',1)
while 'ALIGNMENT_COMPLETE' not in (D/'alignment.log').read_text(errors='replace'):time.sleep(5)
subprocess.run([str(R/'runtime/jev/bin/python'),str(D/'refine.py'),'prepare'],check=True)
for shard in range(2):start([str(R/'runtime/jev/bin/python'),'-u',str(D/'refine.py'),'qwen','--shard',str(shard)],f'qwen_{shard}',2+shard)
(D/'judge_launch.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs),flush=True)
