import os,subprocess,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];jobs=[]
def launch(args,name,gpu):
 p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(D/'run.py')]+args,cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4'),stdout=open(D/(name+'.log'),'a'),stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'name':name,'pid':p.pid})
launch(['jev'],'jev',1)
for k,gpu in enumerate([0,2,3]):launch(['qwen','--shard',str(k)],f'qwen_{k}',gpu)
launch(['arbitrate'],'arbitration',0)
(D/'judge_launch.json').write_text(json.dumps(jobs,indent=2));print(jobs)
