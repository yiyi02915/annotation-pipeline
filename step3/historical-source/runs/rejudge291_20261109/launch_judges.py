import subprocess,os,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]
jobs=[]
for name,gpu in [('qwen',0),('jev',3)]:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 f=open(D/f'{name}_rejudge.log','w');p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(D/f'rejudge_{name}.py')],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);jobs.append({'pid':p.pid,'name':name})
(D/'judge_launch.json').write_text(json.dumps(jobs));print(jobs)
