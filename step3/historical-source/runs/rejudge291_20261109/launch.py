import subprocess,os,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];procs=[]
def launch(args,log,env):
 f=open(D/log,'w');p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,**env),start_new_session=True);procs.append({'pid':p.pid,'args':args,'log':log});return p
base={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
for shard in range(2):launch([str(R/'audio_environment/bin/python'),'-u',str(D/'audio_check.py'),'--shard',str(shard)],f'audio_{shard}.log',dict(base,CUDA_VISIBLE_DEVICES=str(shard+1)))
launch(['bash',str(D.parent/'finegrained_pilot_20261109/start_qwen.sh')],'qwen_server.log',base)
(D/'launch.json').write_text(json.dumps({'time':time.time(),'processes':procs},indent=2));print(json.dumps(procs))
