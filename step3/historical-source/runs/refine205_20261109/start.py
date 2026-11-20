import subprocess,os,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];jobs=[]
def launch(args,name,env=None):
 f=open(D/(name+'.log'),'w');p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**(env or {})));jobs.append({'pid':p.pid,'name':name});return p
base={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
launch([str(R/'audio_environment/bin/python'),'-u',str(D/'repair_alignment.py')],'alignment',dict(base,CUDA_VISIBLE_DEVICES='0'))
source=(D.parent/'finegrained_pilot_20261109/start_qwen.sh').read_text()
for k in [2,3]:
 p=D/f'start_qwen_{k}.sh';p.write_text(source.replace('CUDA_VISIBLE_DEVICES=0',f'CUDA_VISIBLE_DEVICES={k}').replace('--port 18352',f'--port {18450+k}').replace('--max-num-seqs 8','--max-num-seqs 16'))
 launch(['bash',str(p)],f'qwen_server_{k}',base)
(D/'launch.json').write_text(json.dumps({'time':time.time(),'jobs':jobs},indent=2));print(jobs)
