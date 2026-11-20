import os,subprocess,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];jobs=[]
source=(D.parent/'finegrained_pilot_20261109/start_qwen.sh').read_text()
for shard,gpu in enumerate([0,2,3]):
 port=18550+shard;p=D/f'start_qwen_{shard}.sh';p.write_text(source.replace('CUDA_VISIBLE_DEVICES=0',f'CUDA_VISIBLE_DEVICES={gpu}').replace('--port 18352',f'--port {port}').replace('--max-num-seqs 8','--max-num-seqs 16'))
 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 job=subprocess.Popen(['bash',str(p)],stdout=open(D/f'server_{shard}.log','w'),stderr=subprocess.STDOUT,start_new_session=True,env=env);jobs.append({'name':f'qwen_server_{shard}','pid':job.pid,'gpu':gpu,'port':port})
(D/'launch.json').write_text(json.dumps({'time':time.time(),'jobs':jobs},indent=2));print(jobs)
