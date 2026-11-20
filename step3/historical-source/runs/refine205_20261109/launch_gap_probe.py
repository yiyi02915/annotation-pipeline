import os,subprocess,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]
p=subprocess.Popen([str(R/'audio_environment/bin/python'),'-u',str(D/'gap_audio_probe.py')],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4'),stdout=open(D/'gap_audio_probe.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
(D/'gap_probe_launch.json').write_text(json.dumps({'pid':p.pid}));print(p.pid)
