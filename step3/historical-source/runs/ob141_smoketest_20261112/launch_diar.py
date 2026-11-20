import subprocess
from pathlib import Path
r=Path('/data/L202500493/duplex_postprocess/runs/ob141_smoketest_20261112')
p=Path('/data/L202500293/obed/envs/qwenasr/bin/python')
if not p.exists(): raise FileNotFoundError(p)
import os
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='3',HF_HUB_OFFLINE='1')
with (r/'diarize.log').open('wb') as log:
    job=subprocess.Popen([str(p),'-u',str(r/'diarize_sample.py')],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(job.pid)
