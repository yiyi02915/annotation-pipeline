import os,time
from pathlib import Path
D=Path(__file__).resolve().parent;t=time.time()
while True:
 log=(D/'download.log').read_text(errors='replace')
 if 'DOWNLOAD_COMPLETE DuplexJev-32B-Para' in log:break
 if 'Traceback (most recent call last)' in log:raise RuntimeError('Download failed; see download.log')
 if time.time()-t>5400:raise TimeoutError('Weights not ready after90min')
 time.sleep(10)
os.environ['CUDA_VISIBLE_DEVICES']='2';os.environ['HF_HUB_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='4'
py='/data/L202500493/duplex_postprocess/runtime/jev/bin/python'
os.execv(py,[py,'-u',str(D/'run_jev_study.py'),'--size','32B','--batch-size','1'])
