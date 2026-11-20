import os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
variant,gpu=sys.argv[1:]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
with (root/(variant+'.log')).open('a') as log:
    proc=subprocess.Popen([sys.executable,str(root/'train_dpo_quality.py'),
         '--source',str(root.parent/'preference_train_v1b_20261115'),
         '--output',str(root/variant),'--variant',variant],
         cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(proc.pid)
