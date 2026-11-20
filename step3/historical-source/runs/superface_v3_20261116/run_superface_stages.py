"""Detached, checkpointed stage runner. Failed gates stop dependent stages."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def record(value):
    value['time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/'checkpoints.jsonl').open('a') as f:
        f.write(json.dumps(value)+'\n'); f.flush(); os.fsync(f.fileno())


def launch(name, argv, gpu=None):
    env=dict(os.environ, PYTHONUNBUFFERED='1')
    if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (ROOT/(name+'.log')).open('a') as log:
        p=subprocess.Popen([sys.executable,*argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record(dict(stage=name,status='running',pid=p.pid,argv=argv))
    return p


def wait(name,p):
    code=p.wait();record(dict(stage=name,status='complete' if code==0 else 'failed',exit_code=code))
    if code: raise RuntimeError(f'{name} exit {code}')


def main():
    t=launch('audit_text',['audit_multimodal.py','text'])
    a=launch('audit_audio',['audit_multimodal.py','audio'],3)
    wait('audit_text',t);wait('audit_audio',a)
    wait('audit_finalize',launch('audit_finalize',['audit_multimodal.py','finalize']))
    record(dict(stage='checkpoint_21',status='complete',artifact='checkpoint_02_audit.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    if args.detach:
        with (ROOT/'supervisor.log').open('a') as f:
            p=subprocess.Popen([sys.executable,__file__],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'supervisor.pid').write_text(str(p.pid));print(p.pid)
    else:
        try: main()
        except Exception as exc:
            record(dict(stage='supervisor',status='failed',error=str(exc)));raise
