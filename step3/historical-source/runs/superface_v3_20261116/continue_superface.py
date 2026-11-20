"""Gated unattended continuation; no remote publication or deployment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_superface_stages import launch,wait,record
ROOT=Path(__file__).resolve().parent


def main():
    assert (ROOT/'checkpoint_02_audit.json').exists(),'audit must complete first'
    wait('alignment21',launch('alignment21',['check_target_alignment.py']))
    if not (ROOT/'checkpoint_03_smoke32.json').exists():
        wait('smoke32',launch('smoke32',['multimodal_smoke.py'],2))
    assert json.loads((ROOT/'checkpoint_03_smoke32.json').read_text())['status']=='passed'
    wait('expansion',launch('expansion',['expand_multimodal.py']))
    if not (ROOT/'reference_lock.json').exists():
        t=launch('expanded_text',['label_expansion.py','text'])
        a=launch('expanded_audio',['label_expansion.py','audio'],3)
        wait('expanded_text',t);wait('expanded_audio',a)
        wait('reference_freeze',launch('reference_freeze',['label_expansion.py','finalize']))
    a=launch('abc_A',['train_abc_multimodal.py','A'],2)
    b=launch('abc_B',['train_abc_multimodal.py','B'],3)
    wait('abc_A',a);wait('abc_B',b)
    wait('abc_C',launch('abc_C',['train_abc_multimodal.py','C'],2))
    results={arm:json.loads((ROOT/'abc'/arm/'test_summary.json').read_text()) for arm in ['A','B','C']}
    pairs={arm:{r['id']:r for r in json.loads((ROOT/'abc'/arm/'predictions.json').read_text())} for arm in results}
    comparisons={}
    for left,right in [('A','B'),('B','C')]:
        assert pairs[left].keys()==pairs[right].keys()
        corrected=[];regressed=[]
        for key,r in pairs[left].items():
            s=pairs[right][key]
            if r['predicted']!=r['reference'] and s['predicted']==s['reference']:corrected.append(key)
            if r['predicted']==r['reference'] and s['predicted']!=s['reference']:regressed.append(key)
        comparisons[left+'->'+right]=dict(corrected=corrected,regressed=regressed,net=len(corrected)-len(regressed))
    (ROOT/'checkpoint_06_abc.json').write_text(json.dumps(dict(results=results,comparisons=comparisons,automatic_deployment=False,limitations=['automatic references, not human gold','candidate classification pilot, not complete recording event recall','natural-distribution false-positive and paired-interval acceptance still required']),indent=2))
    record(dict(stage='abc_experiment',status='complete',artifact='checkpoint_06_abc.json',uploaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--detach',action='store_true');args=p.parse_args()
    if args.detach:
        pidfile=ROOT/'continuation.pid'
        if pidfile.exists():
            pid=int(pidfile.read_text())
            try:
                os.kill(pid,0)
                raise RuntimeError(f'continuation already running: {pid}')
            except ProcessLookupError:pass
        with (ROOT/'continuation.log').open('a') as log:
            p=subprocess.Popen([sys.executable,__file__],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pidfile.write_text(str(p.pid));print(p.pid)
    else:
        try:main()
        except Exception as exc:
            record(dict(stage='continuation',status='failed',error=f'{type(exc).__name__}: {exc}'));raise
