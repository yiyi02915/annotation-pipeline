"""Finish the real resume test and start the detached queue even if SSH closes."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from durable_training import atomic,alive,lock,read,now
root=Path(__file__).resolve().parent
def main():
    guard=lock(root/'bootstrap.lock')
    atomic(root/'bootstrap_status.json',dict(status='checking_resume',pid=os.getpid(),time=now()))
    probe=root/'jobs/resume_probe'
    for goal in [1,2]:
        while (probe/'status.json').exists() and alive(read(probe/'status.json').get('pid',0),'resume_probe'):time.sleep(3)
        latest=read(probe/'latest.json') if (probe/'latest.json').exists() else {'step':-1}
        if latest['step']<goal:
            with (root/f'probe_bootstrap_{goal}.log').open('a') as log:
                proc=subprocess.Popen([sys.executable,str(root/'durable_training.py'),'train','--job','resume_probe','--pause-after',str(goal)],
                    cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            code=proc.wait();assert code==75, f'Probe stage {goal} failed: {code}'
        assert read(probe/'latest.json')['step']>=goal
    import torch
    latest=read(probe/'latest.json')
    state=torch.load(probe/latest['path']/'state.pt',map_location='cpu',weights_only=False)
    steps={float(v['step']) for v in state['optimizer']['state'].values() if 'step' in v}
    assert steps=={2.},steps
    events=[json.loads(l) for l in (probe/'events.jsonl').read_text().splitlines()]
    assert any(r['event']=='resumed' and r['step']==1 for r in events)
    atomic(root/'resume_test.json',dict(status='passed',checkpoint_step=state['cursor']['step'],optimizer_steps=list(steps),
        evidence='A new process restored checkpoint 1 and completed update 2; model, optimizer and RNG states persisted.',time=now()))
    atomic(root/'plan.json',read(root/'plan.final.json'))
    with (root/'supervisor.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,str(root/'durable_training.py'),'supervise'],cwd=root,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    atomic(root/'bootstrap_status.json',dict(status='supervisor_started',pid=os.getpid(),supervisor_pid=proc.pid,time=now()))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic(root/'bootstrap_status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),time=now()))
        traceback.print_exc();sys.exit(1)
