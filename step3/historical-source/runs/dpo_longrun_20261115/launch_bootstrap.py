import subprocess,sys
from pathlib import Path
from durable_training import alive,read
root=Path(__file__).resolve().parent
state=root/'bootstrap_status.json'
master=root/'supervisor.json'
if master.exists() and alive(read(master).get('pid',0),'supervise'):
    print('SUPERVISOR_ALREADY_RUNNING',read(master)['pid'])
elif state.exists() and alive(read(state).get('pid',0),'bootstrap.py'):
    print('ALREADY_RUNNING',read(state)['pid'])
else:
    with (root/'bootstrap.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,str(root/'bootstrap.py')],cwd=root,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print('DETACHED_BOOTSTRAP_PID',proc.pid)
