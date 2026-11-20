import os,signal,subprocess,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];stopped=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0');cwd=(p/'cwd').resolve()
  if cwd==D and b'clarify_facts.py' in args and args[0].endswith(b'/python'):
   os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
 except (OSError,ProcessLookupError):pass
time.sleep(1)
p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u','clarify_facts.py'],cwd=D,stdout=open(D/'clarify_facts.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
(D/'clarification_restart.json').write_text(json.dumps({'stopped':stopped,'new_pid':p.pid}));print(stopped,p.pid)
