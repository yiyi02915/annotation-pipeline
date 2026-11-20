import json,subprocess,time,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];python=str(R/'runtime/jev/bin/python')
if '--launch' in sys.argv:
 p=subprocess.Popen([python,'-u',str(Path(__file__).resolve())],cwd=D,stdout=open(D/'completion.log','a'),stderr=subprocess.STDOUT,start_new_session=True);(D/'completion_launch.json').write_text(json.dumps({'pid':p.pid}));print(p.pid)
else:
 while not (D/'arbitration.log').exists() or 'ARBITRATION_COMPLETE' not in (D/'arbitration.log').read_text():time.sleep(5)
 try:
  subprocess.run([python,str(D/'finish.py')],check=True,cwd=D);subprocess.run([python,str(D/'cleanup.py')],check=True,cwd=D)
  (D/'completion_status.json').write_text(json.dumps({'status':'complete','time':time.time()}));print('RUN_COMPLETE',flush=True)
 except Exception as ex:
  (D/'completion_status.json').write_text(json.dumps({'status':'error','error':str(ex),'time':time.time()}));raise
