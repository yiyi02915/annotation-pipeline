import os,json,time,signal,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]
if '--launch' in sys.argv:
 p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(Path(__file__).resolve())],cwd=D,stdout=open(D/'scale_arbitration.log','a'),stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)
else:
 pid=next(x['pid'] for x in json.loads((D/'judge_launch.json').read_text()) if x['name']=='arbitration');start=time.time()
 while time.time()-start<1200:
  if 'ARBITRATION_COMPLETE' in (D/'arbitration.log').read_text():print('Already complete');break
  try:
   selected={x['id'] for x in json.loads((D/'arbitration_selection.json').read_text())['selected']};done={json.loads(s)['id'] for s in (D/'arbitration.jsonl').read_text().splitlines() if s.strip()}
   if selected and selected<=done:
    args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0');assert b'arbitrate' in args and str(D/'run.py').encode() in args
    os.kill(pid,signal.SIGTERM);time.sleep(1);p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(D/'run.py'),'arbitrate'],cwd=D,stdout=open(D/'arbitration.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
    result={'old_pid':pid,'new_pid':p.pid,'parallel_requests':48,'checkpointed_ids':len(done)};(D/'arbitration_scale.json').write_text(json.dumps(result,indent=2));print(result,flush=True);break
  except (OSError,json.JSONDecodeError):pass
  time.sleep(1)
