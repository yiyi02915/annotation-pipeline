import os,json,signal,time,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];stopped=[]
assert all('QWEN_COMPLETE' in (D/f'qwen_{k}.log').read_text() for k in range(3))
assert 'ARBITRATION_COMPLETE' not in (D/'arbitration.log').read_text()
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if str(D/'scale_arbitration.py').encode() in args or (str(D/'run.py').encode() in args and b'arbitrate' in args):os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
 except (OSError,ProcessLookupError):pass
time.sleep(2)
assert all(not (Path('/proc')/str(pid)).exists() or (Path('/proc')/str(pid)/'stat').read_text().split()[2]=='Z' for pid in stopped)
p=subprocess.Popen([str(R/'runtime/jev/bin/python'),'-u',str(D/'run.py'),'arbitrate'],cwd=D,stdout=open(D/'arbitration.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
result={'stopped':stopped,'new_pid':p.pid,'parallel_requests':48,'saved_records_retained':len((D/'arbitration.jsonl').read_text().splitlines()),'note':'Unfinished requests may be retried; completed records retained.'};(D/'arbitration_switch.json').write_text(json.dumps(result,indent=2));print(result)
