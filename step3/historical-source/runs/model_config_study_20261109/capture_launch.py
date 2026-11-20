import os,json,time
from pathlib import Path
D=Path(__file__).resolve().parent
boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')));hz=os.sysconf('SC_CLK_TCK');rows=[]
for pid in (1452904,1452905,1452906,1452907):
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'run_jev_study.py' in cmd
 stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split();started=boot+int(stat[19])/hz
 rows.append(dict(pid=pid,process_started_at=started))
(D/'full_launch_metadata.json').write_text(json.dumps(dict(captured_at=time.time(),workers=rows),indent=2))
