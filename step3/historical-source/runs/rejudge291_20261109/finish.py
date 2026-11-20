import json,zipfile,hashlib,os,signal
from pathlib import Path
D=Path(__file__).resolve().parent
assert 'OUTPUT_REPAIR_COMPLETE' in (D/'output_repair.log').read_text()
assert 'JEV_COMPLETE' in (D/'jev_emptyfix.log').read_text()
for script in ['summarize.py','final_report.py','audit.py']:
 import subprocess,sys
 subprocess.run([sys.executable,str(D/script)],check=True)
audit=json.loads((D/'output_audit.json').read_text());assert audit['schema_and_partition_checks']=='passed'
stopped=[]
jobs=json.loads((D/'parallel_launch_v2.json').read_text())['jobs']
for job in jobs:
 if job['role']!='server':continue
 pid=job['pid']
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'vllm' in cmd and b'qwen38-27b' in cmd;os.killpg(pid,signal.SIGTERM);stopped.append(pid)
 except (FileNotFoundError,ProcessLookupError):pass
(D/'cleanup.json').write_text(json.dumps({'stopped_servers':stopped}))
z=D/'Seamless_291自动重判结果_20261109.zip'
with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED) as f:
 for p in D.iterdir():
  if p.suffix in ('.py','.json','.jsonl','.txt','.sh'):f.write(p,p.name)
(D/'bundle_sha256.txt').write_text(hashlib.sha256(z.read_bytes()).hexdigest()+'  '+z.name)
print('FINISHED',z,flush=True)
