import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parent
for _ in range(720):
    ready='DOWNLOAD_COMPLETE' in (root/'download_jev.log').read_text()
    if ready:break
    time.sleep(5)
else:raise TimeoutError('Model download not complete after one hour')
with (root/'jev_benchmark.log').open('w') as log:
    p=subprocess.run(['bash',str(root/'start_jev_benchmark.sh')],stdout=log,stderr=subprocess.STDOUT)
(root/'jev_run_state.json').write_text(json.dumps({'stage':'complete' if p.returncode==0 else 'failed','exit_code':p.returncode}))
raise SystemExit(p.returncode)
