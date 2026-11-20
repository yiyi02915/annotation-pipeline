import os,signal,json,time,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;assert (D/'audit.json').exists();assert 'ARBITRATION_COMPLETE' in (D/'arbitration.log').read_text();results=[]
for job in json.loads((D/'launch.json').read_text())['jobs']:
 pid=job['pid'];p=Path('/proc')/str(pid)
 if not p.exists():continue
 args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'vllm.entrypoints.openai.api_server' in args and f'--port {job["port"]}' in args and os.getpgid(pid)==pid
 os.killpg(pid,signal.SIGTERM);results.append({'pid':pid,'stopped':True})
time.sleep(5);results.append({'gpu_status':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)});(D/'cleanup.json').write_text(json.dumps(results,indent=2));print(results)
