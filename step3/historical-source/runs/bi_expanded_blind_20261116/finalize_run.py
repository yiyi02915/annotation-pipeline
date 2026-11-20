import hashlib,json,os,signal,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
lock=json.loads((ROOT/'source_lock.json').read_text())
assert all(hashlib.sha256((ROOT/'step3'/n).read_bytes()).hexdigest()==h for n,h in lock['source_hashes'].items())
inf=json.loads((ROOT/'inference_lock.json').read_text())
assert hashlib.sha256((ROOT/'reference_locked.json').read_bytes()).hexdigest()==inf['reference_sha256']
assert hashlib.sha256((ROOT/'inputs.json').read_bytes()).hexdigest()==inf['input_sha256']
evaluation=json.loads((ROOT/'evaluation.json').read_text());assert evaluation['metrics']['optimized']['all']['valid']==137
pid=int((ROOT.parent/'bi_expanded_20261116/model.pid').read_text());path=Path('/proc')/str(pid)/'cmdline'
stopped=False
if path.exists():
 cmd=path.read_bytes().replace(b'\0',b' ').decode()
 assert 'vllm.entrypoints.openai.api_server' in cmd and '--port 18771' in cmd
 os.kill(pid,signal.SIGTERM);stopped=True
time.sleep(3)
try:
 with urllib.request.urlopen('http://127.0.0.1:18770/health',timeout=5) as r:original_healthy=r.status==200
except Exception:original_healthy=False
result=dict(at=datetime.now(timezone.utc).isoformat(),event='EXPERIMENT_COMPLETED',windows=30,positions=137,
 source_and_reference_and_input_hashes_unchanged=True,experimental_server_stopped=stopped,original_server_healthy=original_healthy)
(ROOT/'completion_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
