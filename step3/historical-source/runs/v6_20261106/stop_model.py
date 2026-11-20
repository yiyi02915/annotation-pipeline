"""Stop only the GLM process group created and recorded by this task."""
import os,signal,time,json
from pathlib import Path
r=Path(__file__).resolve().parent;pid=int((r/'model_server.pid').read_text());proc=Path('/proc')/str(pid)
def owned_running():
    if not proc.exists():return False
    cmd=(proc/'cmdline').read_bytes()
    return b'vllm.entrypoints.openai.api_server' in cmd and b'glm-judge' in cmd and os.getpgid(pid)==pid
stopped=False
if owned_running():
    os.killpg(pid,signal.SIGTERM)
    for _ in range(30):
        if not owned_running():break
        time.sleep(1)
    if owned_running():os.killpg(pid,signal.SIGKILL)
    stopped=True
else:assert not proc.exists() or not (proc/'cmdline').read_bytes(),'PID exists but does not match the owned GLM service'
report=dict(pid=pid,owned_service_stop_requested=stopped,original_models_unchanged=True)
(r/'service_cleanup.json').write_text(json.dumps(report,indent=2));print(report)
