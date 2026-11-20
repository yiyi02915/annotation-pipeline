"""Close only this experiment's extra inference service after all checkpoints."""
import json,os,signal
from pathlib import Path
root=Path(__file__).resolve().parent
for name in ('anchored_complete.json','context_repair_complete.json','baseline_correction_complete.json'):
    state=json.loads((root/name).read_text());assert state['status']=='complete' and state['errors']==0
pid=int((root/'model.pid').read_text())
cmd=Path(f'/proc/{pid}/cmdline')
if cmd.exists():
    args=cmd.read_bytes().split(b'\0')
    assert b'vllm.entrypoints.openai.api_server' in args and b'18771' in args
    os.kill(pid,signal.SIGTERM)
    print('Stopped experiment service on port 18771; original 18770 service untouched.')
else:print('Experiment service already stopped.')
