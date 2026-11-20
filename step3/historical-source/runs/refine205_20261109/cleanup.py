import os,json,signal,time,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
assert (D/'audit.json').exists()
assert 'CLARIFICATION_COMPLETE' in (D/'clarify_facts.log').read_text()
assert 'GAP_PROBE_COMPLETE' in (D/'gap_audio_probe.log').read_text()
jobs=json.loads((D/'launch.json').read_text())['jobs'];result=[]
for job in jobs:
 if not job['name'].startswith('qwen_server_'):continue
 pid=job['pid'];proc=Path('/proc')/str(pid)
 if not proc.exists():result.append({'pid':pid,'already_exited':True});continue
 args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert 'vllm.entrypoints.openai.api_server' in args and ('--port 18452' in args or '--port 18453' in args)
 assert os.getpgid(pid)==pid
 os.killpg(pid,signal.SIGTERM);result.append({'pid':pid,'terminated_own_process_group':True})
time.sleep(5)
result.append({'gpu_status':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)})
(D/'cleanup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
