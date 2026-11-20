"""Automatic repair, incremental strong-model labeling, recheck and export."""
import argparse,json,subprocess,sys,time,os,signal,hashlib
from pathlib import Path
import requests
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--refresh-audio',action='store_true');p.add_argument('--require-cache',action='store_true');args=p.parse_args()
asr_python='/data/L202500293/envs/qwen3-asr/bin/python'
run=[];owned_pid=None;began=time.time()
digest=lambda:hashlib.sha256((root/'model_calls.jsonl').read_bytes()).hexdigest() if (root/'model_calls.jsonl').exists() else None
before=digest()
def step(script,python=sys.executable):
    start=time.time()
    with (root/('full_'+script[:-3]+'.log')).open('wb') as log:
        cp=subprocess.run([python,str(root/script)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    run.append(dict(script=script,seconds=round(time.time()-start,3),exit_code=cp.returncode))
    if cp.returncode:raise RuntimeError(script+' failed; inspect its full_ log')
    print(script,'complete',flush=True)
try:
    try:healthy=requests.get('http://127.0.0.1:18351/v1/models',timeout=3).ok
    except requests.RequestException:healthy=False
    if not healthy:
        step('start_model.py');owned_pid=int((root/'model_server.pid').read_text())
    if args.refresh_audio or not (root/'alignment_evidence.json').exists():step('repair_alignment.py',asr_python)
    if args.refresh_audio or not (root/'focused_evidence.json').exists():step('focused_audio.py',asr_python)
    for script in ['apply_alignment.py','apply_focused.py','run_v6.py','resolve_directionality.py','reexport_v6.py','finalize_v6.py']:step(script)
    if args.require_cache:assert before==digest(),'Unexpected new model calls in cache-only reproducibility run'
    report=dict(complete=True,steps=run,audio_evidence_refreshed=args.refresh_audio,new_model_calls_added=before!=digest(),elapsed_seconds=round(time.time()-began,3),human_review_used=False)
    (root/'full_pipeline_run.json').write_text(json.dumps(report,indent=2))
    (root/'run_state.json').write_text(json.dumps(dict(stage='complete',unix_time=time.time(),summary=json.loads((root/'summary.json').read_text())),indent=2))
    print(json.dumps(report,indent=2))
finally:
    if owned_pid is not None:
        proc=Path('/proc')/str(owned_pid)
        if proc.exists() and b'glm-judge' in (proc/'cmdline').read_bytes() and os.getpgid(owned_pid)==owned_pid:os.killpg(owned_pid,signal.SIGTERM)
