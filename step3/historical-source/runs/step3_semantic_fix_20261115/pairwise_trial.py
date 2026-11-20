import json,time,sys,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
out=Path(__file__).resolve().parent
sys.path.insert(0,str(out/'step3'))
from pipeline import WordTimeline,gap_candidates
from input_seamless import load_published
from gap_judge import BASE_RULES,BOUNDARY_CHECKS
rows=[json.loads(s) for s in (out/'pairs.jsonl').read_text().splitlines()]
tasks=[]
for row,streams,tracks,problem in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),rows):
    assert not problem
    tasks.extend(gap_candidates(row['conversation_id'],WordTimeline(streams),row.get('core'),row.get('interval')))
rules=BASE_RULES+BOUNDARY_CHECKS+' Compare the two candidate annotations for ONLY this marked gap. Both candidates have equal prior weight. Select the annotation better supported by the exact chronology and local semantic relation. Ignore candidate order. Return preferred A or B, literal evidence before and after, and a concise reason. Candidate names and prior model predictions are not evidence.'
def judge(item):
    payload,reverse=item
    options=['PAUSE','TURN_TRANSITION'][::(-1 if reverse else 1)]
    schema=dict(type='object',properties=dict(preferred=dict(type='string',enum=['A','B']),reason=dict(type='string'),before=dict(type='string'),after=dict(type='string')),required=['preferred','reason','before','after'],additionalProperties=False)
    data=dict(model='qwen38-27b',messages=[dict(role='system',content=rules),dict(role='user',content=json.dumps(dict(evidence=payload,candidates=dict(zip(['A','B'],options)))))],temperature=0,max_tokens=700,chat_template_kwargs=dict(enable_thinking=False),response_format=dict(type='json_schema',json_schema=dict(name='preference',strict=True,schema=schema)))
    start=time.monotonic()
    for attempt in range(2):
        try:
            req=urllib.request.Request('http://127.0.0.1:18770/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as response:raw=json.load(response)
            choice=raw['choices'][0];assert choice['finish_reason']=='stop'
            decision=json.loads(choice['message']['content']);label=options[['A','B'].index(decision['preferred'])]
            return dict(id=payload['id'],reverse=reverse,status='ok',label=label,decision=decision,seconds=time.monotonic()-start)
        except Exception as exc:error=str(exc)
    return dict(id=payload['id'],reverse=reverse,status='error',error=error)
start=time.monotonic()
with ThreadPoolExecutor(4) as pool:
    results=list(pool.map(judge,[(p,r) for p in tasks for r in (False,True)]))
grouped={}
for result in results:grouped.setdefault(result['id'],[]).append(result)
summary=dict(requests=len(results),wall_seconds=time.monotonic()-start,results=grouped,
             note='Untrained same-model AB/BA preference pilot, not trained SuperFace discriminator or DPO. No references supplied.')
(out/'pairwise_results.json').write_text(json.dumps(summary,indent=2))
print('PAIRWISE_COMPLETE',flush=True)
