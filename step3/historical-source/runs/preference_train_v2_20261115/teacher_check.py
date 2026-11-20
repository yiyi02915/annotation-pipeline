"""Independent 27B check of TRAINING preferences; no held-out rows sent."""
import collections,json,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import model_format as fmt
root=Path(__file__).resolve().parent
source=root.parent/'preference_train_v1b_20261115'
rows={r['id']:r for r in map(json.loads,(source/'train.jsonl').read_text().splitlines())}
pairs=json.loads((root/'dedup/pairs.json').read_text())
schema=dict(type='object',properties=dict(preferred=dict(type='string',enum=['A','B']),
            before=dict(type='string'),after=dict(type='string'),reason=dict(type='string')),
            required=['preferred','before','after','reason'],additionalProperties=False)
rule=fmt.RULES+''' Compare the two local candidate labels using the exact target and speaker chronology.
Inspect the immediate expression before and after the target; do not substitute a later sentence.
An explanation of the same answer can continue after a complete sentence. An independent new contribution can occur from the same speaker.
For overlap, first establish who held the floor just before onset. A short attempted entry does not establish a successful floor transfer.
Answers to completed questions and continuing one's own expression are not interruptions solely because voices overlap.
Return the preferred candidate letter, brief literal before/after evidence and a short reason. Ignore option order.'''
def judge(item):
    pair,reverse=item;options=[pair['chosen'],pair['rejected']]
    if reverse:options.reverse()
    body=dict(evidence=fmt.evidence(rows[pair['id']]),candidates=dict(zip(['A','B'],options)))
    payload=dict(model='qwen38-27b',messages=[dict(role='system',content=rule),dict(role='user',content=json.dumps(body))],
         temperature=0,max_tokens=400,chat_template_kwargs=dict(enable_thinking=False),
         response_format=dict(type='json_schema',json_schema=dict(name='preference',strict=True,schema=schema)))
    begin=time.monotonic()
    for attempt in range(2):
        try:
            req=urllib.request.Request('http://127.0.0.1:18770/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=150) as response:raw=json.load(response)
            output=raw['choices'][0];assert output['finish_reason']=='stop'
            answer=json.loads(output['message']['content'])
            return dict(id=pair['id'],reverse=reverse,status='ok',label=options[['A','B'].index(answer['preferred'])],
                        evidence=answer,seconds=time.monotonic()-begin)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
    return dict(id=pair['id'],reverse=reverse,status='error',error=error)
begin=time.monotonic();results=[]
with ThreadPoolExecutor(4) as pool:
    for i,result in enumerate(pool.map(judge,[(r,b) for r in pairs for b in [False,True]]),1):
        results.append(result)
        if i%40==0:print('TEACHER',i,2*len(pairs),flush=True)
grouped=collections.defaultdict(list)
for r in results:grouped[r['id']].append(r)
audit={}
for pair in pairs:
    observations=grouped[pair['id']]
    outcomes=[r.get('label') for r in observations]
    status='verified' if outcomes==[pair['chosen']]*2 else ('contradicted' if outcomes==[pair['rejected']]*2 else 'unstable')
    audit[pair['id']]=dict(status=status,results=observations)
summary=dict(wall_seconds=time.monotonic()-begin,requests=len(results),counts=dict(collections.Counter(r['status'] for r in audit.values())),
    note='Independent 27B training preference agreement, not calibrated correctness; seed labels unchanged.',pairs=audit)
(root/'teacher.json').write_text(json.dumps(summary,indent=2))
print('TEACHER_COMPLETE',summary['counts'],flush=True)
