import json,collections
from pathlib import Path
D=Path(__file__).resolve().parent
def read(p):
 out=[]
 if p.exists():
  for s in p.read_text().splitlines():
   try:out.append(json.loads(s))
   except json.JSONDecodeError:pass
 return out
q={x['id']:x for k in range(3) for x in read(D/f'qwen_{k}.jsonl')};j={x['id']:x for x in read(D/'jev.jsonl')};a={x['id']:x for x in read(D/'arbitration.jsonl')};ready=set(q)&set(j);agreed=0
for ident in ready:
 aa=q[ident].get('judgment',{}).get('label');jj=j[ident].get('answers',{});ll=[aa,jj.get('label',{}).get('answer'),jj.get('check',{}).get('answer')];agreed+=None not in ll and len(set(ll))==1
print(json.dumps({'text_completed':len(q),'audio_completed':len(j),'fully_compared':len(ready),'initial_agreement_so_far':agreed,'arbitrations_completed':len(a),'errors':{'text':sum(x['status']!='ok' for x in q.values()),'audio':sum(x['status']!='ok' for x in j.values()),'arbitration':sum(x['status']!='ok' for x in a.values())},'arbitration_complete':(D/'arbitration.log').exists() and 'ARBITRATION_COMPLETE' in (D/'arbitration.log').read_text(),'completion':json.loads((D/'completion_status.json').read_text()) if (D/'completion_status.json').exists() else None}))
