import json,collections
from pathlib import Path
D=Path(__file__).resolve().parent
def read(p):
 out=[]
 if not p.exists():return out
 for line in p.read_text().splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:pass
 return out
q=[x for k in range(3) for x in read(D/f'qwen_{k}.jsonl')];j=read(D/'jev.jsonl');a=read(D/'arbitration.jsonl')
qm={(x['id'],x['view']):x for x in q if x['status']=='ok'};jm={x['id']:x for x in j if x['status']=='ok'};ready={ident for ident,v in qm if (ident,'a') in qm and (ident,'b') in qm and ident in jm};agreed=0
for ident in ready:
 labels=[qm[ident,'a']['judgment']['label'],qm[ident,'b']['judgment']['label'],jm[ident]['answers']['label']['answer'],jm[ident]['answers']['check']['answer']];agreed+=len(set(labels))==1
print(json.dumps({'fully_compared_events':len(ready),'four_view_agreement_so_far':agreed,'disagreements_so_far':len(ready)-agreed}))
print(json.dumps({'text_views_completed':len(q),'text_events_with_both_views':sum(n==2 for n in collections.Counter(x['id'] for x in q).values()),'text_errors':sum(x['status']!='ok' for x in q),'audio_events_completed':len(j),'audio_errors':sum(x['status']!='ok' for x in j),'arbitrations_completed':len(a),'arbitrations_required':len(json.loads((D/'arbitration_selection.json').read_text())['ids']) if (D/'arbitration_selection.json').exists() else None,'arbitration_complete':(D/'arbitration.log').exists() and 'ARBITRATION_COMPLETE' in (D/'arbitration.log').read_text()}))
