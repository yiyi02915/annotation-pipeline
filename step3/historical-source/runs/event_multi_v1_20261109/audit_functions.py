import json,re,time
from pathlib import Path
import run_multi
D=Path(__file__).resolve().parent
while not (D/'run_v2.log').exists() or 'MULTI_COMPLETE' not in (D/'run_v2.log').read_text():time.sleep(5)
rows=json.loads((D/'results.json').read_text());cases={r['id']:r for r in json.loads((D/'word_inputs.json').read_text())}
import importlib,event_extractor
importlib.reload(event_extractor);importlib.reload(run_multi)
for r in rows:
 if r['status']=='error':
  case=cases[r['clip_id']];cs=event_extractor.extract_candidates(case)
  for attempt in reversed(r['attempts']):
   if 'response' not in attempt:continue
   try:
    judgment=json.loads(attempt['response']['choices'][0]['message']['content'])
    fixed=event_extractor.finalize(case,cs,judgment,event_extractor.speaker_gate(case));r.update(fixed,status='ok',candidates=cs,judgment=judgment,old_label=case['original_label'],target=case['target'],structural_recovery='DROP inheritance for continuation; semantic decisions unchanged');break
   except Exception:continue
cue={'okay','ok','yeah','yes','right','sure','oh','mhm','uhhuh','mmhmm'}
selected=[]
for r in rows:
 if r['status']!='ok':continue
 cs={c['candidate_id']:c for c in r['candidates']}
 for e in r['events']:
  first=cs[e['candidate_ids'][0]]
  if e['label']=='INTERRUPT' and len(e['candidate_ids'])>1 and re.sub(r'[^a-z]','',first['text'].lower()) in cue:
   selected.append(r['clip_id']);break
(D/'results.before_function_audit.json').write_text(json.dumps(rows,indent=2))
run_multi.RULES+='\n'+(D/'function_boundary_audit.txt').read_text()
for ident in selected:
 result=run_multi.classify(cases[ident]);result['function_boundary_audit']=True
 if result['status']=='ok':rows=[result if r['clip_id']==ident else r for r in rows]
 else:(D/('audit_error_'+cases[ident]['number']+'.json')).write_text(json.dumps(result))
 (D/'results.json').write_text(json.dumps(rows,indent=2));print('FUNCTION_AUDIT',cases[ident]['number'],result['status'],flush=True)
(D/'function_audit_selection.json').write_text(json.dumps({'selected':selected,'rule':'INTERRUPT merged event beginning with standalone acknowledgment token; no per-case hardcoding'}))
print('AUDIT_COMPLETE',flush=True)
