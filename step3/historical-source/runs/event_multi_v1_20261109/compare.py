import json,csv
from pathlib import Path
from event_extractor import speaker_gate,finalize
D=Path(__file__).resolve().parent
rows=json.loads((D/'results.json').read_text());inputs={r['id']:r for r in json.loads((D/'word_inputs.json').read_text())};probes={r['id']:r for r in json.loads((D/'speaker_probe.json').read_text())}
comparison=[];flat=[]
speaker_evidence=json.loads((D/'speaker_evidence.json').read_text()) if (D/'speaker_evidence.json').exists() else {}
for r in rows:
 if r['status']!='ok':continue
 case=inputs[r['clip_id']];r.update(finalize(case,r['candidates'],r['judgment'],speaker_gate(case)));ev=r['events'];probe=probes[r['clip_id']]
 # This is explicit user evidence, kept distinct from the failed automated screen.
 evidence=speaker_evidence.get(r['clip_id'])
 r['speaker_gate']=speaker_gate(case,evidence);r['training_events']=ev if r['speaker_gate']['status']=='PASS' else [];r['speaker_probe']=probe
 target=[e for e in ev if e['in_original_target']];ints=[e for e in ev if e['label']=='INTERRUPT'];bcs=[e for e in ev if e['label']=='BACKCHANNEL']
 pairs=[dict(first=a['event_id'],second=b['event_id']) for i,a in enumerate(ints) for b in ints[i+1:] if (a['initiator'],a['recipient'])==(b['recipient'],b['initiator']) and b['start']<a['end']]
 nested=[dict(interruption=i['event_id'],backchannel=b['event_id']) for i in ints for b in bcs if b['initiator']==i['recipient'] and i['start']<=b['start']<i['end']]
 target_dirs={(e['initiator'],e['recipient']) for e in target if e['label']=='INTERRUPT'}
 row={'case':case['number'],'clip_id':r['clip_id'],'before_label':r['old_label'],'before_saved_events':1,'after_clip_events':len(ev),'interruptions':len(ints),'backchannels':len(bcs),'after_original_target_onsets':len(target),'target_bidirectional_interruptions':len(target_dirs)>1,'reverse_interruption_pairs':len(pairs),'nested_backchannels':len(nested),'speaker_gate':r['speaker_gate']['status'],'gate_source':r['speaker_gate']['source'],'training_events':len(r['training_events'])}
 comparison.append(row)
 for e in ev:flat.append(dict(case=case['number'],speaker_gate=r['speaker_gate']['status'],training_eligible=r['speaker_gate']['status']=='PASS',**e))
(D/'annotation_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
(D/'before_after.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf8')
with (D/'before_after.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
with (D/'events.audit.jsonl').open('w',encoding='utf8') as f:
 for e in flat:f.write(json.dumps(e,ensure_ascii=False)+'\n')
with (D/'events.jsonl').open('w',encoding='utf8') as f:
 for e in flat:
  if e['speaker_gate']!='EXCLUDE':f.write(json.dumps(e,ensure_ascii=False)+'\n')
with (D/'training_events.jsonl').open('w',encoding='utf8') as f:
 for e in flat:
  if e['training_eligible']:f.write(json.dumps(e,ensure_ascii=False)+'\n')
summary={'processed':len(comparison),'model_errors':sum(r['status']!='ok' for r in rows),'candidates':sum(len(r.get('candidates',[])) for r in rows),'before_saved_events':len(comparison),'after_clip_events':len(flat),'interruptions':sum(r['interruptions'] for r in comparison),'backchannels':sum(r['backchannels'] for r in comparison),'third_speaker_automatically_detected':sum(any(a.get('answer')=='THREE_OR_MORE' for a in p.get('answers',{}).values()) for p in probes.values()),'excluded_by_user_evidence':[r['case'] for r in comparison if r['speaker_gate']=='EXCLUDE'],'acoustic_gate_unverified':[r['case'] for r in comparison if r['speaker_gate']=='UNVERIFIED'],'training_events':sum(r['training_events'] for r in comparison),'validation_scope':'real word-aligned multi-event model rerun; no independent gold; JEV-4B speaker screen experimental and not allowed to certify two speakers'}
(D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False));print(json.dumps(comparison,ensure_ascii=False))
