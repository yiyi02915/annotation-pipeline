import hashlib,json,collections
from evidence import *
events=read(D/'inputs.jsonl');expected={e['id'] for e in events};assert len(events)==len(expected)==291
for name,rows in [('asr',read(D/'asr_0.jsonl')+read(D/'asr_1.jsonl')),('fa',read(D/'fa_0.jsonl')+read(D/'fa_1.jsonl')),('local_fa',read(D/'local_fa.jsonl')),('jev',read(D/'jev_rejudge.jsonl'))]:
 assert len(rows)==291 and {e['id'] for e in rows}==expected,name
affected={e['id'] for e in events if any(not norm(tr['text']) for tr in e['tracks'].values())};fix=read(D/'jev_rejudge_emptyfix.jsonl');assert {e['id'] for e in fix}==affected
q={e['id']:e for e in read(D/'qwen_rejudge.jsonl')+[x for k in range(4) for x in read(D/f'qwen_rejudge_shard{k}.jsonl')]};assert set(q)==expected
assert all(q[k].get('evidence_version')==2 for k in affected)
final=read(D/'adjudicated.jsonl');assert {e['id'] for e in final}==expected
for e in final:
 if e['status']=='resolved_candidate':assert not e['reasons'] and e['final_label']==e['qwen_label']==e['jev_label']==e['jev_check'] and e['final_label']!='UNCERTAIN'
 if e['status']=='no_longer_pause_candidate':assert e['alignment']['anchor']['new_gap']<=.5 and e['local_alignment']['gap']<=.5
asr=read(D/'asr_0.jsonl')+read(D/'asr_1.jsonl');fa=read(D/'fa_0.jsonl')+read(D/'fa_1.jsonl');repair=read(D/'qwen_output_repair.jsonl')
audit={'selected':291,'unique_final':len(final),'empty_track_fix_events':len(affected),'qwen_consistency_rechecks':len(repair),'qwen_recheck_errors':sum(e.get('status')!='ok' for e in repair),'asr_track_errors':sum('error' in v for e in asr for v in e['tracks'].values()),'fa_track_errors':sum('error' in v for e in fa for v in e['tracks'].values()),'local_fa_errors':sum('error' in e for e in read(D/'local_fa.jsonl')),'schema_and_partition_checks':'passed','accuracy_claimed':False,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.suffix in ('.py','.jsonl') and p.name!='adjudicated.jsonl'}}
(D/'output_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='files_sha256'},indent=2))
