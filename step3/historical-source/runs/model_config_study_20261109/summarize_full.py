import collections,json,math
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent/'finegrained_pilot_20261109'
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
rows=[];runtime=[]
for i in range(4):
 part=read(D/f'jev32B_enum_p4_full_shard{i}.jsonl');assert len(part)==750
 rows+=part;runtime.append(json.loads((D/f'jev32B_full_shard{i}_runtime.json').read_text()))
base=read(B/'benchmark_inputs.jsonl');expected={e['id'] for e in base};byid={e['id']:e for e in rows};assert len(rows)==len(byid)==3000 and set(byid)==expected
labels={'overlap':{'BACKCHANNEL','INTERRUPT','OTHER','UNCERTAIN'},'pause':{'HESITATION','TURN_TRANSITION','OTHER','UNCERTAIN'}}
for e in rows:
 assert e['label'] in labels[e['task']] and e['label']==max(e['probs'],key=e['probs'].get)
 assert abs(sum(e['probs'].values())-1)<1e-4 and all(math.isfinite(v) and 0<=v<=1 for v in e['probs'].values())
q={e['id']:e for e in read(B/'qwen_predictions.jsonl')};j={e['id']:e for e in read(B/'jev_predictions.jsonl')}
summary={'n':3000,'by_task':{},'accuracy_verified':False,'config':{'model':'DuplexJev-32B-Para v1.5','wording':'enum','n_perm':4,'seed':0,'batch_size_per_gpu':1,'gpus':4},'comparison_note':'Old4B differs in both size and n_perm; agreement is not accuracy.'}
for task in labels:
 pool=[e for e in rows if e['task']==task];assert len(pool)==1500
 summary['by_task'][task]=dict(n=len(pool),labels=dict(collections.Counter(e['label'] for e in pool)),qwen_agreements=sum(e['label']==q[e['id']]['label'] for e in pool),old_jev4_agreements=sum(e['label']==j[e['id']]['label'] for e in pool),target_consensus_with_qwen=sum(e['label']==q[e['id']]['label'] and e['label'] not in ('OTHER','UNCERTAIN') for e in pool),qwen_matrix=dict(collections.Counter(q[e['id']]['label']+'|'+e['label'] for e in pool)))
small={e['id']:e for e in read(D/'jev32B_enum_p4.jsonl')};summary['pilot_repeat_changes']=[k for k,e in small.items() if byid[k]['label']!=e['label']]
runs=[x for r in runtime for x in r['runs']];start=min(x['started_at'] for x in runs);end=max(x['finished_at'] for x in runs)
launch=json.loads((D/'full_launch_metadata.json').read_text());coldstart=min(x['process_started_at'] for x in launch['workers'])
summary['runtime']=dict(compute_window_seconds=end-start,compute_window_events_per_second=3000/(end-start),sum_worker_processing_seconds=sum(x['seconds'] for x in runs),process_to_last_result_seconds=end-coldstart,model_load_seconds=[r['load_seconds'] for r in runtime],scope='4 GPUs concurrently; compute window from first worker first task to last task finished, includes inter-task IO and first inference; process window additionally includes startup/import/load')
with (D/'jev32B_enum_p4_full.jsonl').open('w') as f:
 for e in sorted(rows,key=lambda e:e['id']):f.write(json.dumps(e)+'\n')
(D/'full_population_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
