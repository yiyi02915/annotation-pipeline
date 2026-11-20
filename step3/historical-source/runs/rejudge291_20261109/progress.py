from evidence import *
import collections
rows=read(D/'qwen_rejudge.jsonl')+[e for k in range(4) for e in read(D/f'qwen_rejudge_shard{k}.jsonl')];latest={e['id']:e for e in rows}
print(json.dumps({'qwen_unique':len(latest),'qwen_status':dict(collections.Counter(e.get('status') for e in latest.values())),'finish':dict(collections.Counter(e.get('finish_reason') for e in latest.values())),'errors':[e.get('error') for e in latest.values() if e.get('status')!='ok'][:5],'shards':[{'k':k,'lines':len(read(D/f'qwen_rejudge_shard{k}.jsonl')),'complete':'QWEN_COMPLETE' in (D/f'qwen_shard{k}_v2.log').read_text()} for k in range(4)],'jev_fix':len(read(D/'jev_rejudge_emptyfix.jsonl')),'output_repairs':len(read(D/'qwen_output_repair.jsonl'))},indent=2))
