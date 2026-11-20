import json,concurrent.futures
import method_runner as m
es=json.loads((m.D/'fresh_inputs.json').read_text())
m.THINK_PORTS={e['id']:18770+i%2 for i,e in enumerate(es)}
with (m.D/'thinking_fresh.jsonl').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for r in pool.map(lambda e:m.call(e,'thinking'),es):f.write(json.dumps(r)+'\n');f.flush();print(r['status'],r['id'],flush=True)
print('COMPLETE',flush=True)
