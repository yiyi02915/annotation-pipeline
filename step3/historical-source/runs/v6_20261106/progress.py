import json,statistics
from collections import Counter
from pathlib import Path
r=Path(__file__).resolve().parent
if (r/'run_state.json').exists():print('state',json.loads((r/'run_state.json').read_text())['stage'])
for name,key in [('boundary_results.json','decision'),('interaction_results.json','status')]:
    if (r/name).exists():
        d=json.loads((r/name).read_text());print(name,len(d),dict(Counter(x[key] for x in d)),dict(Counter(x.get('v6_origin') for x in d)))
if (r/'model_calls.jsonl').exists():
    calls=[json.loads(s) for s in (r/'model_calls.jsonl').read_text().splitlines()]
    print('calls',len(calls),'parse_errors',sum(x['parsed'] is None for x in calls))
    print('recent_seconds',[x['seconds'] for x in calls[-4:]])
    print('recent_tokens',[(x.get('raw') or {}).get('usage',{}).get('completion_tokens') for x in calls[-4:]])
