import collections,json,hashlib
from pathlib import Path
from benchmark_common import ROOT,read_inputs
rows=read_inputs();assert len(rows)==3000
assert len({r['id'] for r in rows})==3000
assert collections.Counter(r['task'] for r in rows)=={'overlap':1500,'pause':1500}
for r in rows:
    assert r['split']=='train' and r['focus']!=r['other']
    assert r['reference_label'] is None
    assert set(r['audio_paths'])=={r['focus'],r['other']}
    assert Path(r['audio_clip']).is_file()
    assert hashlib.sha256(Path(r['audio_clip']).read_bytes()).hexdigest()==r['audio_clip_sha256']
    if r['task']=='pause':
        assert r['gap_seconds']>.5
        assert abs(r['gap_seconds']-(r['right_word']['start']-r['left_word']['end']))<1e-5
    else:
        assert r['overlap_seconds']>=.08 and r['focus_word_ids']
per=collections.Counter((r['task'],r['conversation_id']) for r in rows)
assert max(per.values())<=12
bins=collections.Counter()
for r in rows:
    if r['task']=='pause':
        g=r['gap_seconds'];bins['0.5-1' if g<=1 else '1-3' if g<=3 else '3-10' if g<=10 else '10-30' if g<=30 else '>30']+=1
result={'passed':True,'count':len(rows),'task_counts':dict(collections.Counter(r['task'] for r in rows)),
        'max_per_task_per_conversation':max(per.values()),'pause_gap_seconds_bins':dict(bins),
        'checks':['unique IDs','strict >0.5 seconds gaps','two distinct speakers','audio clip hashes',
                  'train split only','no reference labels fed into model','per-conversation cap'],
        'behavioral_accuracy_verified':False,'sampled_audio_seconds':sum(r['audio_clip_seconds'] for r in rows)}
(ROOT/'input_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
