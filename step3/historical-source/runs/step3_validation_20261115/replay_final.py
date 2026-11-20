"""Replay saved model decisions through final code; does not call models or references."""
import copy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

out=Path(__file__).resolve().parent
sys.path.insert(0,str(out/'final/step3'))
from pipeline import WordTimeline,windows
from input_seamless import load_published
from audio_judge import audio_duration
from run_multi import classify

rows=[json.loads(line) for line in (out/'pairs.jsonl').read_text().splitlines()]
records={r['conversation_id']:r for r in [json.loads(p.read_text()) for p in (out/'run/recordings').glob('*.json')]}
assert len(records)==len(rows)==30
results=[]
for row,streams,tracks,problem in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),rows):
    assert not problem,problem
    duration=audio_duration(tracks)
    for words in streams.values():
        words[:]=[w for w in words if w['end']<=duration+.001]
    timeline=WordTimeline(streams)
    prior=records[row['conversation_id']]
    saved={w['id']:w for w in prior['windows']}
    events=[]
    for case,_ in windows(row['conversation_id'],timeline,duration,core=row.get('core'),analysis=row.get('interval')):
        record=saved[case['id']]
        body=json.dumps(dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(decisions=record['decisions']))))]))
        with patch('urllib.request.urlopen',side_effect=lambda *args,**kw: io.StringIO(body)):
            result=classify(case,audio=dict(clip_id=case['id'],events=copy.deepcopy(record['audio_votes'])),
                            refined=True,floor_checks=True,recover_invalid_merge=True)
        assert result['status']=='ok',result
        assert result['speaker_gate']['status']=='PASS'
        assert result['training_events']==result['events']
        lo,hi=case['target']
        events.extend(e for e in result['events'] if lo<=e['start']<hi)
    events.sort(key=lambda e:e['start'])
    results.append(dict(conversation_id=row['conversation_id'],events_equal=events==prior['events'],
                        events=len(events),windows=len(saved)))
assert all(r['events_equal'] for r in results),results
(out/'final_replay.json').write_text(json.dumps(dict(recordings=len(results),windows=sum(r['windows'] for r in results),
    events=sum(r['events'] for r in results),all_equal=True,model_requests=0,results=results),indent=2))
print('FINAL_REPLAY_PASS')
