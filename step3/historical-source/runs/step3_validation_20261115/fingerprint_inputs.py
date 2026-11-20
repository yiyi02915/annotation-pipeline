"""Compare model inputs between the live-run snapshot and final housekeeping fixes."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
from pipeline import WordTimeline, windows, gap_candidates
from event_extractor import extract_candidates
from input_seamless import load_published
from audio_judge import audio_duration

out=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (out/'pairs.jsonl').read_text().splitlines()]
root=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
fingerprints={}
for row,streams,tracks,problem in load_published(root,rows):
    if problem:
        raise ValueError(problem)
    duration=audio_duration(tracks)
    for words in streams.values():
        words[:]=[w for w in words if w['end']<=duration+.001]
    timeline=WordTimeline(streams)
    cases=[]
    for case,onsets in windows(row['conversation_id'],timeline,duration,core=row.get('core'),analysis=row.get('interval')):
        cases.append(dict(case=case,onsets=onsets,final_candidates=extract_candidates(case,refined=True,floor_checks=True)))
    payload=dict(duration=duration,cases=cases,gaps=list(gap_candidates(row['conversation_id'],timeline,row.get('core'),row.get('interval'))))
    fingerprints[row['conversation_id']]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
Path(sys.argv[2]).write_text(json.dumps(fingerprints,indent=2))
