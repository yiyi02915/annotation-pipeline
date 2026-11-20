import json
from pathlib import Path
import sys
import time

BASE=Path('/data/L202500493/duplex_postprocess/runs/step3_validation_20261115')
sys.path.insert(0,str(BASE/'final/step3'))
from pipeline import WordTimeline,windows,gap_candidates,context
from event_extractor import extract_candidates
from input_seamless import load_published
from audio_judge import audio_duration,render_mix

out=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (out/'pairs.jsonl').read_text().splitlines()]
records=[]
for row,streams,tracks,problem in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),rows):
    assert not problem,problem
    duration=audio_duration(tracks)
    for words in streams.values():
        words[:]=[w for w in words if w['end']<=duration+.001]
    timeline=WordTimeline(streams)
    cases=[case for case,_ in windows(row['conversation_id'],timeline,duration,core=row.get('core'),analysis=row.get('interval'))]
    gaps=list(gap_candidates(row['conversation_id'],timeline,row.get('core'),row.get('interval')))
    for case in cases:
        case['extended_context']=context(timeline.between(max(0,case['interval'][0]-8),min(duration,case['interval'][1]+8)))
    records.append(dict(row=row,duration=duration,cases=cases,gaps=gaps,
                        tracks={k:[str(p),ch] for k,(p,ch) in tracks.items()}))
(out/'inputs.json').write_text(json.dumps(records,indent=2))
print('INPUTS_READY',flush=True)

from duplexjev import Decider,Question
loaded=time.monotonic()
engine=Decider.from_pretrained('/data/L202500493/duplex_postprocess/models/DuplexJev-32B-Para',device='cuda:0',attn_implementation='sdpa')
load_seconds=time.monotonic()-loaded
question=Question('gap','Classify the marked gap for the focus speaker. PAUSE means RIGHT completes, explains, supports, repairs or restates the same contribution as LEFT. TURN_TRANSITION means the prior contribution ended or was abandoned and RIGHT starts an independent response or contribution. A new subject is not required. Brief listener feedback does not by itself create a new turn. Identify which specific contribution the post-gap words respond to; a new acknowledgment to a new intervening substantive turn is a new response. Use the synchronized audio and context, not the silence length alone.', ['PAUSE','TURN_TRANSITION'],audio='mix')
ordinary=Question('entry','At the specified target onset, which describes the target speaker? An ordinary answer completes the other speaker\'s already available question without displacing it. Continuing one\'s own floor completes the same explanation or list despite another participant\'s attempted entry; it is not a new attempt to reclaim an acquired floor. An intervention either supplies supporting listener feedback or makes a distinct bid against the established current speaker. Judge actual floor ownership from the preceding and following speech, not overlap alone.', ['ORDINARY_ANSWER','CONTINUES_OWN_FLOOR','INTERVENTION'],audio='mix')
target_ids={'V00_S1517_I00000694:P0933A:w105->P0004A','V00_S1833_I00000716:P1154A:w179->P1155A','V00_S1833_I00000716:P1154A:w184->P1155A'}
votes={'gap':{},'entry':{},'load_seconds':load_seconds}
audio_dir=out/'audio';audio_dir.mkdir(exist_ok=True)
started=time.monotonic()
for record in records:
    tracks={k:(Path(p),ch) for k,(p,ch) in record['tracks'].items()}
    for gap in record['gaps']:
        lo,hi=gap['target']['absolute_gap'];span=[max(0,lo-6),min(record['duration'],hi+6)]
        path=render_mix(tracks,span,audio_dir/f"gap_{len(votes['gap']):03d}.wav")
        info=dict(gap,clip_absolute_interval=span,gap_relative_to_audio=[lo-span[0],hi-span[0]])
        answer=engine.decide_batch({'mix':path},[question],context={'mix':json.dumps(info)},lang='en',mode='packed',max_items=1,n_perm=4,seed=0)['mix']['gap']
        votes['gap'][gap['id']]=dict(answer=answer,audio=path,interval=span)
        (out/'audio_votes.json').write_text(json.dumps(votes,indent=2))
        print('GAP_AUDIO',len(votes['gap']),flush=True)
    for case in record['cases']:
        for candidate in extract_candidates(case,refined=True,floor_checks=True):
            if candidate['candidate_id'] not in target_ids:continue
            span=[max(0,candidate['start']-8),min(record['duration'],candidate['end']+5)]
            path=render_mix(tracks,span,audio_dir/f"entry_{len(votes['entry']):03d}.wav")
            info=dict(target=candidate,context=case['extended_context'],clip_absolute_interval=span,target_relative_to_audio=candidate['start']-span[0])
            answer=engine.decide_batch({'mix':path},[ordinary],context={'mix':json.dumps(info)},lang='en',mode='packed',max_items=1,n_perm=4,seed=0)['mix']['entry']
            votes['entry'][candidate['candidate_id']]=dict(answer=answer,audio=path,interval=span)
            (out/'audio_votes.json').write_text(json.dumps(votes,indent=2))
votes['inference_wall_seconds']=time.monotonic()-started
(out/'audio_votes.json').write_text(json.dumps(votes,indent=2))
print('AUDIO_COMPLETE',flush=True)
