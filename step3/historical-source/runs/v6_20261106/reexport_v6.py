"""Reassemble validated cached labels and enforce the audio-context exclusion."""
import json,time,hashlib
from pathlib import Path
from collections import Counter
from run_pipeline import export_results
root=Path(__file__).resolve().parent
load=lambda n:json.loads((root/n).read_text(encoding='utf8'))
save=lambda n,v:(root/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
records=load('records.json');boundaries=load('boundary_results.json');events=load('interaction_results.json')
independent={b['candidate_id']:b for b in load('baseline/adjudicated_boundaries.json')}
export_results(root,records,boundaries,events,independent,'glm-judge')
rows=lambda n:[json.loads(x) for x in (root/n).read_text(encoding='utf8').splitlines()]
training=rows('training_annotations.jsonl');pending=rows('pending_automatic.jsonl');safe=[];excluded=[]
for row in training:
    if row['annotation'].get('audio_context_uncertainty'):
        row.update(pending_reason='unresolved_audio_in_input_or_outcome_context',human_required=False);pending.append(row);excluded.append(row['example_id'])
    else:safe.append(row)
regions=load('unresolved_audio_regions.json')
for a in regions:pending.append(dict(task='audio_transcription_region',annotation=a,human_required=False))
for n,values in [('training_annotations.jsonl',safe),('pending_automatic.jsonl',pending)]:
    (root/n).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf8')
s=load('summary.json');s.update(version='duplex-e2e-v6',baseline_training_annotations=207,boundary_origins=dict(Counter(b.get('v6_origin') for b in boundaries)),event_origins=dict(Counter(e.get('v6_origin') for e in events)),training_annotations=len(safe),pending_items=len(pending),audio_context_excluded_training=len(excluded),unresolved_audio_regions=len(regions))
save('summary.json',s);save('audio_context_exclusions.json',excluded)
m=load('manifest.json');m.update(version='duplex-e2e-v6',effective_prompt_sha256=hashlib.sha256((root/'effective_prompts.json').read_bytes()).hexdigest(),baseline_input_sha256=hashlib.sha256((root/'baseline/records.json').read_bytes()).hexdigest(),directionality_script_sha256=hashlib.sha256((root/'resolve_directionality.py').read_bytes()).hexdigest())
save('manifest.json',m)
save('run_state.json',dict(stage='reexport_complete',unix_time=time.time(),summary=s))
