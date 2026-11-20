import json
import math
from collections import Counter
from pathlib import Path
import soundfile as sf

root = Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
out = Path(__file__).resolve().parent
ids = {json.loads(line)['conversation_id'] for line in (out/'random2000.jsonl').read_text().splitlines()}
counts, cases = Counter(), []
entries = {}
for path in (root/'training/index_v1').glob('*.jsonl'):
    for line in path.open():
        row = json.loads(line)
        if row['interaction_id'] in ids:
            entries[row['interaction_id']] = row
with (root/'seamless_fd_data/metadata.jsonl').open('rb') as source:
    for cid, entry in entries.items():
        source.seek(entry['metadata_byte_offset'])
        meta = json.loads(source.read(entry['metadata_bytes']))
        issues = []
        counts['participant_count_'+str(meta['participant_count'])] += 1
        if meta['participant_count'] != 2:
            issues.append(dict(kind='participant_metadata', count=meta['participant_count']))
        durations = []
        for participant in entry['participants']:
            path = Path(participant['audio_path'])
            if not path.is_file():
                path = root/path.as_posix().split('/seamless_fd/', 1)[1]
            durations.append(sf.info(path).duration)
        if len(durations) == 2 and abs(durations[0]-durations[1])>.08:
            issues.append(dict(kind='duration_mismatch', durations=durations))
        for participant in meta['participants']:
            valid = []
            for i,w in enumerate(participant['words']):
                a,b=float(w['start_sec']),float(w['end_sec'])
                if math.isfinite(a) and math.isfinite(b) and 0<=a<b and w['word'].strip():
                    valid.append(dict(index=i,start=a,end=b,text=w['word']))
                else:
                    counts['invalid_words'] += 1
            if not valid:
                issues.append(dict(kind='empty_word_track',sample_id=participant['sample_id']))
            valid.sort(key=lambda w:(w['start'],w['end']))
            frontier = 0
            for a,b in zip(valid, valid[1:]):
                frontier = max(frontier,a['end'])
                if a['end']-a['start']>2:
                    counts['long_words'] += 1
                if b['start']<a['end']-.001:
                    counts['overlapping_words'] += 1
                if b['start']-a['end']>.5:
                    if frontier>b['start']:
                        issues.append(dict(kind='false_gap_under_active_word',left=a,right=b))
                    if b['index'] != a['index']+1:
                        issues.append(dict(kind='gap_across_missing_or_reordered_word',left=a,right=b))
        if issues:
            cases.append(dict(conversation_id=cid,issues=issues))
            counts.update(i['kind'] for i in issues)
(out/'source_audit.json').write_text(json.dumps(dict(counts=counts,cases=cases),indent=2))
print(json.dumps(counts))
