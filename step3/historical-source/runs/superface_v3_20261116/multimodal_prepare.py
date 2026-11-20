"""Build traceable audit inputs; never infer a missing waveform or speaker track."""
import collections
import hashlib
import json
import sys
from pathlib import Path

BASE = Path('/data/L202500493/duplex_postprocess')
RUN = BASE / 'runs/superface_v3_20261116'
OLD = BASE / 'runs/preference_train_v1b_20261115'
OB = BASE / 'runs/ob141_smoketest_20261112'
sys.path.insert(0, str(RUN / 'step3'))
from audio_judge import audio_duration, render_mix
from input_seamless import load_published
from pipeline import context


def write(name, value):
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    import soundfile as sf
    from duplexjev.decider import load_audio
    rows = [json.loads(s) for s in (OLD / 'train.jsonl').read_text().splitlines()]
    old = json.loads((BASE / 'runs/preference_train_v2_20261115/teacher_targeted.json').read_text())['pairs']
    suspect = {k: v for k, v in old.items() if v['status'] != 'verified'}
    assert len(suspect) == 21
    source = {}
    for name in ['inputs.json', 'holdout_inputs.json', 'blind100_inputs.json', 'risk_blind_inputs.json']:
        for r in json.loads((OB / name).read_text()):
            source[r['id']] = r
    ids = sorted({r['conversation_id'] for r in rows if r['conversation_id'].startswith('V')})
    seamless = {}
    for r, streams, tracks, error in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'), [dict(conversation_id=k) for k in ids]):
        seamless[r['conversation_id']] = (streams, tracks, error)
    output = []
    for row in rows:
        item = json.loads(json.dumps(row))
        item['seed_label'] = row['label']
        item['prior_status'] = old[row['id']]['status']
        item['audit_required'] = row['id'] in suspect
        target = row['input']['target']
        a, b = target.get('absolute_gap', [target.get('start'), target.get('end')])
        digest = hashlib.sha256(row['id'].encode()).hexdigest()[:20]
        audio = RUN / 'audio' / (digest + '.wav')
        audio.parent.mkdir(exist_ok=True)
        flags = []
        try:
            if row['conversation_id'].startswith('V'):
                streams, tracks, error = seamless[row['conversation_id']]
                if error:
                    raise ValueError(error)
                duration = audio_duration(tracks)
                lo, hi = max(0, a-12), min(duration, b+12)
                render_mix(tracks, [lo, hi], audio)
                nearby = {k: [w for w in v if w['end'] > lo and w['start'] < hi] for k, v in streams.items()}
                item['input']['context'] = context(nearby)
                item['audio_source'] = dict(kind='synchronized_dual_track_mix', tracks={k: [str(p), ch] for k, (p, ch) in tracks.items()})
                focus = target.get('initiator', row['input'].get('focus'))
                words = nearby[focus]
                item['input']['focus_after_target'] = ' '.join(w['text'] for w in words if w['start'] >= a-.03)
            else:
                original = source[row['id']]
                path = OB / original['source_audio']
                info = sf.info(path)
                duration = info.duration
                lo, hi = max(0, a-12), min(duration, b+12)
                with sf.SoundFile(path) as f:
                    f.seek(round(lo*f.samplerate))
                    waveform = f.read(round((hi-lo)*f.samplerate), dtype='float32', always_2d=True)
                    waveform = load_audio((waveform, f.samplerate))
                sf.write(audio, waveform, 16000)
                item['audio_source'] = dict(kind='existing_ob141_conversation_audio', path=str(path), channels=info.channels, independent_tracks_available=False)
                # OB141 stored waveform is not falsely represented as two independent tracks.
                if any(x.get('speaker') == 'UNKNOWN' for x in item['input'].get('context', [])):
                    flags.append('speaker_attribution_incomplete')
            if not lo <= a < b <= hi:
                raise ValueError('target_outside_audio')
            item['audio'] = str(audio)
            item['audio_interval'] = [lo, hi]
            item['target_relative'] = [a-lo, b-lo]
            item['audio_ready'] = True
        except Exception as exc:
            item['audio_ready'] = False
            flags.append('audio_or_alignment_error')
            item['preparation_error'] = f'{type(exc).__name__}: {exc}'
        item['risk_reasons'] = flags
        output.append(item)
    write('prepared_train.json', output)
    write('audit21_inputs.json', [r for r in output if r['audit_required']])
    write('checkpoint_01_preparation.json', dict(total=len(output), audit21=sum(r['audit_required'] for r in output), audio_ready=sum(r['audio_ready'] for r in output), errors=[dict(id=r['id'], error=r.get('preparation_error')) for r in output if not r['audio_ready']], reference_type='automatic historical references; not human gold'))
    print('PREPARATION', len(output), sum(r['audio_ready'] for r in output), flush=True)


if __name__ == '__main__':
    main()
