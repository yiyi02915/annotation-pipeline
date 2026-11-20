"""Compare window predictions against a separately locked semantic reference.

This evaluator never calls a model and must not be imported into inference.
Reference labels are not automatically established human ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

LABELS = ('INTERRUPT', 'BACKCHANNEL', 'NEITHER')


def label_at(window, candidate_id):
    if window.get('status') != 'ok':
        return None
    hits = [e['label'] for e in window.get('events', [])
            if e.get('candidate_ids') and e['candidate_ids'][0] == candidate_id]
    if len(hits) > 1:
        raise ValueError('Duplicate event root: ' + candidate_id)
    return hits[0] if hits else 'NEITHER'


def counts(rows):
    return dict(total=len(rows), correct=sum(r['reference'] == r['prediction'] for r in rows),
                failed=sum(r['prediction'] is None for r in rows),
                by_label={label: dict(
                    reference_count=sum(r['reference'] == label for r in rows),
                    correct=sum(r['reference'] == r['prediction'] == label for r in rows),
                    false_positive=sum(r['prediction'] == label and r['reference'] != label for r in rows)
                ) for label in LABELS})


def evaluate(reference, before, after):
    cases = reference['cases']
    if set(cases) != set(before) or set(cases) != set(after):
        raise ValueError('Reference and both predictions must cover the same windows')
    variants = {}
    all_rows = {}
    structures = []
    for name, predictions in [('before', before), ('after', after)]:
        rows = []
        checks = []
        for number, case in cases.items():
            seen = set()
            for event in case['events']:
                uid = event['candidate_id']
                if uid in seen or event['label'] not in LABELS:
                    raise ValueError('Invalid reference position: ' + uid)
                seen.add(uid)
                rows.append(dict(number=number, candidate_id=uid, reference=event['label'],
                                 prediction=label_at(predictions[number], uid),
                                 function=event.get('function'),
                                 reference_quality_flag=case.get('reference_quality_flag')))
            for check in case.get('structure_checks', []):
                window = predictions[number]
                groups = [[e for e in window.get('events', [])
                           if uid in e['candidate_ids'] and e['label'] == check['label']]
                          for uid in (check['parent'], check['child'])]
                valid = window.get('status') == 'ok' and all(len(g) == 1 for g in groups)
                if valid:
                    same = groups[0][0]['candidate_ids'][0] == groups[1][0]['candidate_ids'][0]
                    valid = not same if check['kind'] == 'separate_feedback' else same
                checks.append(dict(variant=name, number=number, correct=bool(valid), **check))
        variants[name] = dict(all=counts(rows),
                             ordinary_response=counts([r for r in rows if r['function'] == 'ordinary_response']),
                             clearer_references=counts([r for r in rows if not r['reference_quality_flag']]),
                             structures={kind: dict(total=sum(c['kind'] == kind for c in checks),
                                                    correct=sum(c['kind'] == kind and c['correct'] for c in checks))
                                         for kind in sorted({c['kind'] for c in checks})})
        all_rows[name] = {(r['number'], r['candidate_id']): r for r in rows}
        structures += checks
    fixed = []
    regressed = []
    for key, row in all_rows['after'].items():
        was_correct = all_rows['before'][key]['reference'] == all_rows['before'][key]['prediction']
        now_correct = row['reference'] == row['prediction']
        if now_correct and not was_correct:
            fixed.append(row)
        if was_correct and not now_correct:
            regressed.append(row)
    return dict(metrics=variants, fixed=fixed, regressed=regressed,
                positions=list(all_rows['after'].values()), structure_checks=structures,
                interpretation='Agreement with supplied locked reference; not independently established human accuracy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('reference', 'lock', 'before', 'after', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    reference_bytes = args.reference.read_bytes()
    lock = json.loads(args.lock.read_text(encoding='utf8'))
    if hashlib.sha256(reference_bytes).hexdigest() != lock['sha256']:
        raise ValueError('Reference has changed since locking')
    result = evaluate(json.loads(reference_bytes),
                      json.loads(args.before.read_text(encoding='utf8')),
                      json.loads(args.after.read_text(encoding='utf8')))
    result['reference_sha256'] = lock['sha256']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
