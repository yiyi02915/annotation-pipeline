import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAUSE = 'PAUSE'
TRANSITION = 'TURN_TRANSITION'
LABELS = (PAUSE, TRANSITION)


def load(name):
    return json.loads((ROOT / name).read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def examples_and_digest():
    raw = (ROOT / 'blind100_inputs.json').read_bytes()
    examples = json.loads(raw)
    excluded = {
        row['conversation_id']
        for filename in ('inputs.json', 'holdout_inputs.json')
        for row in load(filename)
    }
    recordings = Counter(row['conversation_id'] for row in examples)
    if (len(examples) != 100 or len({row['id'] for row in examples}) != 100
            or excluded.intersection(recordings) or max(recordings.values()) > 2
            or len({row['clip'] for row in examples}) != 100):
        raise ValueError('Unexpected sample size, recording overlap, or repeated window')
    return examples, digest(raw)


def view(start, end):
    examples, _ = examples_and_digest()
    for index, row in enumerate(examples, 1):
        if not start <= index <= end:
            continue
        payload = row['payload']
        low, high = payload['target']['absolute_gap']
        print(f'{index:03} [{high-low:.2f}s] {payload["focus_before_target"]} >>> {payload["focus_after_target"]}')
        for item in payload['context']:
            speaker = 'F' if item['speaker'] == payload['focus'] else '?' if item['speaker'] == 'UNKNOWN' else 'O'
            if speaker != 'F':
                print(f' {speaker} {item["start"]-low:+.2f}:{item["end"]-low:+.2f} {item["text"]}')


def lock():
    examples, input_hash = examples_and_digest()
    raw = (ROOT / 'blind100_judgments.txt').read_bytes()
    judgments = {}
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        ordinal, label, evidence, reason = line.split('|', 3)
        index = int(ordinal)
        if index in judgments or label not in ('P', 'T') or evidence not in ('C', 'W') or not reason.strip():
            raise ValueError(f'Invalid or duplicate judgment {ordinal}')
        judgments[index] = {
            'label': PAUSE if label == 'P' else TRANSITION,
            'evidence_strength': 'clear' if evidence == 'C' else 'weak',
            'reason': reason.strip(),
        }
    if set(judgments) != set(range(1, 101)):
        raise ValueError('Expected one judgment for each of the 100 ordinals')
    lock = {
        'source': 'Codex judgment using attributed transcript and context before inference',
        'inputs_sha256': input_hash,
        'judgments_sha256': digest(raw),
        'labels': {row['id']: judgments[index] for index, row in enumerate(examples, 1)},
    }
    with (ROOT / 'blind100_references.json').open('x', encoding='utf-8') as stream:
        json.dump(lock, stream, ensure_ascii=False, indent=2)
    print(Counter((row['label'], row['evidence_strength']) for row in judgments.values()))


def locked_examples():
    examples, input_hash = examples_and_digest()
    reference = load('blind100_references.json')
    if (reference['inputs_sha256'] != input_hash
            or set(reference['labels']) != {row['id'] for row in examples}
            or any(row['label'] not in LABELS for row in reference['labels'].values())):
        raise ValueError('Inputs and reference lock do not match')
    return examples, reference


def evaluate(example):
    from run_ab import evaluate as direct
    from run_staged import evaluate_stage
    from run_holdout import label

    baseline = direct(example, False)
    closure = evaluate_stage(example['payload'], 'closure')
    relation = evaluate_stage(example['payload'], 'relation')
    return {
        'id': example['id'], 'baseline': baseline, 'closure': closure, 'relation': relation,
        'staged': label(closure, relation),
    }


def run():
    from run_holdout import wait_for_model

    examples, _ = locked_examples()
    wait_for_model()
    started = time.monotonic()
    with (ROOT / 'blind100_predictions.jsonl').open('x', encoding='utf-8') as output:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(evaluate, example) for example in examples]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                output.write(json.dumps(result) + '\n')
                output.flush()
                print(count, result['id'], result['baseline'].get('label'), result['staged'], flush=True)
    print('wall_seconds', round(time.monotonic() - started, 3), flush=True)
    print('COMPLETE', flush=True)


def score():
    examples, reference = locked_examples()
    raw_predictions = [
        json.loads(line)
        for line in (ROOT / 'blind100_predictions.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    predictions = {row['id']: row for row in raw_predictions}
    if len(raw_predictions) != 100 or set(predictions) != set(reference['labels']):
        raise ValueError('Predictions do not cover the locked sample exactly once')
    rows = []
    for index, example in enumerate(examples, 1):
        key = example['id']
        ref = reference['labels'][key]
        pred = predictions[key]
        rows.append({
            'index': index,
            'id': key,
            'conversation_id': example['conversation_id'],
            'gap_seconds': example['payload']['target']['gap_seconds'],
            'reference': ref['label'],
            'reference_strength': ref['evidence_strength'],
            'reference_reason': ref['reason'],
            'baseline': pred['baseline'].get('label'),
            'staged': pred['staged'],
            'closure': pred['closure'].get('closure'),
            'relation': pred['relation'].get('relation'),
            'closure_reason': pred['closure'].get('reason'),
            'relation_reason': pred['relation'].get('reason'),
            'before': example['payload']['focus_before_target'],
            'after': example['payload']['focus_after_target'],
        })

    def metrics(items, method):
        return {
            'valid': sum(row[method] in LABELS for row in items),
            'matches': sum(row[method] == row['reference'] for row in items),
            'transition_correct': sum(row[method] == row['reference'] == TRANSITION for row in items),
            'pause_false_transition': sum(row[method] == TRANSITION and row['reference'] == PAUSE for row in items),
        }

    clear = [row for row in rows if row['reference_strength'] == 'clear']
    methods = ('baseline', 'staged')
    report = {
        'reference_counts': dict(Counter(row['reference'] for row in rows)),
        'clear_reference_counts': dict(Counter(row['reference'] for row in clear)),
        'methods': {method: metrics(rows, method) for method in methods},
        'clear_subset': {method: metrics(clear, method) for method in methods},
        'paired': {
            'new_transition_hits': [row['index'] for row in rows if row['staged'] == row['reference'] == TRANSITION and row['baseline'] != TRANSITION],
            'lost_transition_hits': [row['index'] for row in rows if row['baseline'] == row['reference'] == TRANSITION and row['staged'] != TRANSITION],
            'new_false_transitions': [row['index'] for row in rows if row['reference'] == PAUSE and row['staged'] == TRANSITION and row['baseline'] != TRANSITION],
            'removed_false_transitions': [row['index'] for row in rows if row['reference'] == PAUSE and row['baseline'] == TRANSITION and row['staged'] != TRANSITION],
        },
    }
    with (ROOT / 'blind100_comparison.csv').open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    (ROOT / 'blind100_summary.json').write_text(formatted + '\n', encoding='utf-8')
    print(formatted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['view', 'lock', 'run', 'score'])
    parser.add_argument('start', type=int, nargs='?', default=1)
    parser.add_argument('end', type=int, nargs='?', default=100)
    args = parser.parse_args()
    if args.command == 'view':
        view(args.start, args.end)
    else:
        {'lock': lock, 'run': run, 'score': score}[args.command]()
