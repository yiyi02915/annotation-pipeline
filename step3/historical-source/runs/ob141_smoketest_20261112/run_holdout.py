import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_ab import evaluate as baseline
from run_staged import ENDPOINT, evaluate_stage

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_bytes())


def wait_for_model():
    deadline = time.monotonic() + 1200
    while True:
        try:
            with urllib.request.urlopen(ENDPOINT + '/models', timeout=3) as response:
                if 'qwen38-27b' not in {item['id'] for item in json.load(response)['data']}:
                    raise RuntimeError('Unexpected model')
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError('Qwen service did not start')
            time.sleep(4)


def label(first, second):
    if first['status'] != 'ok' or second['status'] != 'ok':
        return None
    return 'TURN_TRANSITION' if first['prior_ended'] and second['next_independent'] else 'PAUSE'


def evaluate(example):
    payload = example['payload']
    direct = baseline(example, False)
    original = evaluate_stage(payload, 'prior_ended')
    revised = evaluate_stage(payload, 'closure')
    # Stage two has the same input and prompt for both versions.
    second = evaluate_stage(payload, 'next_independent')
    return {
        'id': example['id'],
        'baseline': direct,
        'original_first': original,
        'revised_first': revised,
        'shared_second': second,
        'original_label': label(original, second),
        'revised_label': label(revised, second),
    }


def smoke(example):
    result = evaluate_stage(example['payload'], 'closure')
    return {'id': example['id'], 'revised_first': result}


def review(example, stage='dependency_review'):
    second = evaluate_stage(example['payload'], stage)
    return {'id': example['id'], 'review_second': second}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dev', action='store_true')
    mode.add_argument('--review', action='store_true')
    mode.add_argument('--relation', action='store_true')
    args = parser.parse_args()
    original = load('inputs.json')
    if args.dev:
        examples = [original[i - 1] for i in (3, 13, 14)]
    else:
        raw = (ROOT / 'holdout_inputs.json').read_bytes()
        examples = json.loads(raw)
        reference = load('holdout_references.json')
        if (
            len(examples) != 20
            or len({item['id'] for item in examples}) != 20
            or {item['id'] for item in examples} != set(reference['labels'])
            or hashlib.sha256(raw).hexdigest() != reference['inputs_sha256']
            or {item['conversation_id'] for item in examples}
            & {item['conversation_id'] for item in original}
        ):
            raise ValueError('Holdout does not match its lock or shares an original recording')
    wait_for_model()
    started = time.monotonic()
    name, judge = (
        ('closure_dev', smoke) if args.dev else
        ('holdout_relation', lambda example: review(example, 'relation')) if args.relation else
        ('holdout_review', review) if args.review else
        ('holdout_predictions', evaluate)
    )
    with (ROOT / (name + '.jsonl')).open('w', encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(judge, example) for example in examples]
            for job in as_completed(jobs):
                result = job.result()
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(result['id'], flush=True)
    print('wall_seconds', round(time.monotonic() - started, 3), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
