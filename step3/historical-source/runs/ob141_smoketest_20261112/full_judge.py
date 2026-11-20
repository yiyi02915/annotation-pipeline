import json
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from risk_trial import PROMPT, VERIFY, grounded, request, risk, shorter, expected
from run_holdout import wait_for_model

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "full_annotation"
WORKERS = 4


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def source_key(item):
    target = item["payload"]["target"]
    return item["source_audio"].removeprefix("dataset/"), target["left_word_id"], target["right_word_id"]


def timed(payload, prompt):
    start = time.monotonic()
    result = request(payload, prompt)
    return dict(decision=result, interval=[start, time.monotonic()])


def first(item):
    measured = timed(item["payload"], PROMPT)
    return dict(id=item["id"], key=source_key(item), first=measured, risk=risk(item["payload"], measured["decision"]))


def review(item):
    payload = item["payload"]
    reduced = shorter(payload)
    short = timed(reduced, PROMPT)
    verify = timed(payload, VERIFY)
    a, b = short["decision"], verify["decision"]
    proposal = b.get("label")
    if (not proposal or proposal != a.get("label") or expected(a.get("relation")) != proposal
            or expected(b.get("relation")) != proposal or not grounded(reduced, a) or not grounded(payload, b)):
        proposal = None
    return dict(short=short, verify=verify, proposal=proposal)


def union_seconds(intervals):
    total = 0
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(stop, end)
    return total


def phase(items, worker, pool):
    start = time.monotonic()
    futures = {pool.submit(worker, item): item["id"] for item in items}
    results = {}
    for job in as_completed(futures):
        results[futures[job]] = job.result()
    return results, time.monotonic() - start


def snapshot(predictions, first_wall, both_wall, references):
    valid = [x for x in predictions if x["first"]["decision"]["status"] == "ok"]
    matches = [(x, references[tuple(x["key"])]) for x in valid if tuple(x["key"]) in references]
    return dict(candidates=len(predictions), labeled=len(valid), failed=len(predictions)-len(valid),
                reviewed=sum("review" in x for x in predictions),
                first_wall_seconds=first_wall, with_review_wall_seconds=both_wall,
                matched_reference=len(matches),
                reference_agreement=sum(x["final"] == ref for x, ref in matches),
                reference_total=len(references))


def benchmark(examples, review_ids, pool):
    target = RUN / "concurrency_benchmark.json"
    if target.exists():
        return
    # Preserve stable input order, grouping by recording before assigning the four workers.
    items = sorted(examples, key=lambda x: (x["source_audio"], x["payload"]["target"]["absolute_gap"]))
    print("BENCHMARK_START", len(items), flush=True)
    started = time.monotonic()
    first_results, first_wall = phase(items, first, pool)
    second, review_wall = phase([item for item in items if item["id"] in review_ids], review, pool)
    combined_wall = time.monotonic() - started
    durations = subprocess.check_output(
        ["/data/L202500293/envs/fdb3/bin/python", "-c",
         "import json,soundfile as sf,sys; print(json.dumps({x:sf.info(x).duration for x in json.load(sys.stdin)}))"],
        input=json.dumps(sorted({x["source_audio"] for x in items})), text=True,
    )
    durations = json.loads(durations)
    group = defaultdict(list)
    for item in items:
        group[item["source_audio"]].append(item["id"])
    recordings = {}
    for path, ids in group.items():
        primary = [first_results[key]["first"]["interval"] for key in ids]
        extra = [r[field]["interval"] for key in ids if (r := second.get(key)) for field in ("short", "verify")]
        recordings[path.removeprefix("dataset/")] = dict(
            duration_seconds=durations[path], count=len(ids),
            first_active_seconds=union_seconds(primary),
            with_review_active_seconds=union_seconds(primary + extra),
            reviewed=sum(key in second for key in ids),
        )
    valid = sum(x["first"]["decision"]["status"] == "ok" for x in first_results.values())
    successful_reviews = sum(all(row[k]["decision"]["status"] == "ok" for k in ("short", "verify")) for row in second.values())
    save(target, dict(workers=WORKERS, count=len(items), valid=valid, reviewed=len(second),
                      successful_reviews=successful_reviews, first_wall_seconds=first_wall,
                      review_wall_seconds=review_wall, with_review_wall_seconds=combined_wall, recordings=recordings))
    print("BENCHMARK_COMPLETE", valid, round(first_wall, 2), round(review_wall, 2), flush=True)


def main():
    references = {}
    examples = []
    review_ids = set()
    for name, batch in (("blind100", "dev"), ("risk_blind", "blind")):
        inputs = read(ROOT / f"{name}_inputs.json")
        locked = read(ROOT / f"{name}_references.json")["labels"]
        examples.extend(inputs)
        references.update({source_key(x): locked[x["id"]]["label"] for x in inputs})
        review_ids.update(x["id"] for x in read(ROOT / f"risk_{batch}_predictions.json") if "verify" in x)
    assert len(examples) == 125 and len(references) == 125 and len(review_ids) == 25
    output = RUN / "labels"
    output.mkdir(exist_ok=True)
    existing = [read(p) for p in output.glob("*.json")]
    all_rows = [x for batch in existing for x in batch["labels"]]
    seen = {recording for batch in existing for recording in batch["recordings"]}
    first_wall = sum(batch["first_wall_seconds"] for batch in existing)
    both_wall = sum(batch["with_review_wall_seconds"] for batch in existing)
    wait_for_model()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while True:
            candidates = [p for p in sorted((RUN/"candidates").glob("*.json")) if p.stem not in seen]
            if not candidates:
                if (RUN/"extraction_complete.json").exists():
                    benchmark(examples, review_ids, pool)
                    break
                time.sleep(12)
                continue
            paths = []
            items = []
            limit = 20 if not all_rows else 100
            for path in candidates:
                paths.append(path)
                items.extend(read(path)["candidates"])
                if len(items) >= limit or len(paths) >= 12:
                    break
            print("FULL_BATCH_START", len(paths), len(items), flush=True)
            began = time.monotonic()
            measured, wall1 = phase(items, first, pool)
            sorted_rows = sorted(measured.values(), key=lambda x: (-sum(x["risk"].values()), x["id"]))
            picked = {row["id"] for row in sorted_rows[:len(items)//5]}
            extra, wall2 = phase([item for item in items if item["id"] in picked], review, pool)
            combined_wall = time.monotonic() - began
            for row in measured.values():
                if row["id"] in extra:
                    row["review"] = extra[row["id"]]
                # The evaluated review policy did not improve accuracy. Keep its proposal for audit.
                row["final"] = row["first"]["decision"].get("label")
            stems = [p.stem for p in paths]
            save(output/f"batch_{len(list(output.glob('batch_*.json')))+1:05}.json",
                 dict(recordings=stems, labels=list(measured.values()), first_wall_seconds=wall1,
                      with_review_wall_seconds=combined_wall))
            seen.update(stems)
            all_rows.extend(measured.values())
            first_wall += wall1
            both_wall += combined_wall
            status = snapshot(all_rows, first_wall, both_wall, references)
            status.update(recordings=len(seen), review_policy="shadow_only", workers=WORKERS)
            save(RUN/"annotation_status.json", status)
            print("FULL_BATCH_COMPLETE", json.dumps(status), flush=True)
            if all_rows:
                benchmark(examples, review_ids, pool)
    save(RUN/"annotation_complete.json", snapshot(all_rows, first_wall, both_wall, references))


if __name__ == "__main__":
    main()
