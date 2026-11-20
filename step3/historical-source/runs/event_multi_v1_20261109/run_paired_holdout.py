"""Run both audio conditions on exactly the same fixed, blinded windows."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from paired_audio_trial import judge, read, save
from run_multi import classify


def validate_lock(root):
    ref_path = root / "reference.locked.json"
    digest = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    if digest != (root / "reference.sha256").read_text().strip():
        raise ValueError("Reference lock digest mismatch")
    cases = []
    for name, source_digest in read(ref_path)["sample_sha256"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_digest:
            raise ValueError(f"Sample digest mismatch: {name}")
        cases.extend(read(path)["cases"])
    return cases


def audio(root, cases):
    from paired_audio import render_pair

    views = {}
    for case in cases:
        num = case["number"]
        views[num] = render_pair(case["paths"], case["conversation_id"], case["interval"], root / "audio" / num)
    save(root / "audio_manifest.json", views)
    print("RENDERED", len(views), flush=True)
    done = {p.stem for p in (root / "votes").glob("*.json")}
    cases = [c for c in cases if c["number"] not in done]
    if cases:
        judge(cases, {c["number"]: c for c in cases}, views, "32B",
              on_clip=lambda case, row: save(root / "votes" / (case["number"] + ".json"), row))
    (root / "audio.complete").touch()


def semantic(root, cases, refined=False, floor_checks=False):
    from time import sleep

    def run(case, view, votes, resume_truncated):
        res = classify(dict(case, original_label=None), votes, audio_view=view, concise=True,
                       recover_invalid_merge=True, refined=refined, floor_checks=floor_checks,
                       resume_truncated=resume_truncated)
        # Keep decisions and request timing, without duplicating the model response.
        for attempt in res.get("attempts", []):
            attempt.pop("response", None)
        save(output / f"{case['number']}.{view}.json", res)
        print("FINAL", case["number"], view, res["status"], len(res.get("events", [])), flush=True)
        return res["status"]

    jobs = []
    output = root / ("predictions_floor" if floor_checks else "predictions_refined" if refined else "predictions")
    output.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for case in cases:
            vote_path = root / "votes" / (case["number"] + ".json")
            for _ in range(300):
                if vote_path.exists():
                    break
                sleep(2)
            votes = read(vote_path)
            for view in (("mix",) if refined else ("isolated", "mix")):
                path = output / f"{case['number']}.{view}.json"
                previous = read(path) if path.exists() else {}
                if previous.get("status") != "ok":
                    truncated = any(a.get("error") == "generation_truncated" or a.get("keyed_output")
                                    for a in previous.get("attempts", []))
                    jobs.append(pool.submit(run, case, view, votes, truncated))
        failures = sum(job.result() != "ok" for job in jobs)
    print("EVALUATION_COMPLETE", len(jobs), "failures", failures, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("audio", "semantic", "refined", "floor"))
    args = parser.parse_args()
    root = args.root.resolve()
    cases = validate_lock(root)
    for name in ("votes", "predictions"):
        (root / name).mkdir(exist_ok=True)
    if args.stage == "audio":
        audio(root, cases)
    else:
        semantic(root, cases, refined=args.stage in ("refined", "floor"), floor_checks=args.stage == "floor")


if __name__ == "__main__":
    main()
