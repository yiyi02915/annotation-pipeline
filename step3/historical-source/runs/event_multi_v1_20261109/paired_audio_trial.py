"""Use a matched window and the same questions for isolated and joint audio."""
import argparse
import json
import time
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[1]
OUT = D / "paired_audio_trial"
CHOICES = ["BACKCHANNEL", "INTERRUPT", "NEITHER"]
SPEAKER_MODEL = "/data/L202500721/stevezhuang/annotation_pipeline/models/v3/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo"


def read(p):
    return json.loads(p.read_bytes())


def save(p, value):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def inputs():
    cases = read(D / "word_inputs.json")
    jobs_path = D / "extra_audio_jobs.json"
    if not jobs_path.exists():
        jobs_path = D.parent / jobs_path.name
    jobs = {x["number"]: x for x in read(jobs_path) if not x["gap"]}
    return cases, jobs


def prepare(cases, jobs):
    from paired_audio import render_pair

    out = {}
    for case in cases:
        num = case["number"]
        job = jobs[num]
        if (set(job["paths"]) != set(case["streams"]) or job["interval"] != case["interval"]
                or job["target"] != case["target"]):
            raise ValueError(f"Audio and FA window metadata differ: {num}")
        out[num] = render_pair(job["paths"], case["conversation_id"], case["interval"], OUT / num)
    save(OUT / "manifest.json", out)
    print("PAIRS_READY", len(out), flush=True)


def diarize(cases, jobs, tracks):
    from collections import defaultdict
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(SPEAKER_MODEL, map_location="cuda")
    model.eval()
    for key, value in dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300).items():
        setattr(model.sortformer_modules, key, value)
    out = {}
    for case in cases:
        num = case["number"]
        pair = tracks[num]
        views = {**pair["isolated"], "mix": pair["files"]["mix"]}
        out[num] = {}
        for view, path in views.items():
            results = model.diarize(audio=[path], batch_size=1)
            if isinstance(results, tuple):
                results = results[0]
            segments = [(float(a), float(b), s) for a, b, s in map(str.split, results[0])]
            activity = defaultdict(float)
            for a, b, s in segments:
                activity[s] += b-a
            out[num][view] = dict(segments=segments, activity_seconds=dict(activity), count=len(activity))
        save(OUT / "diarization.json", out)
        print("DIAR", num, {k: v["count"] for k, v in out[num].items()}, flush=True)


def judge(cases, jobs, tracks, size, *, history_file=None, on_clip=None):
    from duplexjev import Decider, Question
    from event_extractor import extract_candidates

    started = time.monotonic()
    engine = Decider.from_pretrained(str(R / "models" / f"DuplexJev-{size}-Para"),
                                     device="cuda:0", attn_implementation="sdpa")
    print("JEV_READY", round(time.monotonic() - started, 2), flush=True)
    event_q = Question("event", "At the specified onset, is the target speaker giving supportive "
                       "feedback without taking the floor (BACKCHANNEL), trying to claim/change "
                       "the floor before the other finishes (INTERRUPT), or neither? "
                       "Track who is speaking before onset. A later overlap alone is neither.",
                       CHOICES, audio="*")
    history = {r["clip_id"]: r for r in read(history_file) if r["status"] == "ok"} if history_file else {}
    out = {}
    for case in cases:
        num = case["number"]
        pair = tracks[num]
        focus = jobs[num]["focus"]
        row = dict(clip_id=case["id"], participant_count=2, events=[])
        candidates = extract_candidates(case)
        old = history.get(case["id"], {}).get("events", [])
        # Cover every prior retained onset, the original target, and all plausible
        # fresh entries so a prior text model DROP cannot hide a mixed-audio change.
        prior_ids = {event["candidate_ids"][0] for event in old}
        target_id = case["id"].split(":overlap:", 1)[-1]
        trial = [c for c in candidates if c["candidate_id"] in prior_ids or
                 (c["initiator"] == focus and c["word_ids"][0] == target_id) or
                 (c["other_active_run_at_onset"] and c["overlap_seconds"] > 0)]
        for c in trial:
            a, b = (c[key] - pair["interval"][0] for key in ("start", "end"))
            relevant = []
            for speaker, words in case["streams"].items():
                nearby = [w for w in words if c["start"]-3.0 <= w["start"] <= c["start"]+3.0]
                relevant.append(f'{speaker}: ' + " ".join(w["text"] for w in nearby))
            context = (f"Target: {c['initiator']} speaking to {c['recipient']}; target starts at "
                       f"{a:.3f}s relative to clip, ends at {min(b, pair['frames']/pair['sample_rate']):.3f}s; "
                       f"target words: {c['text']}. Nearby transcript:\n" + "\n".join(relevant))
            began = time.monotonic()
            replies = engine.decide_batch({"isolated": pair["isolated"][c["initiator"]],
                                          "mix": pair["files"]["mix"]}, [event_q],
                                          context={"isolated": context, "mix": context}, lang="en",
                                          mode="packed", max_items=2, n_perm=4, seed=0)
            prior = next((e["label"] for e in old if e["candidate_ids"][0] == c["candidate_id"]), None)
            row["events"].append(dict(candidate_id=c["candidate_id"], initiator=c["initiator"],
                                      recipient=c["recipient"], source_start=c["start"], relative_start=a,
                                      is_original_target=c["initiator"] == focus and c["word_ids"][0] == target_id,
                                      prior_event_label=prior, text=c["text"],
                                      isolated=replies["isolated"]["event"], mix=replies["mix"]["event"],
                                      elapsed_seconds=time.monotonic()-began))
        out[num] = row
        if on_clip is None:
            save(OUT / "audio_comparison.json", out)
        else:
            on_clip(case, row)
        print("JUDGED", num, len(row["events"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "diarize", "judge"])
    parser.add_argument("--size", choices=["4B", "32B"], default="32B")
    args = parser.parse_args()
    cases, jobs = inputs()
    if args.stage == "prepare":
        prepare(cases, jobs)
    elif args.stage == "diarize":
        diarize(cases, jobs, read(OUT / "manifest.json"))
    else:
        judge(cases, jobs, read(OUT / "manifest.json"), args.size, history_file=D / "results.json")


if __name__ == "__main__":
    main()
