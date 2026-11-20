"""Compare new Step 3 labels with the previously frozen source references."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def main():
    ref = read(ROOT / "benchmark_reference.json")
    run = ROOT / "run"
    outputs = {r["conversation_id"]: r for r in map(read, (run / "recordings").glob("*.json"))}
    if set(outputs) != set(ref["recordings"]):
        raise ValueError("Run IDs and frozen sample IDs differ")
    overlap = Counter()
    gaps = Counter()
    patterns = Counter()
    details = []
    for cid, entry in ref["recordings"].items():
        result = outputs[cid]
        events = {e["candidate_ids"][0]: e for e in result.get("events", [])}
        windows = result.get("windows", [])
        candidate_ids = {d["candidate_id"] for w in windows for d in w["decisions"]}
        if "events" in entry:
            for target in entry["events"]:
                expected = target["label"]
                if expected == "UNSCORED":
                    continue
                key = target["candidate_id"]
                overlap[expected, "reference"] += 1
                if key not in candidate_ids:
                    overlap[expected, "candidate_unmatched"] += 1
                found = events[key]["label"] if key in events else "NEITHER"
                overlap[expected, "correct"] += found == expected
                overlap[found, "false_positive"] += found != expected
                if target.get("ordinary_entry"):
                    overlap["ordinary_entry", "reference"] += 1
                    overlap["ordinary_entry", "false_positive"] += found != "NEITHER"
                details.append(dict(number=entry["number"], id=key, reference=expected, predicted=found,
                                    candidate_found=key in candidate_ids))
            for p in entry["patterns"]:
                first, second = [events.get(p[i]) for i in ("first", "second")]
                expected = "INTERRUPT" if p["kind"] == "bidirectional" else "BACKCHANNEL"
                good = bool(first and second and first["label"] == "INTERRUPT" and second["label"] == expected)
                if p["kind"] == "nested" and good:
                    good = first["start"] <= second["start"] < first["end"]
                patterns[p["kind"], "reference"] += 1
                patterns[p["kind"], "correct"] += good
        else:
            gap = entry["gap"]
            target = gap["target"]
            label = gap["label"]
            gaps[label, "reference"] += 1
            matches = [g for g in result.get("gaps", [])
                       if all(g["target"][key] == target[key] for key in ("left_word_id", "right_word_id"))
                       and all(abs(a-b) < .002 for a,b in zip(
                           g["target"]["absolute_gap"], target["absolute_gap"]))]
            if len(matches) > 1:
                raise ValueError("Repeated gap match")
            found = matches[0] if matches else {}
            predicted = found.get("label")
            gaps[label, "matched"] += bool(predicted)
            gaps[label, "correct"] += label == predicted
            gaps[label, "reviewed"] += found.get("reviewed", False)
            details.append(dict(number=entry["number"], id=cid, reference=label, predicted=predicted,
                                reference_target=target, detected_gaps=[g["target"] for g in result.get("gaps", [])]))
    groups = dict(overlap=overlap, gaps=gaps, patterns=patterns)
    metric = {group: {label: {k: v for (name, k), v in counter.items() if name == label}
                      for label in sorted({label for label, _ in counter})}
              for group, counter in groups.items()}
    for label in ("INTERRUPT", "BACKCHANNEL"):
        row = metric["overlap"][label]
        row["reference_agreement"] = row["correct"] / row["reference"]
    for row in metric["gaps"].values():
        row["reference_agreement"] = row["correct"] / row["reference"] if row["reference"] else None
        row["paired_agreement"] = row["correct"] / row["matched"] if row["matched"] else None
        row["pair_coverage"] = row["matched"] / row["reference"] if row["reference"] else None
    out = dict(reference_locks=ref["source_locks"], metrics=metric,
               run=read(run/"summary.json"), details=details)
    (ROOT/"comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": metric, "run": out["run"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
