import argparse
import csv
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P, T = "PAUSE", "TURN_TRANSITION"
LABELS = (P, T)
BASE_RULES = """Classify only the marked gap between adjacent words of the same focus speaker.
Use transcript as evidence. Choose PAUSE for continuation of the same communicative
action (completion, repair, restatement, supporting explanation, or continued
feedback). Choose TURN_TRANSITION if that action has finished, been yielded, or
clearly dropped and RIGHT begins a distinct action. The same speaker can start
a separate action on the same topic. A different speaker is not required.
Brief feedback, punctuation and gap duration are not sufficient. Interpret any
other speaker's words using their timing and communicative function."""
PROMPT = BASE_RULES + """
Compare the purpose immediately before LEFT with the earliest substantive action
at RIGHT in one decision. Ask what specific part RIGHT completes, repairs, or
supports. A repeated greeting or a repeated question can continue the same action.
A complete sentence followed by its reason or example can continue the same action.
An answer, evaluation, or question with a different purpose can be a distinct action,
including after abandoning a fragment. A new entity alone is insufficient. Treat
fillers as fillers and distinguish any later change from the target boundary.
Choose CONTINUATION, INDEPENDENT, or UNCLEAR for the relationship. UNCLEAR signals
evidence quality only; always provide the better-supported binary label. Return
short literal substrings of the supplied focus_before_target and focus_after_target
as left_evidence and right_evidence, then one brief reason and the final label."""
VERIFY = BASE_RULES + """
Independently verify the boundary by weighing both interpretations. For PAUSE,
name the actual assertion or action RIGHT repairs, completes, restates or supports.
For TURN_TRANSITION, identify how the earlier action closed or was abandoned and
what new action starts at RIGHT. Be careful not to invent omitted words or an
unheard speaker response. Use the better-supported interpretation. Report
CONTINUATION, INDEPENDENT, or UNCLEAR, short literal evidence on each side,
one brief reason, and the binary label. UNCLEAR only records evidence quality."""
PROPERTIES = {
    "relation": {"type": "string", "enum": ["CONTINUATION", "INDEPENDENT", "UNCLEAR"]},
    "left_evidence": {"type": "string"},
    "right_evidence": {"type": "string"},
    "reason": {"type": "string"},
    "label": {"type": "string", "enum": list(LABELS)},
}


def load(name):
    return json.loads((ROOT / name).read_bytes())


def inputs(batch):
    name = "blind100" if batch == "dev" else "risk_blind"
    raw = (ROOT / (name + "_inputs.json")).read_bytes()
    examples = json.loads(raw)
    size = 100 if batch == "dev" else 25
    if len(examples) != size or len({x["id"] for x in examples}) != size:
        raise ValueError("Unexpected sample count or duplicate IDs")
    if batch == "blind":
        excluded = {
            x["conversation_id"] for prior in ("inputs.json", "holdout_inputs.json", "blind100_inputs.json")
            for x in load(prior)
        }
        if excluded.intersection(x["conversation_id"] for x in examples):
            raise ValueError("Previously used recording in blind set")
    return name, examples, hashlib.sha256(raw).hexdigest()


def lock_blind():
    name, examples, sha = inputs("blind")
    labels = {}
    for line in (ROOT / "risk_blind_judgments.txt").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        index, label, strength, reason = line.split("|", 3)
        index = int(index)
        if index in labels or label not in ("P", "T") or strength not in ("C", "W") or not reason.strip():
            raise ValueError(f"Invalid reference row {index}")
        labels[index] = dict(label=P if label == "P" else T,
                             evidence_strength="clear" if strength == "C" else "weak", reason=reason.strip())
    if set(labels) != set(range(1, 26)):
        raise ValueError("Expected 25 reference rows")
    references = dict(source="Codex transcript/context judgment locked before blind inference",
                      inputs_sha256=sha, labels={x["id"]: labels[i] for i, x in enumerate(examples, 1)})
    with (ROOT / (name + "_references.json")).open("x", encoding="utf-8") as out:
        json.dump(references, out, ensure_ascii=False, indent=2)
    print(references["inputs_sha256"])


def view():
    _, examples, _ = inputs("blind")
    for index, item in enumerate(examples, 1):
        p = item["payload"]
        low, high = p["target"]["absolute_gap"]
        print(f'{index:02} [{high-low:.2f}s] {p["focus_before_target"]} >>> {p["focus_after_target"]}')
        for s in p["context"]:
            if s["speaker"] != p["focus"]:
                print(f' {s["speaker"]} {s["start"]-low:+.2f}:{s["end"]-low:+.2f} {s["text"]}')


def request(payload, prompt):
    body = {
        "model": "qwen38-27b",
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": json.dumps(payload)}],
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "boundary",
            "strict": True,
            "schema": {"type": "object", "properties": PROPERTIES,
                       "required": list(PROPERTIES), "additionalProperties": False},
        }},
    }
    encoded = json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:18770/v1/chat/completions", data=encoded,
                                 headers={"Content-Type": "application/json"})
    attempts = []
    for _ in range(2):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.load(response)
            choice = result["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError(f"finish_reason={choice['finish_reason']}")
            output = json.loads(choice["message"]["content"])
            if (set(output) != set(PROPERTIES) or output["label"] not in LABELS
                    or output["relation"] not in PROPERTIES["relation"]["enum"]):
                raise ValueError("Invalid decision fields")
            return dict(status="ok", **output, seconds=round(time.monotonic() - started, 3),
                        attempts=len(attempts)+1)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            attempts.append(str(exc))
    return dict(status="error", errors=attempts)


def words(value):
    return re.findall(r"\w+", value.casefold())


def grounded(payload, decision):
    for field, source in (("left_evidence", "focus_before_target"), ("right_evidence", "focus_after_target")):
        excerpt = " ".join(words(decision.get(field, "")))
        if not excerpt or f" {excerpt} " not in " " + " ".join(words(payload[source])) + " ":
            return False
    return True


def expected(relation):
    return {"CONTINUATION": P, "INDEPENDENT": T}.get(relation)


def risk(payload, decision):
    if decision["status"] != "ok":
        return {"invalid_output": 8}
    flags = {}
    relation = expected(decision["relation"])
    if relation is None:
        flags["unclear_relation"] = 3
    elif relation != decision["label"]:
        flags["relation_label_conflict"] = 4
    if not grounded(payload, decision):
        flags["ungrounded_excerpt"] = 3
    low, high = payload["target"]["absolute_gap"]
    if any(s["speaker"] == "UNKNOWN" and s["start"] <= high+1.5 and s["end"] >= low-1.5
           for s in payload["context"]):
        flags["unattributed_near_gap"] = 2
    if min(len(words(payload["focus_before_target"])), len(words(payload["focus_after_target"]))) <= 5:
        flags["short_context_side"] = 1
    return flags


def shorter(payload):
    low, high = payload["target"]["absolute_gap"]
    result = dict(payload)
    for field in ("context", "other_near_target"):
        result[field] = [s for s in payload[field] if s["start"] <= high+6 and s["end"] >= low-6]
    result["focus_before_target"] = " ".join(payload["focus_before_target"].split()[-10:])
    result["focus_after_target"] = " ".join(payload["focus_after_target"].split()[:10])
    return result


def run(batch):
    from run_ab import evaluate as baseline
    from run_holdout import wait_for_model

    name, examples, sha = inputs(batch)
    refs = load(name + "_references.json")
    if refs["inputs_sha256"] != sha or set(refs["labels"]) != {x["id"] for x in examples}:
        raise ValueError("Reference lock does not match inputs")
    cached = {}
    if batch == "dev":
        cached = {x["id"]: x["baseline"] for x in map(json.loads, (ROOT / "blind100_predictions.jsonl").read_text().splitlines())}
    wait_for_model()
    payloads = {x["id"]: x["payload"] for x in examples}

    def first(item):
        old = cached.get(item["id"]) or baseline(item, False)
        decision = request(item["payload"], PROMPT)
        return dict(id=item["id"], baseline=old.get("label"), first=decision, risk=risk(item["payload"], decision))

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = []
        for number, row in enumerate(pool.map(first, examples), 1):
            rows.append(row)
            print("first", number, flush=True)
        ranked = sorted(rows, key=lambda row: (-sum(row["risk"].values()), row["id"]))
        for row in rows:
            row["final"] = row["first"].get("label")

        def review(row):
            p = payloads[row["id"]]
            short = request(shorter(p), PROMPT)
            verify = request(p, VERIFY)
            row.update(short=short, verify=verify)
            candidate = verify.get("label")
            if (candidate in LABELS and candidate == short.get("label")
                    and expected(verify.get("relation")) == candidate
                    and expected(short.get("relation")) == candidate
                    and grounded(p, verify) and grounded(shorter(p), short)):
                row["final"] = candidate
            return row["id"]

        for number, _ in enumerate(pool.map(review, ranked[:len(rows)//5]), 1):
            print("review", number, flush=True)
    with (ROOT / f"risk_{batch}_predictions.json").open("x", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)
    print("COMPLETE", flush=True)


def score(batch):
    name, examples, sha = inputs(batch)
    refs = load(name + "_references.json")
    predictions = load(f"risk_{batch}_predictions.json")
    by_id = {row["id"]: row for row in predictions}
    if (refs["inputs_sha256"] != sha or len(predictions) != len(examples)
            or set(by_id) != set(refs["labels"])):
        raise ValueError("Prediction or reference coverage mismatch")
    rows = []
    for index, item in enumerate(examples, 1):
        key = item["id"]
        result = by_id[key]
        ref = refs["labels"][key]
        rows.append(dict(index=index, id=key, reference=ref["label"], strength=ref["evidence_strength"],
                         baseline=result["baseline"], revised=result["first"].get("label"), routed=result["final"],
                         risk_score=sum(result["risk"].values()), risk_flags="|".join(result["risk"]),
                         review="verify" in result, short=result.get("short", {}).get("label"),
                         verifier=result.get("verify", {}).get("label"), reference_reason=ref["reason"],
                         revised_reason=result["first"].get("reason")))

    def metrics(subset, method):
        return dict(n=len(subset), valid=sum(x[method] in LABELS for x in subset),
                    matches=sum(x[method] == x["reference"] for x in subset),
                    transition_hits=sum(x[method] == x["reference"] == T for x in subset),
                    false_transitions=sum(x[method] == T and x["reference"] == P for x in subset))

    methods = ("baseline", "revised", "routed")
    reviewed = [x for x in rows if x["review"]]
    errors = [x for x in rows if x["revised"] != x["reference"]]
    flagged = [x for x in errors if x["review"]]
    flipped = [x for x in reviewed if x["short"] in LABELS and x["short"] != x["revised"]]
    flag_counts = {}
    for flag in sorted({f for row in predictions for f in row["risk"]}):
        subset = [row for row in rows if flag in row["risk_flags"].split("|")]
        flag_counts[flag] = dict(n=len(subset), errors=sum(x["revised"] != x["reference"] for x in subset))
    report = dict(
        references={label: sum(x["reference"] == label for x in rows) for label in LABELS},
        methods={m: metrics(rows, m) for m in methods},
        clear={m: metrics([x for x in rows if x["strength"] == "clear"], m) for m in methods},
        risk=dict(reviewed=len(reviewed), errors=len(errors), caught=len(flagged),
                  window_flips=len(flipped), window_flip_errors=sum(x in errors for x in flipped),
                  changed=sum(x["routed"] != x["revised"] for x in reviewed),
                  fixed=sum(x["routed"] == x["reference"] for x in flagged),
                  harmed=sum(x["routed"] != x["reference"] and x not in errors for x in reviewed),
                  flag_counts=flag_counts),
    )
    with (ROOT / f"risk_{batch}_comparison.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    (ROOT / f"risk_{batch}_summary.json").write_text(formatted + "\n", encoding="utf-8")
    print(formatted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("view", "lock", "run", "score"))
    parser.add_argument("batch", choices=("dev", "blind"), nargs="?", default="blind")
    args = parser.parse_args()
    if args.command in ("view", "lock"):
        {"view": view, "lock": lock_blind}[args.command]()
    else:
        {"run": run, "score": score}[args.command](args.batch)
