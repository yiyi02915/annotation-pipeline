"""Sample new Seamless windows before either evaluation system is run."""
import argparse
import json
import random
import sys
from pathlib import Path

from event_extractor import extract_candidates

D = Path(__file__).resolve().parent
DATA = Path("/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd")
SEED = 20261113


def sample(output, prior, supplement=False, seed=SEED, transfer=False):
    sys.path.insert(0, str(D.parent / "finegrained_pilot_20261109"))
    from build_candidates import context, good, runs

    excluded = set(json.loads(prior.read_bytes()))
    indices = []
    with (DATA / "training/index_v1/train.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["interaction_id"] not in excluded:
                indices.append(row)
    rng = random.Random(seed)
    rng.shuffle(indices)
    wanted = ({"substantive_reverse": 6, "feedback_during_substantive_entry": 4} if supplement else
              {"random_overlap": 12, "reverse_onset": 4, "brief_reverse_onset": 4})
    if transfer:
        wanted = {"substantive_reverse": 12, "feedback_during_substantive_entry": 10,
                  "near_question_end": 8}
    counts = dict.fromkeys(wanted, 0)
    cases = []
    sessions = set()
    with (DATA / "seamless_fd_data/metadata.jsonl").open("rb") as f:
        for row in indices:
            cid = row["interaction_id"]
            session = cid.split("_I")[0]
            if session in sessions or len(row["participants"]) != 2:
                continue
            paths = {
                p["sample_id"].rsplit("_", 1)[1]: p["audio_path"].replace("/hbt_share02/202610/", "/data2/202610/")
                for p in row["participants"]
            }
            if not all(Path(p).is_file() for p in paths.values()):
                continue
            f.seek(row["metadata_byte_offset"])
            meta = json.loads(f.read(row["metadata_bytes"]))
            assert meta["interaction_id"] == cid
            if meta["participant_count"] != 2:
                continue
            full = {}
            duration = min(p["duration_seconds"] for p in row["participants"])
            for p in meta["participants"]:
                sp = p["sample_id"].rsplit("_", 1)[1]
                words = [dict(id=f"{sp}:w{i}", text=w["word"], start=float(w["start_sec"]),
                              end=float(w["end_sec"]), score=float(w.get("score", 1)))
                         for i, w in enumerate(p["words"])]
                full[sp] = sorted((w for w in words if 0 <= w["start"] < w["end"] <= duration+.01),
                                  key=lambda w: (w["start"], w["end"]))
            assert set(full) == set(paths)
            proposals = []
            for sp, words in full.items():
                other = next(k for k in full if k != sp)
                other_runs = runs(full[other])
                for group in runs(words):
                    a, b = group[0]["start"], group[-1]["end"]
                    if not 5 < a < duration-7 or b-a > 5 or not all(good(w) for w in group):
                        continue
                    if not any(w["start"] < a-.04 and w["end"] > a+.04 and good(w) for w in full[other]):
                        continue
                    reverse = [r for r in other_runs if a+.15 < r[0]["start"] < min(b-.04, a+3.5)
                               and any(w["start"] < r[0]["start"]-.03 < w["end"]-.05 for w in group)]
                    category = ("brief_reverse_onset" if any(len(r) <= 3 for r in reverse)
                                else "reverse_onset" if reverse else None)
                    if transfer:
                        terminal_question = any(w['text'].rstrip().endswith('?') and 0 < w['end']-a < .28
                                                for w in full[other])
                        options = []
                        if any(len(r) >= 4 for r in reverse) and len(group) >= 4:
                            options.append("substantive_reverse")
                        if len(group) >= 4 and any(len(r) <= 3 and r[0]["text"].lower().strip(".,!?") in
                            {"yes", "yeah", "right", "okay", "oh", "uh-huh", "mm-hmm", "exactly"} for r in reverse):
                            options.append("feedback_during_substantive_entry")
                        if terminal_question:
                            options.append("near_question_end")
                        proposals.extend((a, b, sp, group[0]["id"], k) for k in options)
                        continue
                    if supplement:
                        if len(group) < 4 or b-a < 1 or not any(
                                w["start"] > a+.16 for w in full[other]
                                if a-.1 < w["start"] < a+1.1):
                            continue
                        if any(len(r) >= 5 for r in reverse):
                            category = "substantive_reverse"
                        elif any(len(r) <= 2 and r[0]["text"].lower().strip(".,!?") in
                                 {"yeah", "right", "okay", "oh", "uh-huh", "mm-hmm", "exactly"} for r in reverse):
                            category = "feedback_during_substantive_entry"
                        else:
                            continue
                    proposals.append((a, b, sp, group[0]["id"], category))
            rng.shuffle(proposals)
            found = False
            for a, b, focus, word_id, category in proposals:
                strata = ([category] if category else []) + ["random_overlap"]
                stratum = next((k for k in strata if k in counts and counts[k] < wanted[k]), None)
                if stratum is None:
                    continue
                # Core onsets are scored. Pre-roll and tail supply the preceding floor and reaction.
                core = [round(a-1.7, 6), round(a+4, 6)]
                interval = [round(a-4.5, 6), round(a+6.8, 6)]
                streams = {sp: [w for w in words if w["end"] > interval[0] and w["start"] < interval[1]]
                           for sp, words in full.items()}
                all_words = [w for ws in streams.values() for w in ws]
                if len(all_words) < 10 or sum(good(w) for w in all_words)/len(all_words) < .88:
                    continue
                prefix = "T" if transfer else "S"
                case = dict(number=f"{prefix}{len(cases)+(21 if supplement else 1):02}", id=f"{cid}:overlap:{word_id}", conversation_id=cid,
                            focus=focus, paths=paths, kind=row["label"], stratum=stratum, interval=interval,
                            core=core, target=[a, b], streams=streams, context=context(streams, *interval))
                if not any(c["word_ids"][0] == word_id and c["other_active_run_at_onset"]
                           for c in extract_candidates(case)):
                    continue
                cases.append(case)
                counts[stratum] += 1
                sessions.add(session)
                found = True
                break
            if found and counts == wanted:
                break
    if counts != wanted:
        raise RuntimeError(counts)
    output.write_text(json.dumps(dict(seed=seed, excluded_dialogue_count=len(excluded), strata=counts,
                                      cases=cases), ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAMPLED", counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--transfer", action="store_true")
    args = parser.parse_args()
    sample(args.output, args.prior, args.supplement, args.seed, args.transfer)
