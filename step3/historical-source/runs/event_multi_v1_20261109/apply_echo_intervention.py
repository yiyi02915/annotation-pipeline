"""Apply the same deterministic echo check to judgments already on disk."""
import argparse
from pathlib import Path

from event_extractor import finalize, recover_echo_intervention
from paired_audio_trial import read, save
from run_paired_holdout import validate_lock


def main(root):
    for case in validate_lock(root):
        path = root / "predictions_floor" / f"{case['number']}.mix.json"
        if not path.exists():
            continue
        row = read(path)
        if row["status"] != "ok":
            continue
        repairs = recover_echo_intervention(row["candidates"], row["judgment"])
        if repairs:
            final = finalize(case, row["candidates"], row["judgment"], row["speaker_gate"])
            for key in ("events", "filtered_candidates", "training_events"):
                row[key] = final[key]
            row.setdefault("entry_repairs", []).extend(repairs)
            save(path, row)
            print(case["number"], len(repairs), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
