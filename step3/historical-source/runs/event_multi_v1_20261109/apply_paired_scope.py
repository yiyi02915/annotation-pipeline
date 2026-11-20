"""Apply the current two-participant scope to saved mixed-audio labels."""
import json
from pathlib import Path

from event_extractor import finalize, speaker_gate

D = Path(__file__).resolve().parent


def read(path):
    return json.loads((D / path).read_bytes())


def main():
    path = D / "results.paired.json"
    rows = read(path.name)
    cases = {c["id"]: c for c in read("word_inputs.json")}
    views = read("paired_audio_trial/diarization.json")
    audio = read("paired_audio_trial/audio_comparison.json")
    assert len(rows) == len(cases) and {r["clip_id"] for r in rows} == set(cases)
    for r in rows:
        assert r["status"] == "ok"
        number = r["number"]
        case = cases[r["clip_id"]]
        assert audio[number]["clip_id"] == case["id"]
        gate = speaker_gate(case, dict(mixed_audio_checked=True, mixed_cluster_count=views[number]["mix"]["count"]))
        r.update(finalize(case, r["candidates"], r["judgment"], gate))
        r.update(annotation_source="qwen_text_plus_jev_mixed_audio", audio_semantic_judge_used=True,
                 audio_evidence="paired_audio_trial/audio_comparison.json", scope_version="known_two_participants")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    print(f"Applied dual-participant scope to {len(rows)} clips")


if __name__ == "__main__":
    main()
