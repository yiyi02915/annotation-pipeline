import json
import time
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import soundfile as sf
from nemo.collections.asr.models import SortformerEncLabelModel
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "full_annotation"
MANIFEST = {row["Path"]: row["Size"] for row in json.loads((ROOT / "manifest.json").read_bytes())}
AUDIO = sorted(path for path in MANIFEST if path.endswith(".wav"))
SPEAKER_MODEL = "/data/L202500721/stevezhuang/annotation_pipeline/models/v3/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def assign(rows, tracks, left, right):
    words = []
    for row in rows:
        fa = row["force_align"]
        offset = row["audio_start_sec"]
        if len(fa["words"]) != len(fa["timestamps"]):
            continue
        for index, (text, (start, end)) in enumerate(zip(fa["words"], fa["timestamps"])):
            start += offset
            end += offset
            if not left <= start < end <= right or not any(ch.isalnum() or ch == "_" for ch in text):
                continue
            hits = Counter()
            for lo, hi, speaker in tracks:
                hits[speaker] += max(0, min(hi, end) - max(lo, start)) / (end - start)
            ranked = hits.most_common()
            speaker = (ranked[0][0] if ranked and ranked[0][1] >= .7
                       and (len(ranked) == 1 or ranked[1][1] < .2) else "UNKNOWN")
            words.append(dict(text=text, start=start, end=end, speaker=speaker, id=f'{row["chunk_id"]}:w{index}'))
    return sorted(words, key=lambda w: (w["start"], w["end"], w["id"]))


def joined(words):
    return " ".join(w["text"] for w in words)


def find_gaps(words, tracks, core_left, core_right, clip_start, path, conversation):
    for focus in sorted({w["speaker"] for w in words} - {"UNKNOWN"}):
        focused = [(i, w) for i, w in enumerate(words) if w["speaker"] == focus]
        for j, ((li, left), (ri, right)) in enumerate(zip(focused, focused[1:])):
            lo, hi = left["end"], right["start"]
            if not .500001 < hi - lo <= 18 or not core_left <= (lo + hi) / 2 < core_right:
                continue
            if any(w["speaker"] == "UNKNOWN" for w in words[li+1:ri]):
                continue
            if any(s == focus and min(v, hi-.1) - max(u, lo+.1) > .16 for u, v, s in tracks):
                continue
            window = [w for w in words if w["end"] >= lo-12 and w["start"] <= hi+12]
            speakers = {w["speaker"] for w in window} - {"UNKNOWN"}
            if len(speakers) != 2:
                continue
            other = next(iter(speakers - {focus}))
            before = [w for _, w in focused[max(0, j-20):j+1] if w["start"] >= lo-12]
            after = [w for _, w in focused[j+1:j+22] if w["start"] <= hi+12]
            if min(len(before), len(after)) < 3:
                continue
            segments = []
            for w in window:
                if (not segments or w["speaker"] != segments[-1]["speaker"]
                        or w["start"] - segments[-1]["end"] > .6):
                    segments.append(dict(segment_id=f"S{len(segments)+1}", speaker=w["speaker"],
                                         start=round(w["start"], 3), end=round(w["end"], 3), text=w["text"]))
                else:
                    segments[-1]["text"] += " " + w["text"]
                    segments[-1]["end"] = round(max(segments[-1]["end"], w["end"]), 3)
            target = dict(left_word=left["text"], right_word=right["text"],
                          left_word_id=left["id"], right_word_id=right["id"],
                          absolute_gap=[round(lo, 3), round(hi, 3)],
                          clip_relative_gap=[round(lo-clip_start, 3), round(hi-clip_start, 3)],
                          gap_seconds=round(hi-lo, 3))
            key = f'{conversation}:{left["id"]}:{right["id"]}'
            yield dict(id=key, source_audio="dataset/"+path, conversation_id=conversation,
                       payload=dict(id=key, focus=focus, other=other, target=target,
                                    focus_before_target=joined(before), focus_after_target=joined(after),
                                    other_near_target=[s for s in segments if s["speaker"] == other and s["end"] >= lo-3 and s["start"] <= hi+3],
                                    context=segments))


def extract(path, rows, model):
    began = time.monotonic()
    source = ROOT / "dataset" / path
    info = sf.info(source)
    if info.channels != 1 or info.duration <= 0:
        raise ValueError("Expected a nonempty mono recording")
    if max(row["audio_end_sec"] for row in rows) > info.duration + 1:
        raise ValueError("FA and source recording clocks differ")
    result = {}
    tmp = RUN / "diarization_window.wav"
    conversation = rows[0]["conversation_id"]
    with sf.SoundFile(source) as audio:
        for start in range(0, int(info.duration) + 1, 32):
            stop = min(info.duration, start + 32)
            if stop <= start:
                break
            lo = max(0, start - 16)
            hi = min(info.duration, stop + 16)
            aligned = [r for r in rows if r["audio_end_sec"] > lo and r["audio_start_sec"] < hi]
            if not aligned:
                continue
            rate = audio.samplerate
            audio.seek(round(lo * rate))
            waveform = audio.read(round(hi * rate) - round(lo * rate), dtype="float32")
            divisor = gcd(rate, 16000)
            sf.write(tmp, resample_poly(waveform, 16000 // divisor, rate // divisor), 16000)
            labels = model.diarize(audio=[str(tmp)], batch_size=1)
            if isinstance(labels, tuple):
                labels = labels[0]
            tracks = [(float(a)+lo, float(b)+lo, speaker) for a, b, speaker in map(str.split, labels[0])]
            for row in find_gaps(assign(aligned, tracks, lo, hi), tracks, start, stop, lo, path, conversation):
                result[row["id"]] = row
    return dict(path=path, duration_seconds=info.duration, extraction_seconds=time.monotonic()-began,
                candidates=sorted(result.values(), key=lambda x: x["payload"]["target"]["absolute_gap"]))


def main():
    output = RUN / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    index = defaultdict(list)
    loaded = set()
    failed = {}
    model = SortformerEncLabelModel.restore_from(SPEAKER_MODEL, map_location="cuda")
    model.eval()
    for k, v in dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300).items():
        setattr(model.sortformer_modules, k, v)
    print("DIARIZATION_READY", flush=True)
    while True:
        for rel in sorted(MANIFEST):
            if not rel.endswith("/predictions.jsonl") or rel in loaded:
                continue
            for base in ("dataset", "priority"):
                source = ROOT / base / rel
                if source.is_file() and source.stat().st_size == MANIFEST[rel]:
                    for line in source.open(encoding="utf-8"):
                        row = json.loads(line)
                        if row["alignment_status"] == "aligned":
                            suffix = row["source_audio_path"].rsplit("/enhance/enhance/", 1)[-1]
                            index["audio/enhance/enhance/"+suffix].append(row)
                    loaded.add(rel)
                    print("FA_READY", rel, len(index), flush=True)
                    break
        ready = []
        for path in AUDIO:
            key = Path(path).parent.name
            destination = output / (key + ".json")
            audio = ROOT / "dataset" / path
            if (path in index and path not in failed and not destination.exists()
                    and audio.is_file() and audio.stat().st_size == MANIFEST[path]):
                ready.append((path, destination))
        for path, destination in ready[:12]:
            try:
                result = extract(path, index[path], model)
                save(destination, result)
                print("EXTRACTED", path, len(result["candidates"]), flush=True)
            except Exception as exc:
                failed[path] = repr(exc)
                print("EXTRACTION_ERROR", path, repr(exc), flush=True)
            save(RUN / "extraction_status.json", dict(ready_recordings=len(index), extracted=len(list(output.glob("*.json"))),
                                                      total_audio=len(AUDIO), shards=len(loaded), failures=failed))
        if not ready:
            if len(loaded) == sum(p.endswith("/predictions.jsonl") for p in MANIFEST) and all(
                    (ROOT/"dataset"/p).is_file() and (ROOT/"dataset"/p).stat().st_size == MANIFEST[p] for p in AUDIO):
                save(RUN/"extraction_complete.json", dict(total_audio=len(AUDIO), indexed=len(index), failures=failed,
                                                           without_aligned_rows=sorted(set(AUDIO)-index.keys())))
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
