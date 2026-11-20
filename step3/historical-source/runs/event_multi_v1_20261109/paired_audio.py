"""Render one time window from two verified, synchronized source tracks."""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def render_pair(paths, conversation, interval, destination, rate=16000):
    paths = {speaker: Path(path) for speaker, path in paths.items()}
    if (len(paths) != 2 or len({p.resolve() for p in paths.values()}) != 2
            or any(not p.name.startswith(f"{conversation}_{speaker}_") for speaker, p in paths.items())):
        raise ValueError("Expected two distinct Seamless tracks for this conversation and its participant IDs")
    start, stop = interval
    frames = round((stop - start) * rate)
    if start < 0 or frames <= 0:
        raise ValueError("Invalid source window")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    waveforms = {}
    for speaker, path in paths.items():
        with sf.SoundFile(path) as track:
            if track.channels != 1 or stop > track.frames / track.samplerate + 1e-6:
                raise ValueError(f"Missing or incompatible source coverage for {speaker}")
            lo = round(start * track.samplerate)
            track.seek(lo)
            samples = track.read(round(stop * track.samplerate) - lo, dtype="float32")
            divisor = gcd(track.samplerate, rate)
            if track.samplerate != rate:
                samples = resample_poly(samples, rate // divisor, track.samplerate // divisor)
            if abs(len(samples) - frames) > 1:
                raise ValueError(f"Resampled window length mismatch for {speaker}")
            waveforms[speaker] = np.pad(samples[:frames], (0, max(0, frames - len(samples))))
    channels = np.column_stack(list(waveforms.values()))
    # Fixed gain retains relative loudness and the timing of overlaps; no silence trimming.
    mix = np.mean(channels, axis=1)
    gain = min(1.0, .999 / max(.999, float(np.max(np.abs(channels)))))
    files = {"mix": str(destination / "mix.wav"), "stereo": str(destination / "stereo.wav")}
    sf.write(files["mix"], mix * gain, rate)
    sf.write(files["stereo"], channels * gain, rate)
    isolated = {}
    for index, speaker in enumerate(waveforms):
        isolated[speaker] = str(destination / f"track_{index}.wav")
        sf.write(isolated[speaker], channels[:, index] * gain, rate)
    return dict(files=files, isolated=isolated, channel_order=list(waveforms), interval=[start, stop],
                sample_rate=rate, frames=frames, gain=gain, source_tracks={k: str(v) for k, v in paths.items()})
