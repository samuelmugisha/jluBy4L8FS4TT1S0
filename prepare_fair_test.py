"""
Builds a matched real-vs-cloned dataset for a fairer fake audio detection test:
- data/fair/cloned_audio: the SV2TTS clones from clone_speakers.py
- data/fair/real_audio: one real TIMIT recording from each of the same speakers
Leading/trailing silence is trimmed from every file the same way, which also removes
the 1s of silence synthesize_audio appends to each clone.

Usage:
    python prepare_fair_test.py
"""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

CLONED_DIR = Path("data/cloned_audio")
OUT_DIR = Path("data/fair")
TOP_DB = 30


def trim_and_save(src, dst):
    audio, sr = librosa.load(src, sr=16000)
    trimmed, _ = librosa.effects.trim(audio, top_db=TOP_DB)
    sf.write(dst, trimmed, sr)
    return len(audio) / sr, len(trimmed) / sr


def main():
    (OUT_DIR / "cloned_audio").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_audio").mkdir(parents=True, exist_ok=True)
    speaker_dirs = {p.name: p for split in ["TIMIT_orig/TRAIN", "TIMIT_orig/TEST"] for p in Path(split).glob("*/*")}

    durations = {"cloned": [], "real": []}
    for cloned in sorted(CLONED_DIR.glob("*.wav")):
        speaker = cloned.stem
        # Longest SX/SI sentence of the speaker, to roughly match the clones' 11+ word texts
        # (SA sentences are skipped because every speaker reads the same two)
        candidates = sorted(speaker_dirs[speaker].glob("S[XI]*.WAV.wav"))
        real = max(candidates, key=lambda f: sf.info(f).duration)
        durations["cloned"].append(trim_and_save(cloned, OUT_DIR / "cloned_audio" / f"{speaker}.wav"))
        durations["real"].append(trim_and_save(real, OUT_DIR / "real_audio" / f"{speaker}.wav"))

    for label, values in durations.items():
        before, after = np.array(values).T
        print(f"{label}: {len(values)} files, mean duration {before.mean():.2f}s -> {after.mean():.2f}s after trimming")


if __name__ == "__main__":
    main()
