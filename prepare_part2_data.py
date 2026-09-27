"""
Builds a small stand-in dataset for Part 2 (fake audio detection) when the
original TIMIT / cloned / Blizzard / Common Voice data is not available.

Real audio: utterances from LibriSpeech dev-clean (https://www.openslr.org/12)
Synthetic audio: the same transcripts spoken by macOS `say` TTS voices

Usage:
    python prepare_part2_data.py --librispeech .cache/LibriSpeech/dev-clean --n 300
"""
import argparse
import itertools
import random
import subprocess
from pathlib import Path

import soundfile as sf

VOICES = ["Daniel", "Karen", "Moira", "Rishi", "Samantha", "Fred", "Ralph", "Kathy",
          "Eddy (English (US))", "Flo (English (UK))", "Reed (English (US))", "Tessa"]


def available_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = {line.split("  ")[0].strip() for line in out.splitlines()}
    return [v for v in VOICES if v in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech", default=".cache/LibriSpeech/dev-clean")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    random.seed(42)
    real_dir = Path(args.out) / "real_audio"
    fake_dir = Path(args.out) / "synthetic_audio"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Collect (flac path, transcript) pairs of 6-25 words so clip lengths are comparable
    utterances = []
    for trans in Path(args.librispeech).rglob("*.trans.txt"):
        for line in trans.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            if 6 <= len(text.split()) <= 25:
                utterances.append((trans.parent / f"{utt_id}.flac", text.lower()))
    random.shuffle(utterances)
    utterances = utterances[:args.n]

    voices = itertools.cycle(available_voices())
    for i, (flac, text) in enumerate(utterances, 1):
        audio, sr = sf.read(flac)
        sf.write(real_dir / f"{flac.stem}.wav", audio, sr)
        subprocess.run(["say", "-v", next(voices), "-o", str(fake_dir / f"{flac.stem}.wav"),
                        "--data-format=LEI16@22050", text], check=True)
        if i % 50 == 0:
            print(f"{i} pairs written")
    print(f"Done: {len(utterances)} real and {len(utterances)} synthetic files in {args.out}/")


if __name__ == "__main__":
    main()
