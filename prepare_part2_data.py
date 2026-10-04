"""
Builds the real (positive) class for Part 2 (fake audio detection):
utterances from LibriSpeech dev-clean (https://www.openslr.org/12), written to data/real_audio.
The synthetic class is the SV2TTS clones in data/cloned_audio (from clone_speakers.py).

Usage:
    python prepare_part2_data.py --librispeech .cache/LibriSpeech/dev-clean --n 300
"""
import argparse
import random
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech", default=".cache/LibriSpeech/dev-clean")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    random.seed(42)
    real_dir = Path(args.out) / "real_audio"
    real_dir.mkdir(parents=True, exist_ok=True)

    # Collect (flac path, transcript) pairs of 6-25 words so clip lengths are comparable
    utterances = []
    for trans in Path(args.librispeech).rglob("*.trans.txt"):
        for line in trans.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            if 6 <= len(text.split()) <= 25:
                utterances.append((trans.parent / f"{utt_id}.flac", text.lower()))
    random.shuffle(utterances)
    utterances = utterances[:args.n]

    for flac, _ in utterances:
        audio, sr = sf.read(flac)
        sf.write(real_dir / f"{flac.stem}.wav", audio, sr)
    print(f"Done: {len(utterances)} real files in {real_dir}/")


if __name__ == "__main__":
    main()
