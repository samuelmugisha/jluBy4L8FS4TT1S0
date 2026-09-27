"""
Batch version of the last step of Part 1: merges each TIMIT speaker's recordings into one
long reference file and clones that voice reading a TIMIT prompt (long audio, short text).
Speakers and texts are picked in the same order as voice_cloning_part1.ipynb, and speakers
that already have a clone in data/cloned_audio are skipped, so the script can be resumed.

Usage (run several workers in parallel to use more CPU cores):
    python clone_speakers.py --n 300 --worker 0 --workers 2
    python clone_speakers.py --n 300 --worker 1 --workers 2
"""
import argparse
import os
import re
import time
from pathlib import Path

LONG_DIR = Path("data/long_version_audio")
CLONED_DIR = Path("data/cloned_audio")


def speaker_files():
    my_dict = {}
    for split in ["TIMIT_orig/TRAIN", "TIMIT_orig/TEST"]:
        for file_path in sorted(Path(split).glob("*/*/*.WAV.wav")):
            my_dict.setdefault(file_path.parent.name, []).append(str(file_path))
    return my_dict


def prompt_texts():
    text_list = []
    with open("TIMIT_orig/DOC/PROMPTS.TXT") as text_file:
        for line in text_file:
            if line.startswith(";"):
                continue
            a = re.sub(r"\s*\(\w+\)\s*$", "", line.strip())
            a = a.replace(".", "").replace("?", "").replace("!", "")
            if len(a.split()) >= 11:
                text_list.append(a)
    return text_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(max(1, os.cpu_count() // (2 * args.workers)))
    from moviepy.editor import concatenate_audioclips, AudioFileClip
    import functions

    LONG_DIR.mkdir(parents=True, exist_ok=True)
    CLONED_DIR.mkdir(parents=True, exist_ok=True)
    my_dict = speaker_files()
    text_list = prompt_texts()
    speakers = sorted(my_dict)[:args.n]

    for index, key in enumerate(speakers):
        if index % args.workers != args.worker or (CLONED_DIR / f"{key}.wav").exists():
            continue
        start = time.time()
        long_file = LONG_DIR / f"{key}.wav"
        if not long_file.exists():
            clips = [AudioFileClip(clip) for clip in my_dict[key]]
            concatenate_audioclips(clips).write_audiofile(str(long_file), logger=None)
        functions.synthesize_audio(str(long_file), text_list[index], str(CLONED_DIR / f"{key}.wav"))
        done = len(list(CLONED_DIR.glob("*.wav")))
        print(f"[worker {args.worker}] {key} cloned in {time.time() - start:.0f}s ({done}/{len(speakers)} total)", flush=True)


if __name__ == "__main__":
    main()
