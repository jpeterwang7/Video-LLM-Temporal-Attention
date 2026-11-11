"""
Split DyBench into train, validation and test, keeping each video and its _reversed twin together.

Test is every pair with a labelled video, so Dynin-Omni is tested, and its attention checked against the labels, on
videos it never trained on. The rest is shuffled by pair within each category and 10% goes to validation.

Usage:
    python make_split.py labels/labels_peter*.json

Writes outputs/split.json: {"train": [[category, video], ...], "val": [...], "test": [...]}.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

CATEGORIES = ["event_sequence", "moving_direction", "reversible_dynamics"]


def pair_name(video):
    return video.replace("_reversed.mp4", ".mp4")


def make_split(questions, labelled, val_fraction, seed):
    held_out = {(c, pair_name(v)) for c, v in labelled}
    pairs = {}
    for category, video in questions:
        pairs.setdefault((category, pair_name(video)), []).append([category, video])
    split = {"train": [], "val": [], "test": []}
    for key in held_out:
        split["test"] += pairs[key]
    rng = random.Random(seed)
    for category in CATEGORIES:
        rest = sorted(k for k in pairs if k[0] == category and k not in held_out)
        rng.shuffle(rest)
        count = round(val_fraction * len(rest))
        for i, key in enumerate(rest):
            split["val" if i < count else "train"] += pairs[key]
    return {name: sorted(videos) for name, videos in split.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files: their videos and twins become the test set")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out", type=Path, default=Path("outputs/split.json"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    questions = []
    for category in CATEGORIES:
        questions += [(category, q["video"]) for q in json.load(open(args.data_dir / "json" / f"{category}.json"))]
    labelled = set()
    for path in args.labels:
        labelled |= {(l["category"], l["video"]) for l in json.load(open(path))["labels"]}
    split = make_split(questions, labelled, args.val_fraction, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(args.out, "w"), indent=1)
    for name, videos in split.items():
        print(f"{name}: {len(videos)} videos, {dict(Counter(c for c, _ in videos))}")
    print(f"Split: {args.out}")


if __name__ == "__main__":
    main()
