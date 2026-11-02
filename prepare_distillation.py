"""
Prepare Qwen3-VL's attention as the training target for Dynin-Omni, and the train/validation/test split.

The split keeps a video and its _reversed twin together. Test is every pair with a labelled video, so attention can
be checked against the labels on videos Dynin-Omni never trained on. The rest is shuffled by pair within each
category and 10% goes to validation. Each teacher map is Qwen3-VL's attention at the answer position, with sinks
(each video's top 1%) left out, averaged over layers 22-28, moved onto the frames and centre crop Dynin-Omni sees,
and scaled to sum to 1. Only videos Qwen3-VL answered correctly get a map.

Usage:
    python prepare_distillation.py labels/labels_peter*.json

Writes <out-dir>/split.json and <out-dir>/maps/<category>/<video>.npz (map [frames, rows, cols], num_frames).
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from run_attention_dynin import CATEGORIES, crop_window, pick_frames


def pair_name(video):
    return video.replace("_reversed.mp4", ".mp4")


def make_split(questions, labelled, val_fraction, seed):
    """{"train": [[category, video], ...], "val": [...], "test": [...]}, keeping pairs together."""
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


def teacher_map(attention, first_layer, last_layer, picked, crop, rows, cols, scale=8):
    """attention [layers, T, H, W] -> [frames, rows, cols] summing to 1. Frame k takes the map of the time step it
    falls in; each square is split into scale x scale pieces so the crop's edges fall where they should."""
    flat = attention.mean(axis=0).ravel()
    keep = np.ones(flat.size, dtype=bool)
    keep[np.argsort(flat)[-math.ceil(0.01 * flat.size):]] = False
    steps = attention[first_layer - 1:last_layer].mean(axis=0) * keep.reshape(attention.shape[1:])
    x0, y0, x1, y1 = crop
    frames = []
    for p in picked:
        fine = np.kron(steps[p // 2], np.ones((scale, scale), dtype=steps.dtype))
        height, width = fine.shape
        cropped = fine[int(round(y0 * height)):int(round(y1 * height)), int(round(x0 * width)):int(round(x1 * width))]
        frames.append(cv2.resize(cropped, (cols, rows), interpolation=cv2.INTER_AREA))
    result = np.stack(frames)
    return result / result.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files: their videos and twins become the test set")
    parser.add_argument("--teacher-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/distillation"))
    parser.add_argument("--first-layer", type=int, default=22)
    parser.add_argument("--last-layer", type=int, default=28)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=224)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(args.out_dir / "split.json", "w"), indent=1)
    for name, videos in split.items():
        print(f"{name}: {len(videos)} videos, {dict(Counter(c for c, _ in videos))}")

    made, missing = 0, 0
    rows_per_frame = cols_per_frame = args.resolution // 16
    for line in open(args.teacher_dir / "results.jsonl"):
        row = json.loads(line)
        if "error" in row or not row["correct"]:
            continue
        category, video = row["category"], row["video"]
        attention_path = args.teacher_dir / row["attention_file"]
        if not attention_path.exists():
            missing += 1
            continue
        attention = np.load(attention_path)["vision_attention"][0].astype(np.float64)
        cap = cv2.VideoCapture(str(args.data_dir / "video" / category / category / video))
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        num_frames = 2 * attention.shape[1]
        result = teacher_map(attention, args.first_layer, args.last_layer, pick_frames(num_frames, args.frames),
                             crop_window(width, height, args.resolution), rows_per_frame, cols_per_frame)
        out = args.out_dir / "maps" / category / (Path(video).stem + ".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, map=result.astype(np.float32), num_frames=num_frames)
        made += 1
    print(f"{made} teacher maps written to {args.out_dir / 'maps'}"
          + (f"; WARNING: {missing} attention files were not found" if missing else ""))


if __name__ == "__main__":
    main()
