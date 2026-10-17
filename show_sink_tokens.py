"""
Draw each video's most-attended token on the frame it belongs to, to see what the attention sink sits on.

Runs on a computer with the DyBench videos plus analysis/per_video.csv and analysis/sinks/per_video.csv
from check_attention_sinks.py; no model or GPU needed. Frames are sampled and sized exactly as
run_attention.py fed them to the model (default --fps 2, --max-frames 32, --max-vision-tokens 3072).
Writes <run-dir>/analysis/sinks/top_token_examples.png.
"""
import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from analyze_attention import CATEGORIES, INK, INK_SECONDARY, ORANGE, SURFACE, TITLES

PATCH = 32  # pixels per video token side (16-pixel patches merged 2x2)


def sample_indices(total, native_fps, fps, max_frames):
    """Same frame choice as read_video_frames in the Qwen3-VL backend."""
    num = int(total / native_fps * fps)
    num = min(max(num, 4), max_frames, total)
    num -= num % 2
    return [round(i * (total - 1) / (num - 1)) for i in range(num)]


def resized_size(num_frames, height, width, max_pixels, min_pixels=128 * 32 * 32):
    """Qwen3-VL's smart_resize for video: the frame size the model actually saw."""
    min_pixels = min(min_pixels, max_pixels)
    h_bar, w_bar = round(height / PATCH) * PATCH, round(width / PATCH) * PATCH
    t_bar = round(num_frames / 2) * 2
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt(num_frames * height * width / max_pixels)
        h_bar = max(PATCH, math.floor(height / beta / PATCH) * PATCH)
        w_bar = max(PATCH, math.floor(width / beta / PATCH) * PATCH)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / PATCH) * PATCH
        w_bar = math.ceil(width * beta / PATCH) * PATCH
    return h_bar, w_bar


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    for _ in range(index):
        cap.grab()
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def example(data_dir, row, steps, args):
    path = data_dir / "video" / row["category"] / row["category"] / row["video"]
    cap = cv2.VideoCapture(str(path))
    native_fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    indices = sample_indices(total, native_fps, args.fps, args.max_frames)
    h_bar, w_bar = resized_size(len(indices), height, width, args.max_vision_tokens * PATCH * PATCH * 2)
    if (h_bar // PATCH) * (w_bar // PATCH) * steps != int(row["tokens"]):
        return None  # layout doesn't match what the model saw; skip rather than mislabel
    t = int(row["top1_t"])
    frame = read_frame(path, indices[2 * t])
    if frame is None:
        return None
    sy, sx = height / h_bar, width / w_bar
    box = (int(row["top1_col"]) * PATCH * sx, int(row["top1_row"]) * PATCH * sy, PATCH * sx, PATCH * sy)
    return frame, box, t, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    args = parser.parse_args()

    with open(args.run_dir / "analysis" / "per_video.csv") as f:
        steps = {(r["category"], r["video"]): int(r["time_steps"]) for r in csv.DictReader(f)}
    with open(args.run_dir / "analysis" / "sinks" / "per_video.csv") as f:
        rows = [r for r in csv.DictReader(f) if not r["video"].endswith("_reversed.mp4")]

    rng = random.Random(args.seed)
    picks = []
    for category in CATEGORIES:
        pool = [r for r in rows if r["category"] == category]
        rng.shuffle(pool)
        found = 0
        for row in pool:
            result = example(args.data_dir, row, steps[(category, row["video"])], args)
            if result:
                picks.append((row, *result))
                found += 1
            if found == args.per_category:
                break

    cols = args.per_category
    fig, axes = plt.subplots(len(CATEGORIES), cols, figsize=(3.6 * cols, 3.3 * len(CATEGORIES)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.02, 0.985, "Where each video's most-attended token sits", fontsize=13, fontweight="bold",
             color=INK, va="top")
    fig.text(0.02, 0.955, "The box marks the single video token holding the most attention for the answer "
             "(one 32×32 square of the frame the model saw), drawn on the first frame of its time step.",
             fontsize=9.5, color=INK_SECONDARY, va="top")
    for ax in axes.ravel():
        ax.axis("off")
    for i, (row, frame, box, t, n_steps) in enumerate(picks):
        ax = axes[CATEGORIES.index(row["category"])][i % cols]
        ax.imshow(frame)
        x, y, w, h = box
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=4.5))
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=ORANGE, linewidth=2.5))
        answer = "right" if row["correct"] == "True" else "wrong"
        ax.set_title(f"{TITLES[row['category']]} · {answer} answer\n"
                     f"{100 * float(row['top1_share']):.0f}% of video attention · time step {t + 1} of {n_steps}",
                     fontsize=8.5, color=INK_SECONDARY, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.run_dir / "analysis" / "sinks" / "top_token_examples.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Drew {len(picks)} examples to {out}")


if __name__ == "__main__":
    main()
