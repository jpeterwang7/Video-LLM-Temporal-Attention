"""
Draw the model's attention as a heat map on each labelled video, with the human boxes on top.

For each video in the labels file, shows the first frame of the middle labelled time step, the
answer token's attention on that time step's squares (averaged over layers, or one layer with
--layer), and the labeller's boxes. Sink squares (each video's top 1%) are left out and the heat
map is scaled within each frame, so it shows where in the frame attention goes.
Runs where the attention files and DyBench videos are. Writes
<run-dir>/analysis/labels/attention_vs_labels.png (or attention_vs_labels_layer<N>.png).
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from analyze_attention import INK, INK_SECONDARY, ORANGE, SURFACE
from score_attention import SINK_FRACTION, label_mask, score_video
from show_sink_tokens import read_frame, sample_indices

SHORT = {"moving_direction": "MD", "reversible_dynamics": "RD"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--layer", type=int, default=None, help="Show one layer (1-36) instead of the mean over layers")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    args = parser.parse_args()

    results = {}
    with open(args.run_dir / "results.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if "error" not in row:
                results[(row["category"], row["video"])] = row
    labels = [l for l in json.load(open(args.labels))["labels"] if l["done"] and not l["unsure"]]
    labels.sort(key=lambda l: (l["category"], l["video"]))

    cols = 4
    rows = math.ceil(len(labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    which = f"layer {args.layer}" if args.layer else "averaged over layers"
    fig.text(0.01, 0.995, "Where the model's attention falls, against the human boxes", fontsize=14,
             fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.975, f"Blue = the answer token's attention on each square ({which}, sink squares left out, "
             "darker = more, scaled within each frame). Orange boxes = human labels.",
             fontsize=10, color=INK_SECONDARY, va="top")
    for ax in axes.ravel():
        ax.axis("off")

    for ax, label in zip(axes.ravel(), labels):
        result = results[(label["category"], label["video"])]
        attention = np.load(args.run_dir / result["attention_file"])["vision_attention"][0].astype(np.float64)
        layer_map = attention[args.layer - 1] if args.layer else attention.mean(axis=0)
        flat = layer_map.ravel().copy()
        flat[np.argsort(attention.mean(axis=0).ravel())[-math.ceil(SINK_FRACTION * flat.size):]] = np.nan
        layer_map = flat.reshape(layer_map.shape)

        steps = sorted(int(t) for t in label["needed_steps"])
        t = steps[len(steps) // 2]
        heat = layer_map[t]
        heat = (heat - np.nanmin(heat)) / (np.nanmax(heat) - np.nanmin(heat) + 1e-12)

        path = args.data_dir / "video" / label["category"] / label["category"] / label["video"]
        cap = cv2.VideoCapture(str(path))
        native_fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        indices = sample_indices(total, native_fps, args.fps, args.max_frames)
        h_bar, w_bar = label["frame_size"]
        frame = cv2.resize(read_frame(path, indices[2 * t]), (w_bar, h_bar), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        ax.imshow(gray, cmap="gray", vmin=0, vmax=255, alpha=0.6, extent=(0, w_bar, h_bar, 0))
        ax.imshow(np.ma.masked_invalid(heat), cmap="Blues", vmin=0, vmax=1, alpha=0.6,
                  interpolation="nearest", extent=(0, w_bar, h_bar, 0))
        for x0, y0, x1, y1 in label["boxes"].get(str(t), label["boxes"].get(t, [])):
            box = (x0 * w_bar, y0 * h_bar), (x1 - x0) * w_bar, (y1 - y0) * h_bar
            ax.add_patch(Rectangle(*box, fill=False, edgecolor="white", linewidth=4))
            ax.add_patch(Rectangle(*box, fill=False, edgecolor=ORANGE, linewidth=2.2))
        ax.set_xlim(0, w_bar)
        ax.set_ylim(h_bar, 0)

        score = score_video(attention, label_mask(label))["no_sink"]
        auroc = score["layer_auroc"][args.layer - 1] if args.layer else score["auroc"]
        answer = "right" if result["correct"] else "wrong"
        ax.set_title(f"{SHORT.get(label['category'], label['category'])} {label['video'].replace('.mp4', '')} · {answer}\n"
                     f"AUROC {auroc:.2f} · step {t + 1} of {label['grid'][0]}",
                     fontsize=9, color=INK_SECONDARY, loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"attention_vs_labels_layer{args.layer}.png" if args.layer else "attention_vs_labels.png"
    out = args.run_dir / "analysis" / "labels" / name
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"Drew {len(labels)} videos to {out}")


if __name__ == "__main__":
    main()
