"""
Attention over layers and relative time, for the labelled videos.

Time is relative: each video's time steps are placed by where they fall in the video (0% = start,
100% = end) and grouped into equal bins, so videos of different lengths line up. Sink squares
(each video's top 1%) are left out. Two views, each as a 3D surface and as a flat heat map:
    when_each_layer_looks.png  share of a layer's video attention falling in each time bin,
                               relative to an even spread over time (1 = even)
    tracking_the_object.png    attention on the boxed squares at each moment, relative to the
                               boxes' share of the frame at that moment (1 = no better than even)
Writes to <run-dir>/analysis/<out-name>/ (default time_layer).
"""
import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from analyze_attention import INK, INK_SECONDARY, MUTED, SURFACE, TITLES
from score_attention import SINK_FRACTION, label_mask

# Diverging around 1 (= an even spread): blue below, red above, neutral grey at 1
DIVERGING = LinearSegmentedColormap.from_list("even", ["#2a78d6", "#f0efec", "#e34948"])


def time_layer_maps(attention, mask, bins):
    """attention [layers, T, H, W], mask [T, H, W] -> two [layers, bins] maps (NaN where no step falls)."""
    layers, steps = attention.shape[:2]
    flat = attention.mean(axis=0).ravel()
    keep = np.ones(flat.size, dtype=bool)
    keep[np.argsort(flat)[-math.ceil(SINK_FRACTION * flat.size):]] = False
    keep = keep.reshape(mask.shape)

    a = attention * keep
    per_step = a.sum(axis=(2, 3))                                   # [layers, T]
    time_share = per_step / per_step.sum(axis=1, keepdims=True) * steps
    marked = mask & keep
    box_fraction = marked.sum(axis=(1, 2)) / keep.sum(axis=(1, 2))  # [T]
    with np.errstate(invalid="ignore", divide="ignore"):
        box_lift = (a * marked).sum(axis=(2, 3)) / per_step / box_fraction

    which_bin = np.minimum(((np.arange(steps) + 0.5) / steps * bins).astype(int), bins - 1)
    share_map = np.full((layers, bins), np.nan)
    lift_map = np.full((layers, bins), np.nan)
    for b in range(bins):
        sel = which_bin == b
        if sel.any():
            share_map[:, b] = time_share[:, sel].mean(axis=1)
            valid = sel & (box_fraction > 0)
            if valid.any():
                lift_map[:, b] = box_lift[:, valid].mean(axis=1)
    return share_map, lift_map


def draw(groups, key, title, subtitle, colour_label, path, bins):
    fig = plt.figure(figsize=(6.2 * len(groups), 10.5))
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.02, 0.985, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.957, textwrap.fill(subtitle, 150), fontsize=10, color=INK_SECONDARY, va="top")

    maps = {name: np.nanmean(np.stack([v[key] for v in videos]), axis=0) for name, videos in groups.items()}
    low = min(np.nanmin(m) for m in maps.values())
    high = max(np.nanmax(m) for m in maps.values())
    norm = TwoSlopeNorm(vmin=min(low, 0.95), vcenter=1.0, vmax=max(high, 1.05))
    centres = (np.arange(bins) + 0.5) / bins * 100
    edges = np.linspace(0, 100, bins + 1)
    tick_labels = [f"{edges[i]:.0f}–{edges[i + 1]:.0f}%" for i in range(bins)]

    for i, (name, videos) in enumerate(groups.items()):
        m = maps[name]
        layers = np.arange(1, m.shape[0] + 1)

        ax3 = fig.add_subplot(2, len(groups), i + 1, projection="3d")
        X, Y = np.meshgrid(centres, layers)
        ax3.plot_surface(X, Y, m, facecolors=DIVERGING(norm(np.nan_to_num(m, nan=1.0))), shade=False,
                         rstride=1, cstride=1, linewidth=0.3, edgecolor=(0.5, 0.5, 0.5, 0.25), antialiased=True)
        ax3.set_title(f"{name}  (n = {len(videos)})", fontsize=11, color=INK_SECONDARY, loc="left")
        ax3.set_xlabel("Relative time in video (%)", fontsize=9, color=INK_SECONDARY, labelpad=6)
        ax3.set_ylabel("Layer", fontsize=9, color=INK_SECONDARY, labelpad=6)
        ax3.set_zlabel(colour_label, fontsize=9, color=INK_SECONDARY, labelpad=4)
        ax3.set_xticks([0, 50, 100])
        ax3.tick_params(colors=MUTED, labelsize=8)
        ax3.view_init(elev=24, azim=-112)
        ax3.set_zlim(norm.vmin, norm.vmax)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.set_pane_color((0.99, 0.99, 0.98, 1))

        ax2 = fig.add_subplot(2, len(groups), len(groups) + i + 1)
        image = ax2.imshow(m, aspect="auto", origin="lower", cmap=DIVERGING, norm=norm,
                           extent=(0, 100, 0.5, m.shape[0] + 0.5), interpolation="nearest")
        ax2.set_xticks(centres, tick_labels, fontsize=8, color=MUTED)
        ax2.set_yticks(sorted({1, *range(6, m.shape[0], 6), m.shape[0]}))
        ax2.tick_params(colors=MUTED, labelsize=8, length=0)
        ax2.set_xlabel("Relative time in video", fontsize=9, color=INK_SECONDARY)
        ax2.set_ylabel("Layer", fontsize=9, color=INK_SECONDARY)
        for side in ax2.spines.values():
            side.set_visible(False)
        bar = fig.colorbar(image, ax=ax2, fraction=0.05, pad=0.02)
        bar.set_label(colour_label, fontsize=9, color=INK_SECONDARY)
        bar.ax.tick_params(labelsize=8, colors=MUTED)
        bar.outline.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--out-name", default="time_layer")
    parser.add_argument("--bins", type=int, default=5, help="Relative-time bins (default 5: fifths of the video)")
    args = parser.parse_args()

    results = {}
    with open(args.run_dir / "results.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if "error" not in row:
                results[(row["category"], row["video"])] = row

    groups = {}
    for path in args.labels:
        for label in json.load(open(path))["labels"]:
            if not label["done"] or label["unsure"]:
                continue
            result = results[(label["category"], label["video"])]
            attention = np.load(args.run_dir / result["attention_file"])["vision_attention"][0].astype(np.float64)
            share, lift = time_layer_maps(attention, label_mask(label), args.bins)
            groups.setdefault(TITLES[label["category"]], []).append(
                {"share": share, "lift": lift, "correct": bool(result["correct"])})
    groups = {name: groups[name] for name in TITLES.values() if name in groups}
    total = sum(len(v) for v in groups.values())
    print(f"Loaded {total} labelled videos")

    out_dir = args.run_dir / "analysis" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(groups, "share",
         "When each layer looks: attention over relative time and layers",
         f"Share of each layer's attention on the video falling in each 1/{args.bins} of the video, relative to an "
         f"even spread over time (1 = even; red = more, blue = less). Sinks left out. Labelled videos.",
         "Relative to even", out_dir / "when_each_layer_looks.png", args.bins)
    draw(groups, "lift",
         "Tracking the labelled object: attention on the boxes over relative time and layers",
         "Attention on the boxed squares at each moment, relative to the boxes' share of the frame then "
         "(1 = no better than even; red = more, blue = less). Sinks left out. Labelled videos.",
         "Lift on boxes", out_dir / "tracking_the_object.png", args.bins)
    print(f"Files written to {out_dir}")


if __name__ == "__main__":
    main()
