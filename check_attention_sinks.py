"""
Check whether the start-of-video attention in temporal_profile.png comes from attention sinks:
a few video tokens that absorb a large share of attention whatever the video shows.

Uses the attention of the first generated token (the answer letter), averaged over layers.
Writes to <run-dir>/analysis/sinks/:
    summary.txt               how concentrated attention is, where the top token sits, and the twin test
    per_video.csv             the same measures for every video
    temporal_profile_sinks.png  graph 1 redrawn with and without the top 1% of tokens
    top_share_by_layer.png    share of video attention held by the top 1% of tokens, per layer
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_attention import (
    BLUE, CATEGORIES, MUTED, ORANGE, TIME_BINS, TITLES, finish, new_figure, rebin, write_csv,
)

TOP_FRACTION = 0.01


def top_k(n_tokens):
    return max(1, math.ceil(TOP_FRACTION * n_tokens))


def video_measures(attention):
    """attention: [layers, T, H, W] for the answer token."""
    mean_map = attention.mean(axis=0)
    total = mean_map.sum()
    flat = mean_map.ravel()
    n = flat.size
    k = top_k(n)
    order = np.argsort(flat)[::-1]

    trimmed = flat.copy()
    trimmed[order[:k]] = 0.0
    trimmed = trimmed.reshape(mean_map.shape).sum(axis=(1, 2))

    flat_layers = attention.reshape(attention.shape[0], -1)
    per_layer_top = np.sort(flat_layers, axis=1)[:, ::-1][:, :k].sum(axis=1) / flat_layers.sum(axis=1)

    t, row, col = np.unravel_index(order[0], mean_map.shape)
    return {
        "tokens": n,
        "top1_share": flat[order[0]] / total,
        "top1pct_share": flat[order[:k]].sum() / total,
        "top1_index": int(order[0]),
        "top1_t": int(t), "top1_row": int(row), "top1_col": int(col),
        "first_token_share": flat[0] / total,
        "profile_all": rebin(mean_map.sum(axis=(1, 2)) / total),
        "profile_trimmed": rebin(trimmed / trimmed.sum()),
        "per_layer_top": per_layer_top,
        "shape": mean_map.shape,
    }


def load(run_dir):
    videos = {}
    with open(run_dir / "results.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if "error" in row:
                continue
            attention = np.load(run_dir / row["attention_file"])["vision_attention"][0].astype(np.float64)
            measures = video_measures(attention)
            measures.update(category=row["category"], video=row["video"], correct=bool(row["correct"]))
            measures["first_tenth_all"] = measures["profile_all"][0]
            measures["first_tenth_trimmed"] = measures["profile_trimmed"][0]
            videos[(row["category"], row["video"])] = measures
    return videos


def twin_positions(videos, pair_types):
    """For time-reversed pairs: is the twin's top token at the same place, or at the mirrored time?"""
    counts = {c: {"pairs": 0, "same": 0, "mirrored": 0, "chance": []} for c in CATEGORIES}
    for (category, name), original in videos.items():
        if pair_types.get(category, {}).get(name) != "time_reversed":
            continue
        twin = videos.get((category, name.replace(".mp4", "_reversed.mp4")))
        if twin is None or twin["shape"] != original["shape"]:
            continue
        steps = original["shape"][0]
        c = counts[category]
        c["pairs"] += 1
        c["same"] += twin["top1_index"] == original["top1_index"]
        c["mirrored"] += (twin["top1_t"], twin["top1_row"], twin["top1_col"]) == (
            steps - 1 - original["top1_t"], original["top1_row"], original["top1_col"])
        c["chance"].append(1 / original["tokens"])
    return counts


def plot_profiles(videos, path):
    fig, axes = new_figure(
        len(CATEGORIES),
        "Graph 1 with and without the most-attended tokens",
        "Share of the attention on the video in each tenth of its length. The orange line drops each "
        "video's top 1% of tokens. An even spread is 10%.",
    )
    x = (np.arange(TIME_BINS) + 0.5) / TIME_BINS * 100
    for ax, category in zip(axes, CATEGORIES):
        rows = [v for v in videos.values() if v["category"] == category]
        if not rows:
            continue
        for key, label, color in (("profile_all", "All video tokens", BLUE),
                                  ("profile_trimmed", "Top 1% of tokens removed", ORANGE)):
            ax.plot(x, 100 * np.mean([r[key] for r in rows], axis=0), color=color, linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round", label=label)
        ax.axhline(100 / TIME_BINS, color=MUTED, linewidth=1)
        ax.set_title(f"{TITLES[category]}  (n = {len(rows)})", fontsize=10, color="#52514e", loc="left")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["start", "middle", "end"])
    axes[0].set_ylabel("% of attention on the video", fontsize=9.5, color="#52514e")
    finish(fig, axes, path)


def plot_top_share_by_layer(videos, path):
    fig, axes = new_figure(
        len(CATEGORIES),
        "How much of the video attention the top 1% of tokens hold, per layer",
        "If attention were spread evenly, the top 1% of tokens would hold 1%. Far above that means a few "
        "tokens absorb most of it.",
    )
    for ax, category in zip(axes, CATEGORIES):
        rows = [v for v in videos.values() if v["category"] == category]
        if not rows:
            continue
        layers = np.arange(1, len(rows[0]["per_layer_top"]) + 1)
        ax.plot(layers, 100 * np.mean([r["per_layer_top"] for r in rows], axis=0), color=BLUE, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", label="Top 1% of tokens")
        ax.axhline(100 * TOP_FRACTION, color=MUTED, linewidth=1, label="Even spread (1%)")
        ax.set_title(f"{TITLES[category]}  (n = {len(rows)})", fontsize=10, color="#52514e", loc="left")
        ax.set_xlim(1, layers[-1])
        ax.set_xlabel("Layer", fontsize=9.5, color="#52514e")
    axes[0].set_ylabel("% of attention on the video", fontsize=9.5, color="#52514e")
    finish(fig, axes, path)


def summary_lines(videos, twins):
    def pct(values):
        return f"{100 * np.mean(values):.1f}%"

    lines = ["Attention sinks among video tokens (first generated token, mean over layers)", ""]
    header = (f"{'category':<22}{'videos':>7}{'top token':>11}{'even':>8}{'top 1%':>9}"
              f"{'top=1st token':>15}{'top in step 1':>15}{'top at corner':>15}")
    lines.append(header)
    for category in CATEGORIES:
        rows = [v for v in videos.values() if v["category"] == category]
        if not rows:
            continue
        lines.append(
            f"{category:<22}{len(rows):>7}{pct([r['top1_share'] for r in rows]):>11}"
            f"{pct([1 / r['tokens'] for r in rows]):>8}{pct([r['top1pct_share'] for r in rows]):>9}"
            f"{pct([r['top1_index'] == 0 for r in rows]):>15}{pct([r['top1_t'] == 0 for r in rows]):>15}"
            f"{pct([r['top1_row'] == 0 and r['top1_col'] == 0 for r in rows]):>15}"
        )
    lines += [
        "",
        "top token:     share of the video attention held by the single most-attended token",
        "even:          what one token would hold if attention were spread evenly",
        "top 1%:        share held by the most-attended 1% of tokens",
        "top=1st token: videos whose most-attended token is the very first video token",
        "top in step 1: videos whose most-attended token is in the first time step",
        "top at corner: videos whose most-attended token is the top-left cell of a frame",
        "",
        "First tenth of the video: share of video attention, all tokens vs top 1% removed",
        "",
    ]
    for category in CATEGORIES:
        rows = [v for v in videos.values() if v["category"] == category]
        if rows:
            lines.append(f"{category:<22}{pct([r['first_tenth_all'] for r in rows]):>8} -> "
                         f"{pct([r['first_tenth_trimmed'] for r in rows])}   (even spread: 10.0%)")
    lines += ["", "Twin test (time-reversed pairs): where is the twin's most-attended token?", ""]
    lines.append(f"{'category':<22}{'pairs':>6}{'same place':>12}{'mirrored time':>15}{'chance':>9}")
    for category in CATEGORIES:
        c = twins[category]
        if c["pairs"]:
            lines.append(f"{category:<22}{c['pairs']:>6}{100 * c['same'] / c['pairs']:>11.1f}%"
                         f"{100 * c['mirrored'] / c['pairs']:>14.1f}%{100 * np.mean(c['chance']):>8.2f}%")
    lines += [
        "",
        "same place:    the twin's top token sits at the same position, although that position now shows",
        "               different content. Far above chance means the position, not the content, draws attention.",
        "mirrored time: it moved to where the same frame now plays. Far above chance means it follows content.",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--pair-types", type=Path, default=Path("dybench_pair_types.json"))
    args = parser.parse_args()

    with open(args.pair_types) as f:
        pair_types = json.load(f)
    videos = load(args.run_dir)
    print(f"Loaded {len(videos)} videos from {args.run_dir}")

    out_dir = args.run_dir / "analysis" / "sinks"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_video.csv", videos.values(),
              ["category", "video", "correct", "tokens", "top1_share", "top1pct_share", "top1_t", "top1_row",
               "top1_col", "first_token_share", "first_tenth_all", "first_tenth_trimmed"])
    plot_profiles(videos, out_dir / "temporal_profile_sinks.png")
    plot_top_share_by_layer(videos, out_dir / "top_share_by_layer.png")

    lines = summary_lines(videos, twin_positions(videos, pair_types))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nFiles written to {out_dir}")


if __name__ == "__main__":
    main()
