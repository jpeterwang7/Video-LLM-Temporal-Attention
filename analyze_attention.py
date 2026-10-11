"""
Summarise the attention recorded by run_attention.py.

Everything uses the attention of the first generated token, which is the answer letter.
Writes to <run-dir>/analysis/:
    per_video.csv         one row per video: where its attention went
    per_pair.csv          one row per original/twin pair: how the attention compares
    summary.txt           the tables printed on screen
    temporal_profile.png  where in the video the model looks, right vs wrong answers
    vision_share_by_layer.png  how much attention the video gets in each layer
    pair_similarity.png   whether attention follows the content or the position when a video is reversed
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

CATEGORIES = ["event_sequence", "moving_direction", "reversible_dynamics"]
TITLES = {
    "event_sequence": "Event sequence",
    "moving_direction": "Moving direction",
    "reversible_dynamics": "Reversible dynamics",
}
OUTCOMES = ["both_right", "one_right", "both_wrong"]
OUTCOME_LABELS = {"both_right": "Both right", "one_right": "One right", "both_wrong": "Both wrong"}
TIME_BINS = 10

SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def rebin(profile, bins=TIME_BINS):
    """Resample a per-time-step attention profile onto a fixed number of equal time bins."""
    cdf = np.concatenate([[0.0], np.cumsum(profile)])
    edges = np.linspace(0, len(profile), bins + 1)
    return np.diff(np.interp(edges, np.arange(len(profile) + 1), cdf))


def correlation(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_ci(values):
    """Mean and 95% confidence half-width, ignoring NaNs."""
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), 0
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else float("nan")
    return float(values.mean()), float(half), len(values)


def load_videos(run_dir, layers):
    videos = {}
    with open(run_dir / "results.jsonl") as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        if "error" in row:
            continue
        data = np.load(run_dir / row["attention_file"])
        attention = data["vision_attention"][0].astype(np.float64)  # [layers, T, H, W]
        per_layer_share = attention.sum(axis=(1, 2, 3))
        time_profile = attention[layers].mean(axis=0).sum(axis=(1, 2))
        time_profile = time_profile / time_profile.sum()
        steps = len(time_profile)
        centers = (np.arange(steps) + 0.5) / steps
        entropy = -(time_profile * np.log(time_profile + 1e-12)).sum() / np.log(steps) if steps > 1 else 0.0
        videos[(row["category"], row["video"])] = {
            "category": row["category"],
            "video": row["video"],
            "correct": bool(row["correct"]),
            "response": row["response"],
            "time_steps": steps,
            "vision_share": float(per_layer_share[layers].mean()),
            "time_center": float((time_profile * centers).sum()),
            "time_spread": float(entropy),
            "per_layer_share": per_layer_share,
            "time_profile": time_profile,
            "time_profile_binned": rebin(time_profile),
        }
    return videos


def build_pairs(videos, pair_types):
    pairs = []
    for (category, name), original in videos.items():
        twin = videos.get((category, name.replace(".mp4", "_reversed.mp4")))
        if twin is None or name.endswith("_reversed.mp4"):
            continue
        n_right = original["correct"] + twin["correct"]
        pair = {
            "category": category,
            "video": name,
            "twin_type": pair_types.get(category, {}).get(name, "unknown"),
            "outcome": OUTCOMES[2 - n_right],
            "original_correct": original["correct"],
            "twin_correct": twin["correct"],
            "same_position": float("nan"),
            "same_content": float("nan"),
        }
        a, b = original["time_profile"], twin["time_profile"]
        if len(a) == len(b):
            # Does the twin's attention sit at the same time steps as the original's?
            pair["same_position"] = correlation(a, b)
            if pair["twin_type"] == "time_reversed":
                # Or on the same frames? In a time-reversed twin, step t shows what step T-1-t showed.
                pair["same_content"] = correlation(a, b[::-1])
        pairs.append(pair)
    return pairs


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def new_figure(n_panels, title, subtitle):
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=9.5, color=INK_SECONDARY, va="top")
    for ax in axes[0]:
        style_axes(ax)
    return fig, axes[0]


def finish(fig, axes, path, ylim=None):
    # Panels share the y-axis, so set its range once, after every panel is drawn
    if ylim is None:
        ylim = (0, axes[0].get_ylim()[1] * 1.08)
    axes[0].set_ylim(*ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=len(labels),
               frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def right_wrong_lines(ax, x, groups, key):
    for label, color in (("Right answers", BLUE), ("Wrong answers", ORANGE)):
        rows = groups[label]
        if rows:
            ax.plot(x, 100 * np.mean([r[key] for r in rows], axis=0), color=color, linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round", label=f"{label}")


def plot_temporal_profile(videos, path):
    fig, axes = new_figure(
        len(CATEGORIES),
        "Where in the video the model looks when it answers",
        "Share of the attention on the video that falls in each tenth of its length. An even spread is 10%.",
    )
    x = (np.arange(TIME_BINS) + 0.5) / TIME_BINS * 100
    for ax, category in zip(axes, CATEGORIES):
        rows = [v for v in videos.values() if v["category"] == category]
        groups = {"Right answers": [r for r in rows if r["correct"]], "Wrong answers": [r for r in rows if not r["correct"]]}
        right_wrong_lines(ax, x, groups, "time_profile_binned")
        ax.set_title(f"{TITLES[category]}  (n = {len(rows)})", fontsize=10, color=INK_SECONDARY, loc="left")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["start", "middle", "end"])
    axes[0].set_ylabel("% of attention on the video", fontsize=9.5, color=INK_SECONDARY)
    finish(fig, axes, path)


def plot_vision_share_by_layer(videos, path):
    fig, axes = new_figure(
        len(CATEGORIES),
        "How much attention the video gets in each layer",
        "Share of the answer token's attention that lands on video tokens; the rest goes to the text.",
    )
    for ax, category in zip(axes, CATEGORIES):
        rows = [v for v in videos.values() if v["category"] == category]
        if not rows:
            continue
        groups = {"Right answers": [r for r in rows if r["correct"]], "Wrong answers": [r for r in rows if not r["correct"]]}
        n_layers = len(rows[0]["per_layer_share"])
        right_wrong_lines(ax, np.arange(1, n_layers + 1), groups, "per_layer_share")
        ax.set_title(f"{TITLES[category]}  (n = {len(rows)})", fontsize=10, color=INK_SECONDARY, loc="left")
        ax.set_xlim(1, n_layers)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Layer", fontsize=9.5, color=INK_SECONDARY)
    axes[0].set_ylabel("% of attention on the video", fontsize=9.5, color=INK_SECONDARY)
    finish(fig, axes, path)


def plot_pair_similarity(pairs, path):
    reversed_pairs = [p for p in pairs if p["twin_type"] == "time_reversed"]
    categories = [c for c in CATEGORIES if sum(p["category"] == c for p in reversed_pairs) >= 20]
    if not categories:
        return False
    fig, axes = new_figure(
        len(categories),
        "When a video is reversed, does attention follow the content or the position?",
        "Correlation between the original's and the reversed twin's attention over time. Bars show 95% intervals.",
    )
    series = (("same_position", "Same time steps", BLUE, -0.12), ("same_content", "Same frames", ORANGE, 0.12))
    for ax, category in zip(axes, categories):
        rows = [p for p in reversed_pairs if p["category"] == category]
        for key, label, color, offset in series:
            for i, outcome in enumerate(OUTCOMES):
                mean, half, n = mean_ci([p[key] for p in rows if p["outcome"] == outcome])
                if n == 0:
                    continue
                if not np.isnan(half):
                    ax.plot([i + offset] * 2, [mean - half, mean + half], color=color, linewidth=2, solid_capstyle="round")
                ax.plot(i + offset, mean, "o", color=color, markersize=9, markeredgecolor=SURFACE,
                        markeredgewidth=2, label=label if i == 0 else None)
        counts = [sum(p["outcome"] == o for p in rows) for o in OUTCOMES]
        ax.set_xticks(range(len(OUTCOMES)), [f"{OUTCOME_LABELS[o]}\n{n} pairs" for o, n in zip(OUTCOMES, counts)])
        ax.set_xlim(-0.5, len(OUTCOMES) - 0.5)
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(TITLES[category], fontsize=10, color=INK_SECONDARY, loc="left")
    axes[0].set_ylabel("Correlation", fontsize=9.5, color=INK_SECONDARY)
    finish(fig, axes, path, ylim=(-1.05, 1.05))
    return True


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_lines(videos, pairs):
    lines = ["Per video (first generated token)", ""]
    lines.append(f"{'category':<22}{'answers':<8}{'n':>6}{'on video':>10}{'time center':>13}{'time spread':>13}")
    for category in CATEGORIES:
        for label, flag in (("right", True), ("wrong", False)):
            rows = [v for v in videos.values() if v["category"] == category and v["correct"] == flag]
            if rows:
                lines.append(
                    f"{category:<22}{label:<8}{len(rows):>6}"
                    f"{100 * np.mean([r['vision_share'] for r in rows]):>9.1f}%"
                    f"{np.mean([r['time_center'] for r in rows]):>13.3f}"
                    f"{np.mean([r['time_spread'] for r in rows]):>13.3f}"
                )
    lines += [
        "",
        "on video:    share of the answer token's attention that lands on video tokens",
        "time center: 0 = all attention on the start of the video, 1 = all on the end",
        "time spread: 1 = spread evenly over time, 0 = on a single time step",
        "",
        "Pairs: how each '_reversed' twin was made",
        "",
    ]
    for category in CATEGORIES:
        counts = defaultdict(int)
        for p in pairs:
            if p["category"] == category:
                counts[p["twin_type"]] += 1
        lines.append(f"{category:<22}" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    lines += ["", "Time-reversed pairs: correlation of attention over time, original vs twin (mean +/- 95% interval)", ""]
    lines.append(f"{'category':<22}{'outcome':<12}{'pairs':>6}{'same time steps':>20}{'same frames':>20}")
    for category in CATEGORIES:
        for outcome in OUTCOMES:
            rows = [p for p in pairs if p["category"] == category and p["twin_type"] == "time_reversed" and p["outcome"] == outcome]
            if rows:
                position, content = mean_ci([p["same_position"] for p in rows]), mean_ci([p["same_content"] for p in rows])
                lines.append(
                    f"{category:<22}{outcome:<12}{len(rows):>6}"
                    f"{position[0]:>12.3f} +/- {position[1]:.3f}{content[0]:>12.3f} +/- {content[1]:.3f}"
                )
    lines += [
        "",
        "same time steps: high when the twin's attention sits at the same moments of playback as the original's",
        "same frames:     high when it sits on the same frames, which play at mirrored moments in a reversed twin",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--pair-types", type=Path, default=Path("dybench_pair_types.json"))
    parser.add_argument("--layers", default=":", help="Layers to average for the time profiles, as start:end (default all)")
    args = parser.parse_args()

    start, _, end = args.layers.partition(":")
    layers = slice(int(start) if start else None, int(end) if end else None)

    with open(args.pair_types) as f:
        pair_types = json.load(f)
    videos = load_videos(args.run_dir, layers)
    pairs = build_pairs(videos, pair_types)
    print(f"Loaded {len(videos)} videos and {len(pairs)} pairs from {args.run_dir}")

    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    write_csv(out_dir / "per_video.csv", videos.values(),
              ["category", "video", "correct", "response", "time_steps", "vision_share", "time_center", "time_spread"])
    write_csv(out_dir / "per_pair.csv", pairs,
              ["category", "video", "twin_type", "outcome", "original_correct", "twin_correct", "same_position", "same_content"])

    plot_temporal_profile(videos, out_dir / "temporal_profile.png")
    plot_vision_share_by_layer(videos, out_dir / "vision_share_by_layer.png")
    if not plot_pair_similarity(pairs, out_dir / "pair_similarity.png"):
        print("Too few time-reversed pairs for pair_similarity.png")

    lines = summary_lines(videos, pairs)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nFiles written to {out_dir}")


if __name__ == "__main__":
    main()
