"""
Accuracy of every model on the labelled videos, each model on the same videos.

The hand-labelled videos were chosen to over-sample Qwen3-VL's mistakes, so results are given for all labelled
videos and for the API-labelled ones alone, which were picked at random. Models are compared with exact McNemar
tests on the videos where exactly one of the two is right. A pair counts only when a video and its _reversed twin
are both labelled, and is right only when both are answered correctly.

Usage:
    python labelled_accuracy.py labels/labels_peter*.json

Writes accuracy_labelled.txt, accuracy_labelled.csv and accuracy_labelled.png to outputs/model_comparison/.
"""
import argparse
import csv
import itertools
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from analyze_attention import BLUE, INK_SECONDARY, MUTED, ORANGE, TITLES, finish, new_figure

MODELS = {
    "Qwen3-VL-8B (autoregressive)": "outputs/qwen3vl_8b",
    "LLaVA-Video-7B (autoregressive)": "outputs/llava_video_7b",
    "VidLaDA-8B (diffusion)": "outputs/vidlada_8b",
    "Dynin-Omni-8B (diffusion)": "outputs/dynin_omni",
    "Dream-VL-7B (diffusion)": "outputs/dreamvl_7b",
}
COLOURS = [BLUE, ORANGE, "#1baf7a", "#9b59d0", "#d6457a"]
HAND_FILES = {"labels_peter.json", "labels_peter_labeling_round2.json"}
CHANCE = {"moving_direction": 25.0, "reversible_dynamics": 100 / 3}


def load_results(run_dir):
    results = {}
    for line in open(Path(run_dir) / "results.jsonl"):
        row = json.loads(line)
        if "error" not in row:
            results[(row["category"], row["video"])] = bool(row["correct"])
    return results


def mcnemar(a, b):
    """Exact two-sided McNemar test on two lists of right/wrong: (only a right, only b right, p)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def counts(results, videos):
    """(correct, videos, pairs correct, pairs) on these videos."""
    videos = set(videos)
    pairs = [(c, v) for c, v in videos if not v.endswith("_reversed.mp4") and (c, v.replace(".mp4", "_reversed.mp4")) in videos]
    pairs_right = sum(results[(c, v)] and results[(c, v.replace(".mp4", "_reversed.mp4"))] for c, v in pairs)
    return sum(results[k] for k in videos), len(videos), pairs_right, len(pairs)


def plot(models, sets, path):
    groups = [*CHANCE, "overall"]
    fig, axes = new_figure(len(sets), "Accuracy on the labelled videos",
                           "Each model on the same videos. Grey marks show the accuracy of guessing at random.")
    width = 0.15
    for ax, (set_name, videos) in zip(axes, sets.items()):
        for g, group in enumerate(groups):
            chosen = [k for k in videos if group in ("overall", k[0])]
            chance = np.mean([CHANCE[c] for c, _ in chosen])
            ax.plot([g - 0.42, g + 0.42], [chance, chance], color=MUTED, linewidth=2, solid_capstyle="round", zorder=3)
            for m, (colour, (name, results)) in enumerate(zip(COLOURS, models.items())):
                right, total = counts(results, chosen)[:2]
                ax.bar(g + (m - 2) * (width + 0.015), 100 * right / total, width, color=colour, zorder=2,
                       label=name if g == 0 else None)
        ax.set_xticks(range(len(groups)), [TITLES.get(g, "All").replace(" ", "\n") for g in groups], fontsize=9, color=INK_SECONDARY)
        ax.set_title(f"{set_name} (n = {len(videos)})", fontsize=10, color=INK_SECONDARY, loc="left")
    axes[0].set_ylabel("Accuracy (%)", fontsize=9.5, color=INK_SECONDARY)
    finish(fig, axes, path, ylim=(0, 100))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/model_comparison"))
    args = parser.parse_args()

    hand, api = set(), set()
    for path in args.labels:
        keys = {(l["category"], l["video"]) for l in json.load(open(path))["labels"]}
        (hand if path.name in HAND_FILES else api).update(keys)
    wanted = hand | api
    models = {name: load_results(run_dir) for name, run_dir in MODELS.items()}

    lines = [f"{len(wanted)} labelled videos: {len(hand)} by hand, {len(api)} by the API and reviewed", ""]
    shared = set(wanted)
    for name, results in models.items():
        missing = wanted - set(results)
        shared &= set(results)
        if missing:
            lines.append(f"WARNING: {name} has no answer for {len(missing)} labelled videos; left out for every model")
    sets = {"All labelled": sorted(shared), "API-labelled, picked at random": sorted(shared & api)}
    sets = {name: videos for name, videos in sets.items() if videos}

    rows = []
    for set_name, videos in sets.items():
        lines += [f"{set_name}: {len(videos)} videos", ""]
        for name, results in models.items():
            for group in [*CHANCE, "overall"]:
                c = counts(results, [k for k in videos if group in ("overall", k[0])])
                rows.append({"videos": set_name, "model": name, "category": group, "correct": c[0], "total": c[1],
                             "accuracy_pct": round(100 * c[0] / c[1], 1), "pairs_correct": c[2], "pairs": c[3],
                             "pair_accuracy_pct": round(100 * c[2] / max(c[3], 1), 1)})
                lines.append(f"{name:<34}{group:<22}{c[0]:>4} / {c[1]:<4} = {100 * c[0] / c[1]:5.1f}%   "
                             f"pairs {c[2]:>3} / {c[3]:<3} = {100 * c[2] / max(c[3], 1):5.1f}%")
            lines.append("")
        lines += ["Paired comparison (exact McNemar test):", ""]
        for (name_a, a), (name_b, b) in itertools.combinations(models.items(), 2):
            only_a, only_b, p = mcnemar([a[k] for k in videos], [b[k] for k in videos])
            lines.append(f"{name_a:<34}vs {name_b:<34}only first right {only_a:>3}, only second right {only_b:>3}, p = {p:.3g}")
        lines.append("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "accuracy_labelled.txt").write_text("\n".join(lines) + "\n")
    with open(args.out_dir / "accuracy_labelled.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(models, sets, args.out_dir / "accuracy_labelled.png")
    print("\n".join(lines))
    print(f"Files written to {args.out_dir}")


if __name__ == "__main__":
    main()
