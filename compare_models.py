"""
Compare models' attention on the same labelled videos, following TabAlign's AR-vs-diffusion comparison.

Usage (one --model per model: a display name, its run directory, then its labels files):
    python compare_models.py \\
        --model "Qwen3-VL-8B" outputs/qwen3vl_8b labels/labels_peter.json labels/labels_peter_labeling_round2.json \\
        --model "VidLaDA-8B" outputs/vidlada_8b outputs/vidlada_8b/labels/labels_peter.json outputs/vidlada_8b/labels/labels_peter_labeling_round2.json

For every labelled video and model: AUROC of the answer position's attention against the human
labels, on the mean of the last 4 layers (TabAlign's convention) and on the mean of all layers, with
each video's top 1% of tokens (sinks) left out; how much of the attention reaches the video; how much
of that the top 1% holds; and where in the clip attention sits. Models are compared on the videos
they share with paired Wilcoxon signed-rank tests, every pair of models. Writes outputs/model_comparison/summary.txt and
auroc_by_depth.png (AUROC per layer, with depth scaled 0-1 so models with different layer counts
line up).
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.stats import wilcoxon

from analyze_attention import BLUE, MUTED, ORANGE, TITLES, finish, new_figure
from score_attention import SINK_FRACTION, auroc, label_mask, mean_ci

COLOURS = [BLUE, ORANGE, "#1baf7a", "#9b59d0"]


def video_scores(attention, mask):
    """attention [layers, T, H, W] -> per-video measures, sinks (top 1% on the layer mean) left out."""
    layers = attention.shape[0]
    flat = attention.reshape(layers, -1)
    keep = np.ones(flat.shape[1], dtype=bool)
    keep[np.argsort(flat.mean(axis=0))[-math.ceil(SINK_FRACTION * flat.shape[1]):]] = False
    marked = mask.ravel()
    kept, m = flat[:, keep], marked[keep]
    mean_map = flat.mean(axis=0)
    steps = attention.shape[1]
    per_step = (attention.mean(axis=0) * keep.reshape(attention.shape[1:])).sum(axis=(1, 2))
    centre = float((per_step / per_step.sum() * (np.arange(steps) + 0.5) / steps).sum())
    return {
        "auroc_all": auroc(kept.mean(axis=0), m),
        "auroc_last4": auroc(kept[-4:].mean(axis=0), m),
        "layer_auroc": np.array([auroc(kept[l], m) for l in range(layers)]),
        "vision_share": float(mean_map.sum()),
        "sink_share": float(np.sort(mean_map)[::-1][:math.ceil(SINK_FRACTION * mean_map.size)].sum() / mean_map.sum()),
        "time_lean": centre - 0.5,
    }


def load_model(name, run_dir, label_paths):
    results = {}
    for line in open(run_dir / "results.jsonl"):
        row = json.loads(line)
        if "error" not in row:
            results[(row["category"], row["video"])] = row
    videos = {}
    for path in label_paths:
        for label in json.load(open(path))["labels"]:
            key = (label["category"], label["video"])
            if not label["done"] or label["unsure"] or key not in results:
                continue
            attention = np.load(run_dir / results[key]["attention_file"])["vision_attention"][0].astype(np.float64)
            scores = video_scores(attention, label_mask(label))
            scores.update(correct=bool(results[key]["correct"]), category=label["category"])
            videos[key] = scores
    return {"name": name, "videos": videos}


def lean_flip(videos):
    """For time-reversed pairs: correlation of the original's and twin's time lean (+ = same lean = habit)."""
    pairs = [(v, videos[(c, n.replace(".mp4", "_reversed.mp4"))]) for (c, n), v in videos.items()
             if not n.endswith("_reversed.mp4") and (c, n.replace(".mp4", "_reversed.mp4")) in videos]
    a = np.array([p[0]["time_lean"] for p in pairs])
    b = np.array([p[1]["time_lean"] for p in pairs])
    return len(pairs), float(np.mean(np.sign(a) == np.sign(b))), float(np.corrcoef(a, b)[0, 1])


def summary(models):
    fmt = lambda values: "%.3f +/- %.3f" % mean_ci(values)[:2]
    lines = ["Attention against the human labels, per model (answer position, sinks left out)", ""]
    lines.append(f"{'model':<34}{'videos':>7}{'accuracy':>10}{'AUROC last 4':>20}{'AUROC all layers':>20}"
                 f"{'on video':>10}{'top 1% hold':>13}")
    for model in models:
        v = list(model["videos"].values())
        lines.append(f"{model['name']:<34}{len(v):>7}{100 * np.mean([x['correct'] for x in v]):>9.1f}%"
                     f"{fmt([x['auroc_last4'] for x in v]):>20}{fmt([x['auroc_all'] for x in v]):>20}"
                     f"{100 * np.mean([x['vision_share'] for x in v]):>9.1f}%{100 * np.mean([x['sink_share'] for x in v]):>12.1f}%")
    lines += ["", "By category (AUROC, all layers)", ""]
    for model in models:
        for category in TITLES:
            v = [x for x in model["videos"].values() if x["category"] == category]
            if v:
                lines.append(f"{model['name']:<34}{category:<22}{fmt([x['auroc_all'] for x in v]):>20}"
                             f"   accuracy {100 * np.mean([x['correct'] for x in v]):.1f}% of {len(v)}")
    for reference, other in itertools.combinations(models, 2):
        shared = sorted(set(reference["videos"]) & set(other["videos"]))
        lines += ["", f"Paired comparison on {len(shared)} shared videos: {other['name']} minus {reference['name']}", ""]
        for key, label in (("auroc_last4", "AUROC, last 4 layers"), ("auroc_all", "AUROC, all layers")):
            a = np.array([other["videos"][k][key] for k in shared])
            b = np.array([reference["videos"][k][key] for k in shared])
            ok = ~(np.isnan(a) | np.isnan(b))
            p = wilcoxon(a[ok], b[ok]).pvalue
            lines.append(f"{label:<26} difference {fmt(list(a[ok] - b[ok]))}, higher in {int(np.sum(a[ok] > b[ok]))} of "
                         f"{int(ok.sum())}, Wilcoxon p = {p:.3g}")
    lines += ["", "Time lean when a video is reversed (all layers; + correlation = lean stays put = position habit)", ""]
    for model in models:
        n, same, r = lean_flip(model["videos"])
        lines.append(f"{model['name']:<34}{n:>4} pairs   same direction {100 * same:.0f}%   correlation {r:+.2f}")
    lines += [
        "",
        "AUROC: 0.5 = attention ranks labelled tokens no higher than the rest; 1 = all labelled tokens on top.",
        "Accuracy is on these labelled videos, which were chosen to over-sample Qwen3-VL's mistakes.",
    ]
    return lines


def plot_depth(models, path):
    categories = [c for c in TITLES if any(x["category"] == c for m in models for x in m["videos"].values())]
    fig, axes = new_figure(len(categories), "Attention on the labelled tokens through each model's depth",
                           "AUROC per layer against the human labels (sinks left out), depth scaled from first (0) to "
                           "last (1) layer. 0.5 is chance.")
    for ax, category in zip(axes, categories):
        for colour, model in zip(COLOURS, models):
            v = [x for x in model["videos"].values() if x["category"] == category]
            if not v:
                continue
            curve = np.nanmean([x["layer_auroc"] for x in v], axis=0)
            depth = np.linspace(0, 1, len(curve))
            ax.plot(depth, curve, color=colour, linewidth=2, solid_capstyle="round", label=model["name"])
        ax.axhline(0.5, color=MUTED, linewidth=1)
        ax.set_title(f"{TITLES[category]}", fontsize=10, color="#52514e", loc="left")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Depth (first layer = 0, last = 1)", fontsize=9.5, color="#52514e")
    axes[0].set_ylabel("AUROC", fontsize=9.5, color="#52514e")
    finish(fig, axes, path, ylim=(0.3, 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", action="append", required=True, metavar="NAME RUN_DIR LABELS",
                        help="Display name, run directory, then labels files")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/model_comparison"))
    args = parser.parse_args()

    models = [load_model(m[0], Path(m[1]), [Path(p) for p in m[2:]]) for m in args.model]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_depth(models, args.out_dir / "auroc_by_depth.png")
    lines = summary(models)
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nFiles written to {args.out_dir}")


if __name__ == "__main__":
    main()
