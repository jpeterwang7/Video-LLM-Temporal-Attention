"""
Score the model's attention against human labels from the labelling page.

For each labelled video, the label marks video tokens (time step, row, column) a person needs to see.
Uses the attention of the first generated token (the answer letter) and reports, per layer and
averaged over layers, with all tokens and with the sink tokens (each video's top 1%) left out:
    AUROC  how well attention ranks marked tokens above unmarked ones (0.5 = chance, 1 = perfect)
    lift   share of attention on marked tokens divided by the share of tokens that are marked
           (1 = no better than an even spread)
Writes to <run-dir>/analysis/labels/: summary.txt, per_video.csv, auroc_by_layer.png.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_attention import BLUE, MUTED, ORANGE, TITLES, finish, new_figure, write_csv

SINK_FRACTION = 0.01


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney), with tied scores given their average rank."""
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def label_mask(label):
    mask = np.zeros(label["grid"], dtype=bool)
    for t, cells in label["cells"].items():
        for row, col in cells:
            mask[int(t), row, col] = True
    return mask


def score_video(attention, mask):
    """attention: [layers, T, H, W]; mask: [T, H, W]."""
    flat = attention.reshape(attention.shape[0], -1)
    marked = mask.ravel()
    sinks = np.zeros_like(marked)
    sinks[np.argsort(flat.mean(axis=0))[-math.ceil(SINK_FRACTION * marked.size):]] = True
    out = {}
    for name, keep in (("all", np.ones_like(marked)), ("no_sink", ~sinks)):
        a, m = flat[:, keep], marked[keep]
        layer_auroc = np.array([auroc(a[l], m) for l in range(a.shape[0])])
        share = a[:, m].sum(axis=1) / a.sum(axis=1)
        mean_map = a.mean(axis=0)
        out[name] = {
            "auroc": auroc(mean_map, m),
            "lift": float((mean_map[m].sum() / mean_map.sum()) / m.mean()),
            "layer_auroc": layer_auroc,
            "layer_lift": share / m.mean(),
        }
    out["marked_fraction"] = float(marked.mean())
    out["sink_on_label"] = bool(marked[np.argmax(flat.mean(axis=0))])
    return out


def mean_ci(values):
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(values) < 2:
        return (float(values.mean()) if len(values) else float("nan")), float("nan"), len(values)
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values))), len(values)


def plot_by_layer(rows, path):
    categories = [c for c in TITLES if any(r["category"] == c for r in rows)]
    fig, axes = new_figure(
        len(categories),
        "Does attention land on what a person needs to see? AUROC per layer",
        "AUROC of the answer token's attention against the labelled tokens: 0.5 is chance, 1 is perfect. "
        "Mean over labelled videos.",
    )
    for ax, category in zip(axes, categories):
        group = [r for r in rows if r["category"] == category]
        layers = np.arange(1, len(group[0]["scores"]["all"]["layer_auroc"]) + 1)
        for key, label, color in (("all", "All video tokens", BLUE), ("no_sink", "Sink tokens left out", ORANGE)):
            curve = np.nanmean([r["scores"][key]["layer_auroc"] for r in group], axis=0)
            ax.plot(layers, curve, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label)
        ax.axhline(0.5, color=MUTED, linewidth=1)
        ax.set_title(f"{TITLES[category]}  (n = {len(group)})", fontsize=10, color="#52514e", loc="left")
        ax.set_xlim(1, layers[-1])
        ax.set_xlabel("Layer", fontsize=9.5, color="#52514e")
    axes[0].set_ylabel("AUROC", fontsize=9.5, color="#52514e")
    finish(fig, axes, path, ylim=(0.0, 1.0))


def summary_lines(rows):
    lines = ["Attention against human labels (first generated token, mean over layers)", ""]
    for key, title in (("all", "All video tokens"), ("no_sink", "Sink tokens (top 1%) left out")):
        lines += [title, f"{'group':<34}{'videos':>7}{'AUROC':>18}{'lift':>16}"]
        groups = [("all labelled videos", rows)]
        for category in TITLES:
            groups.append((category, [r for r in rows if r["category"] == category]))
        groups += [("right answers", [r for r in rows if r["correct"]]),
                   ("wrong answers", [r for r in rows if not r["correct"]])]
        for name, group in groups:
            if not group:
                continue
            a, a_ci, n = mean_ci([r["scores"][key]["auroc"] for r in group])
            lf, lf_ci, _ = mean_ci([r["scores"][key]["lift"] for r in group])
            lines.append(f"{name:<34}{n:>7}{a:>10.3f} +/- {a_ci:.3f}{lf:>8.2f} +/- {lf_ci:.2f}")
        lines.append("")
    layer = np.nanmean([r["scores"]["no_sink"]["layer_auroc"] for r in rows], axis=0)
    best = np.argsort(layer)[::-1][:5] + 1
    lines += [
        f"Layers with the highest AUROC once sinks are left out: {', '.join(map(str, best))}",
        f"Share of tokens marked, on average: {100 * np.mean([r['scores']['marked_fraction'] for r in rows]):.1f}%",
        f"Videos whose top (sink) token falls on a marked token: "
        f"{sum(r['scores']['sink_on_label'] for r in rows)} of {len(rows)}",
        "",
        "AUROC: 0.5 = attention ranks marked tokens no higher than unmarked ones; 1 = all marked tokens on top.",
        "lift: share of attention on marked tokens / share of tokens marked; 1 = no better than an even spread.",
        "With few labelled videos, read the +/- 95% intervals before drawing conclusions.",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+", help="labels_<name>.json files from the labelling page")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    args = parser.parse_args()

    results = {}
    with open(args.run_dir / "results.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if "error" not in row:
                results[(row["category"], row["video"])] = row

    rows = []
    for path in args.labels:
        data = json.load(open(path))
        for label in data["labels"]:
            if not label["done"] or label["unsure"]:
                continue
            result = results[(label["category"], label["video"])]
            attention = np.load(args.run_dir / result["attention_file"])["vision_attention"][0].astype(np.float64)
            mask = label_mask(label)
            if attention.shape[1:] != mask.shape:
                raise ValueError(f"Label grid {mask.shape} does not match attention {attention.shape[1:]} "
                                 f"for {label['id']}")
            rows.append({"annotator": data["annotator"], "category": label["category"], "video": label["video"],
                         "correct": bool(result["correct"]), "wrong_answer_flag": label["note"].startswith("WRONG ANSWER"),
                         "scores": score_video(attention, mask)})
    print(f"Scored {len(rows)} labelled videos from {len(args.labels)} file(s)")

    out_dir = args.run_dir / "analysis" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_video.csv", [
        {"annotator": r["annotator"], "category": r["category"], "video": r["video"], "correct": r["correct"],
         "wrong_answer_flag": r["wrong_answer_flag"], "marked_fraction": r["scores"]["marked_fraction"],
         "auroc_all": r["scores"]["all"]["auroc"], "lift_all": r["scores"]["all"]["lift"],
         "auroc_no_sink": r["scores"]["no_sink"]["auroc"], "lift_no_sink": r["scores"]["no_sink"]["lift"],
         "sink_on_label": r["scores"]["sink_on_label"]} for r in rows],
        ["annotator", "category", "video", "correct", "wrong_answer_flag", "marked_fraction", "auroc_all",
         "lift_all", "auroc_no_sink", "lift_no_sink", "sink_on_label"])
    plot_by_layer(rows, out_dir / "auroc_by_layer.png")
    lines = summary_lines(rows)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nFiles written to {out_dir}")


if __name__ == "__main__":
    main()
