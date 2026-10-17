"""
Score the model's attention against human labels from the labelling page.

For each labelled video, the label marks video tokens (time step, row, column) a person needs to see.
Uses the attention of the first generated token (the answer letter) and reports, per layer and
averaged over layers, with all tokens and with the sink tokens (each video's top 1%) left out:
    AUROC  how well attention ranks marked tokens above unmarked ones (0.5 = chance, 1 = perfect)
    lift   share of attention on marked tokens divided by the share of tokens that are marked
           (1 = no better than an even spread)
A control scores each video's labels against the attention of other videos with the same token
grid. If attention simply favours some part of the frame where objects tend to be (such as the
centre), the control scores as well as the real attention; only alignment with this video's
content beats it.
Writes to <run-dir>/analysis/labels/: summary.txt, per_video.csv, auroc_by_layer.png, lift_by_layer.png.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_attention import BASELINE, BLUE, MUTED, ORANGE, TITLES, finish, new_figure, write_csv

SINK_FRACTION = 0.01
CONTROL_VIDEOS = 20


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney), with tied scores given their average rank."""
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
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


def layer_panels(rows, title, subtitle, series, ylabel, reference, path, ylim=None):
    """series: (label, color, function from a row to its per-layer values)."""
    categories = [c for c in TITLES if any(r["category"] == c for r in rows)]
    fig, axes = new_figure(len(categories), title, subtitle)
    for ax, category in zip(axes, categories):
        group = [r for r in rows if r["category"] == category]
        for label, color, values in series:
            curves = [values(r) for r in group if values(r) is not None]
            if curves:
                ax.plot(np.arange(1, len(curves[0]) + 1), np.nanmean(curves, axis=0), color=color, linewidth=2,
                        solid_capstyle="round", solid_joinstyle="round", label=label)
        ax.axhline(reference, color=BASELINE, linewidth=1)
        ax.set_title(f"{TITLES[category]}  (n = {len(group)})", fontsize=10, color="#52514e", loc="left")
        ax.set_xlim(1, len(group[0]["scores"]["all"]["layer_auroc"]))
        ax.set_xlabel("Layer", fontsize=9.5, color="#52514e")
    axes[0].set_ylabel(ylabel, fontsize=9.5, color="#52514e")
    finish(fig, axes, path, ylim=ylim)


def plot_by_layer(rows, out_dir):
    layer_panels(
        rows,
        "Does attention land on what a person needs to see? AUROC per layer",
        "Each video's own attention against its labels, and other videos' attention as a control. 0.5 is chance.",
        [("This video's attention (sinks left out)", ORANGE, lambda r: r["scores"]["no_sink"]["layer_auroc"]),
         ("Control: other videos' attention", MUTED,
          lambda r: r["control"]["layer_auroc"] if r["control"] is not None else None)],
        "AUROC", 0.5, out_dir / "auroc_by_layer.png", ylim=(0.0, 1.0),
    )
    layer_panels(
        rows,
        "How much attention the labelled tokens get, compared with an even spread",
        "Lift = share of attention on labelled tokens / share of tokens labelled. 1 is an even spread.",
        [("All video tokens", BLUE, lambda r: r["scores"]["all"]["layer_lift"]),
         ("Sink tokens left out", ORANGE, lambda r: r["scores"]["no_sink"]["layer_lift"])],
        "Lift", 1.0, out_dir / "lift_by_layer.png",
    )


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
    with_control = [r for r in rows if r["control"] is not None]
    if with_control:
        real = [r["scores"]["no_sink"]["auroc"] for r in with_control]
        ctrl = [r["control"]["auroc"] for r in with_control]
        diff = mean_ci([a - b for a, b in zip(real, ctrl)])
        beats = sum(a > b for a, b in zip(real, ctrl))
        lines += [
            "Control: the same labels scored against other videos' attention (same token grid, sinks left out)",
            f"{'videos with a control':<34}{len(with_control):>7}",
            f"{'AUROC, real attention':<34}{'':>7}{mean_ci(real)[0]:>10.3f} +/- {mean_ci(real)[1]:.3f}",
            f"{'AUROC, control':<34}{'':>7}{mean_ci(ctrl)[0]:>10.3f} +/- {mean_ci(ctrl)[1]:.3f}",
            f"{'difference (real - control)':<34}{'':>7}{diff[0]:>10.3f} +/- {diff[1]:.3f}",
            f"{'videos where real beats control':<34}{beats:>7} of {len(with_control)}",
            "",
        ]
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

    by_grid = {}
    for key, row in results.items():
        by_grid.setdefault(tuple(row["grid_thw"]), []).append(key)
    rng = np.random.default_rng(0)
    cache = {}

    def attention_of(key):
        if key not in cache:
            cache[key] = np.load(args.run_dir / results[key]["attention_file"])["vision_attention"][0].astype(np.float64)
        return cache[key]

    def control(label, mask):
        base = label["video"].replace("_reversed.mp4", ".mp4")
        twins = {base, base.replace(".mp4", "_reversed.mp4")}
        pool = [k for k in by_grid.get(tuple(label["grid"]), []) if k[1] not in twins]
        if not pool:
            return None
        picks = [pool[i] for i in rng.choice(len(pool), size=min(CONTROL_VIDEOS, len(pool)), replace=False)]
        scored = [score_video(attention_of(k), mask)["no_sink"] for k in picks]
        return {"auroc": float(np.nanmean([s["auroc"] for s in scored])),
                "layer_auroc": np.nanmean([s["layer_auroc"] for s in scored], axis=0), "videos": len(picks)}

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
                         "scores": score_video(attention, mask), "control": control(label, mask)})
    print(f"Scored {len(rows)} labelled videos from {len(args.labels)} file(s)")

    out_dir = args.run_dir / "analysis" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_video.csv", [
        {"annotator": r["annotator"], "category": r["category"], "video": r["video"], "correct": r["correct"],
         "wrong_answer_flag": r["wrong_answer_flag"], "marked_fraction": r["scores"]["marked_fraction"],
         "auroc_all": r["scores"]["all"]["auroc"], "lift_all": r["scores"]["all"]["lift"],
         "auroc_no_sink": r["scores"]["no_sink"]["auroc"], "lift_no_sink": r["scores"]["no_sink"]["lift"],
         "sink_on_label": r["scores"]["sink_on_label"],
         "auroc_control": r["control"]["auroc"] if r["control"] else "",
         "control_videos": r["control"]["videos"] if r["control"] else 0} for r in rows],
        ["annotator", "category", "video", "correct", "wrong_answer_flag", "marked_fraction", "auroc_all",
         "lift_all", "auroc_no_sink", "lift_no_sink", "sink_on_label", "auroc_control", "control_videos"])
    plot_by_layer(rows, out_dir)
    lines = summary_lines(rows)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nFiles written to {out_dir}")


if __name__ == "__main__":
    main()
