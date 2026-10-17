"""
Graph the per-video results of score_attention.py: one row per labelled video.

Reads <run-dir>/analysis/labels/per_video.csv and writes per_video_scores.png next to it:
    left   AUROC with sinks left out, next to the control (other videos' attention on the same labels)
    right  lift with sinks left out (share of attention on the boxed squares / share of squares boxed)
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_attention import BASELINE, GRID, INK, INK_SECONDARY, MUTED, ORANGE, SURFACE, style_axes

SHORT = {"moving_direction": "MD", "reversible_dynamics": "RD"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    args = parser.parse_args()

    folder = args.run_dir / "analysis" / "labels"
    with open(folder / "per_video.csv") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["category"], float(r["auroc_no_sink"])))
    names = [f"{SHORT.get(r['category'], r['category'])}  {r['video'].replace('.mp4', '')}  "
             f"({'right' if r['correct'] == 'True' else 'wrong'})" for r in rows]
    y = list(range(len(rows)))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 0.42 * len(rows) + 2.2), sharey=True,
                                      gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.01, 0.985, f"Does attention land on the boxed squares? Each of the {len(rows)} labelled videos",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.95, "Answer token's attention, averaged over layers, sink squares (top 1%) left out. "
             "MD = moving direction, RD = reversible dynamics.", fontsize=9.5, color=INK_SECONDARY, va="top")

    for ax in (left, right):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=1)

    for i, r in enumerate(rows):
        real = float(r["auroc_no_sink"])
        if r["auroc_control"]:
            control = float(r["auroc_control"])
            left.plot([control, real], [i, i], color=BASELINE, linewidth=2, zorder=1)
            left.plot(control, i, "o", color=MUTED, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                      zorder=2, label="Control: other videos' attention" if i == 0 else None)
        left.plot(real, i, "o", color=ORANGE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
                  zorder=3, label="This video's attention" if i == 0 else None)
    left.axvline(0.5, color=MUTED, linewidth=1)
    left.set_xlim(0.3, 0.9)
    left.set_xlabel("AUROC (0.5 = chance)", fontsize=9.5, color=INK_SECONDARY)
    left.set_yticks(y, names, fontsize=8.5, color=INK_SECONDARY)

    lifts = [float(r["lift_no_sink"]) for r in rows]
    right.barh(y, [l - 1 for l in lifts], left=1, height=0.55, color=ORANGE)
    right.axvline(1, color=MUTED, linewidth=1)
    right.set_xlim(0.6, max(2.1, max(lifts) + 0.1))
    right.set_xlabel("Lift (1 = even spread)", fontsize=9.5, color=INK_SECONDARY)
    for i, l in enumerate(lifts):
        right.text(max(l, 1) + 0.03, i, f"{l:.2f}", va="center", fontsize=8, color=INK_SECONDARY)

    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    out = folder / "per_video_scores.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
