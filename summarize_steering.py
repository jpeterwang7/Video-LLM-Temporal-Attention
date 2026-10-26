"""
Summarise steer_attention.py: does pushing attention onto the labelled object change the answers?

Reads outputs/steering/<model>/results.jsonl for every model present. For each model and bias: accuracy and
the mean probability of the correct letter with the object boosted, with as many random squares boosted (the
control), and unsteered; the paired difference between object and random (Wilcoxon signed-rank over videos);
how many answers flip; and how much of the answer position's attention the object then gets. Writes
outputs/steering/summary.txt and p_correct_by_bias.png.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.stats import wilcoxon

from analyze_attention import BLUE, MUTED, ORANGE, finish, new_figure
from plot_time_layer import MODELS
from score_attention import mean_ci

ORDER = ["qwen3vl_8b", "llava_video_7b", "vidlada_8b", "dynin_omni", "dreamvl_7b"]


def load(path):
    rows = {}
    for line in open(path):
        row = json.loads(line)
        if "error" not in row:
            rows[(row["category"], row["video"], row["target"], row["bias"])] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("outputs/steering"))
    args = parser.parse_args()

    runs = {name: load(args.dir / name / "results.jsonl") for name in ORDER if (args.dir / name / "results.jsonl").exists()}
    fmt = lambda values: "%.3f +/- %.3f" % mean_ci(values)[:2]
    lines = ["Pushing the answer position's attention onto the labelled object (bias b multiplies the attention",
             "weights of the boosted squares by e^b in every layer). Control: as many random unlabelled squares from",
             "the same time steps. p = probability of the correct letter among the options.", ""]
    curves = {}
    for name, rows in runs.items():
        videos = sorted({(c, v) for c, v, t, b in rows if t == "none"})
        biases = sorted({b for _, _, t, b in rows if t != "none"})
        base = {k: rows[(*k, "none", 0.0)] for k in videos}
        title = MODELS.get(name, (name, ""))[0]
        lines += [f"{title}: {len(videos)} videos",
                  f"  unsteered: accuracy {100 * np.mean([r['correct'] for r in base.values()]):.1f}%, "
                  f"p {fmt([r['p_correct'] for r in base.values()])}, "
                  f"object holds {100 * np.mean([r['object_share'] for r in base.values()]):.2f}% of the answer position's attention",
                  f"  {'bias':>5}  {'boost':>8}  {'accuracy':>9}  {'p (mean +/- 95% CI)':>21}  {'change in p':>18}  "
                  f"{'object share':>12}  {'flips':>12}"]
        curves[name] = {"none": np.mean([r["p_correct"] for r in base.values()]), "object": [], "random": [], "biases": biases}
        for b in biases:
            for target in ("object", "random"):
                cur = [rows.get((*k, target, b)) for k in videos]
                shared = [(r, base[k]) for r, k in zip(cur, videos) if r is not None]
                p = [r["p_correct"] for r, _ in shared]
                change = [r["p_correct"] - r0["p_correct"] for r, r0 in shared]
                gained = sum(r["correct"] and not r0["correct"] for r, r0 in shared)
                lost = sum(r0["correct"] and not r["correct"] for r, r0 in shared)
                lines.append(f"  {b:>5g}  {target:>8}  {100 * np.mean([r['correct'] for r, _ in shared]):>8.1f}%  {fmt(p):>21}  "
                             f"{fmt(change):>18}  {100 * np.mean([r['object_share'] for r, _ in shared]):>11.1f}%  "
                             f"+{gained} / -{lost}")
                curves[name][target].append(np.mean(p))
            pairs = [(rows[(*k, "object", b)]["p_correct"], rows[(*k, "random", b)]["p_correct"]) for k in videos
                     if (*k, "object", b) in rows and (*k, "random", b) in rows]
            d = np.array([o - r for o, r in pairs])
            p_value = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
            lines.append(f"  {'':>5}  object minus random: {fmt(list(d))}, object higher in {int((d > 0).sum())} of "
                         f"{len(d)}, Wilcoxon p = {p_value:.3g}")
        lines.append("")
    lines += ["flips: answers turned right (+) and wrong (-) compared with the unsteered model.",
              "Accuracy here is the letter with the highest score at the answer position in one forward pass."]
    (args.dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    subtitle = ("Mean probability of the correct letter when the answer position's attention to the labelled squares (blue) "
                "or to as many random squares (orange) is multiplied by e^bias in every layer; grey dashed = unsteered.")
    fig, axes = new_figure(len(curves), "Pushing attention onto the labelled object",
                           textwrap.fill(subtitle, int(4.2 * len(curves) * 13)))  # about 13 characters per inch
    for ax, (name, c) in zip(axes, curves.items()):
        x = np.arange(len(c["biases"]) + 1)
        ax.plot(x, [c["none"]] + c["object"], color=BLUE, linewidth=2, marker="o", markersize=4, label="Labelled object boosted")
        ax.plot(x, [c["none"]] + c["random"], color=ORANGE, linewidth=2, marker="o", markersize=4, label="Random squares boosted")
        ax.axhline(c["none"], color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(x, ["0"] + [f"{b:g}" for b in c["biases"]])
        ax.set_xlabel("Bias (attention weight x e^bias)", fontsize=9.5, color="#52514e")
        ax.set_title(MODELS.get(name, (name, ""))[0], fontsize=10, color="#52514e", loc="left")
    axes[0].set_ylabel("P(correct letter)", fontsize=9.5, color="#52514e")
    finish(fig, axes, args.dir / "p_correct_by_bias.png", ylim=(0, 1))
    print(f"\nFiles written to {args.dir}")


if __name__ == "__main__":
    main()
