"""
Compare the distillation runs on the held-out test videos.

For each run: accuracy and pair accuracy on the test split; exact McNemar tests against the answers-only run (the
control any gain must beat) and the unchanged model; how close its attention is to Qwen3-VL's (KL, on test videos with
a teacher map, over the last 7 layers); and how well that attention finds the labelled boxes (AUROC, sinks left out,
on finished, approved labels other than --skip).

Usage (one --run per run: a display name, then its run directory):
    python distill_results.py --run "Unchanged" outputs/dynin_omni --run "Answers only" outputs/distillation/answers_only

Writes <data>/results.txt and <data>/results.csv.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from labelled_accuracy import counts, load_results, mcnemar
from score_attention import SINK_FRACTION, auroc, label_mask


def attention_scores(run_dir, data, test, skip, layers):
    """(mean KL to the teacher, videos; mean AUROC against the boxes, videos) on the test split."""
    run_dir = Path(run_dir)
    files = {}
    for line in open(run_dir / "results.jsonl"):
        row = json.loads(line)
        if "error" not in row:
            files[(row["category"], row["video"])] = row["attention_file"]

    kls = []
    for key in test:
        map_path = data / "maps" / key[0] / (Path(key[1]).stem + ".npz")
        if key in files and map_path.exists():
            teacher = np.load(map_path)["map"].astype(np.float64)
            student = np.load(run_dir / files[key])["vision_attention"][0][-layers:].mean(axis=0).astype(np.float64)
            student = student / student.sum()
            used = teacher > 0
            kls.append(float((teacher[used] * (np.log(teacher[used]) - np.log(np.maximum(student[used], 1e-10)))).sum()))

    aurocs = []
    for path in sorted((run_dir / "labels").glob("labels_peter*.json")):
        for label in json.load(open(path))["labels"]:
            key = (label["category"], label["video"])
            if not label["done"] or label["unsure"] or not label.get("approved", True) or label["video"] in skip or key not in files:
                continue
            attention = np.load(run_dir / files[key])["vision_attention"][0][-layers:].mean(axis=0).astype(np.float64).ravel()
            marked = label_mask(label).ravel()
            keep = np.ones(attention.size, dtype=bool)
            keep[np.argsort(attention)[-math.ceil(SINK_FRACTION * attention.size):]] = False
            aurocs.append(auroc(attention[keep], marked[keep]))
    return float(np.mean(kls)) if kls else float("nan"), len(kls), float(np.nanmean(aurocs)) if aurocs else float("nan"), len(aurocs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("NAME", "RUN_DIR"))
    parser.add_argument("--data", type=Path, default=Path("outputs/distillation"))
    parser.add_argument("--control", default="Answers only", help="Name of the run any gain must beat")
    parser.add_argument("--skip", nargs="*", default=["video_14052_reversed.mp4"])
    parser.add_argument("--layers", type=int, default=7, help="Last layers to average, as in training")
    args = parser.parse_args()

    test = sorted(tuple(k) for k in json.load(open(args.data / "split.json"))["test"])
    runs = {name: load_results(run_dir) for name, run_dir in args.run}
    shared = [k for k in test if all(k in r for r in runs.values())]
    lines = [f"{len(test)} test videos, {len(shared)} answered by every run", ""]

    rows = []
    for name, run_dir in args.run:
        results = runs[name]
        for group in sorted({c for c, _ in shared}) + ["overall"]:
            c = counts(results, [k for k in shared if group in ("overall", k[0])])
            rows.append({"run": name, "category": group, "correct": c[0], "total": c[1],
                         "accuracy_pct": round(100 * c[0] / c[1], 1), "pairs_correct": c[2], "pairs": c[3],
                         "pair_accuracy_pct": round(100 * c[2] / max(c[3], 1), 1)})
            lines.append(f"{name:<28}{group:<22}{c[0]:>4} / {c[1]:<4} = {100 * c[0] / c[1]:5.1f}%   "
                         f"pairs {c[2]:>3} / {c[3]:<3} = {100 * c[2] / max(c[3], 1):5.1f}%")
        lines.append("")

    lines += ["Against the other runs (exact McNemar test on the same videos):", ""]
    for name in runs:
        for other in dict.fromkeys([args.control, args.run[0][0]]):
            if other in runs and name != other:
                only_a, only_b, p = mcnemar([runs[name][k] for k in shared], [runs[other][k] for k in shared])
                lines.append(f"{name:<28}vs {other:<20}only first right {only_a:>3}, only second right {only_b:>3}, p = {p:.3g}")
    lines += ["", f"Attention, last {args.layers} layers:", ""]
    for name, run_dir in args.run:
        kl, kl_videos, score, score_videos = attention_scores(run_dir, args.data, test, set(args.skip), args.layers)
        lines.append(f"{name:<28}KL to Qwen3-VL {kl:.3f} ({kl_videos} videos)   AUROC on the boxes {score:.3f} ({score_videos} videos)")

    (args.data / "results.txt").write_text("\n".join(lines) + "\n")
    with open(args.data / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"\nFiles written to {args.data}")


if __name__ == "__main__":
    main()
