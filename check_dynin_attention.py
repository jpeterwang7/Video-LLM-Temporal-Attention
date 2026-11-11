"""
Check whether Dynin-Omni's attention could stand in for a plan's targets, before building on it.

Check 1, does Dynin-Omni's attention find the labelled objects? For each finished, approved label, AUROC of Dynin-Omni's
attention at the answer (sinks, each video's top 1%, left out) against the label on Dynin-Omni's grid, next to a
control: the same label scored with the attention of 20 other labelled videos. Attention that only favours where objects
usually are, such as the centre, scores as well as the control; only finding this video's object beats it.

Check 2, does Qwen3-VL look at Dynin-Omni's target more when it answers correctly? Dynin-Omni's attention is moved onto
Qwen3-VL's grid (each of its 5 frames onto the time step it came from, its centre crop onto the full frame), and its top
10% of squares is the target. The score is the AUROC of Qwen3-VL's attention (sinks left out) against that target,
over the squares Dynin-Omni saw. If the score is to work as a reward, it must be higher when Qwen3-VL is right: this is
measured per category as the AUROC of the score for telling right answers from wrong (0.5 = no signal), and within
reversed pairs Qwen3-VL answered one right and one wrong, which share their content. The same is shown with the human
labels in place of Dynin-Omni's target, on the labelled videos, for comparison.

Usage (on the server, where both models' attention is saved for every video):
    python check_dynin_attention.py

Writes outputs/dynin_attention_check/summary.txt and per_video.csv.
"""
import argparse
import json
import math
import random
from math import comb
from pathlib import Path

import cv2
import numpy as np

from analyze_attention import write_csv
from run_attention_dynin import crop_window
from score_attention import SINK_FRACTION, auroc, label_mask

CONTROL_VIDEOS = 20


def load_results(run_dir):
    results = {}
    for line in open(run_dir / "results.jsonl"):
        row = json.loads(line)
        if "error" not in row:
            results[(row["category"], row["video"])] = row
    return results


def layer_mean(attention, layers):
    """[layers, ...] -> mean of the last `layers` (all if 0), and the squares that are not sinks."""
    mean = (attention[-layers:] if layers else attention).mean(axis=0)
    keep = np.ones(mean.size, dtype=bool)
    keep[np.argsort(attention.mean(axis=0).ravel())[-math.ceil(SINK_FRACTION * mean.size):]] = False
    return mean, keep.reshape(mean.shape)


def to_qwen_grid(dynin, frames_used, grid, crop, scale=8):
    """Dynin-Omni's [frames, 14, 14] map -> Qwen3-VL's [T, H, W] grid, and which squares it covers."""
    steps, height, width = grid
    x0, y0, x1, y1 = crop
    top, bottom = int(round(y0 * height * scale)), int(round(y1 * height * scale))
    left, right = int(round(x0 * width * scale)), int(round(x1 * width * scale))
    mapped, covered = np.zeros(grid), np.zeros(grid, dtype=bool)
    for t in range(steps):
        frames = [k for k, p in enumerate(frames_used) if p // 2 == t]
        if not frames:
            continue
        fine = np.zeros((height * scale, width * scale), dtype=np.float32)
        fine[top:bottom, left:right] = cv2.resize(dynin[frames].mean(axis=0).astype(np.float32),
                                                  (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
        inside = np.zeros_like(fine)
        inside[top:bottom, left:right] = 1
        mapped[t] = cv2.resize(fine, (width, height), interpolation=cv2.INTER_AREA)
        covered[t] = cv2.resize(inside, (width, height), interpolation=cv2.INTER_AREA) > 0.5
    return mapped, covered


def sign_test(diffs):
    """Exact two-sided sign test on paired differences (ties dropped)."""
    up, down = int(sum(d > 0 for d in diffs)), int(sum(d < 0 for d in diffs))    # Python ints: 2 ** n overflows numpy's
    n = up + down
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n) if n else 1.0


def separation(scores, correct, rng, permutations=5000):
    """AUROC of the score for telling right answers from wrong, and a permutation p-value."""
    scores, correct = np.asarray(scores), np.asarray(correct, dtype=bool)
    observed = auroc(scores, correct)
    if math.isnan(observed):
        return observed, float("nan")
    null = [auroc(scores, rng.permutation(correct)) for _ in range(permutations)]
    # Counting the observed split as one of the permutations keeps p above zero
    return observed, (1 + sum(abs(x - 0.5) >= abs(observed - 0.5) for x in null)) / (1 + permutations)


def twin(video):
    return video.replace("_reversed.mp4", ".mp4") if video.endswith("_reversed.mp4") else video.replace(".mp4", "_reversed.mp4")


def pair_lines(rows, key):
    """Within reversed pairs Qwen3-VL answered one right and one wrong: right twin's score minus the wrong twin's."""
    by_video = {(r["category"], r["video"]): r for r in rows if not math.isnan(r[key])}
    diffs = []
    for (category, video), r in by_video.items():
        other = by_video.get((category, twin(video)))
        if other and r["qwen_correct"] and not other["qwen_correct"]:
            diffs.append(r[key] - other[key])
    if not diffs:
        return "no pairs with one right and one wrong"
    return (f"{len(diffs)} pairs: right minus wrong {np.mean(diffs):+.3f}, higher on the right twin in "
            f"{sum(d > 0 for d in diffs)} of {len(diffs)}, sign test p = {sign_test(diffs):.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dynin-dir", type=Path, default=Path("outputs/dynin_omni"))
    parser.add_argument("--qwen-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--labels", type=Path, nargs="+", default=sorted(Path("labels").glob("labels_peter*.json")),
                        help="Original labels files (on Qwen3-VL's grid); Dynin-Omni's versions are read from its run")
    parser.add_argument("--skip", nargs="*", default=["video_14052_reversed.mp4"], help="Labels to leave out")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--dynin-layers", type=int, default=4, help="Dynin-Omni's last layers to average (0 = all)")
    parser.add_argument("--target-fraction", type=float, default=0.1, help="Share of Dynin-Omni's squares in the target")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dynin_attention_check"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dynin, qwen = load_results(args.dynin_dir), load_results(args.qwen_dir)
    usable = lambda l: l["done"] and not l["unsure"] and l.get("approved", True) and l["video"] not in args.skip
    labels = {}
    for path in args.labels:
        labels.update({(l["category"], l["video"]): l for l in json.load(open(path))["labels"] if usable(l)})
    dynin_labels = {}
    for path in args.labels:
        converted = args.dynin_dir / "labels" / path.name
        if converted.exists():
            dynin_labels.update({(l["category"], l["video"]): l for l in json.load(open(converted))["labels"] if usable(l)})
    lines = []

    # Check 1: Dynin-Omni's attention against the labels, beyond a control
    maps = {}
    for key, label in dynin_labels.items():
        row = dynin.get(key)
        if row is None or label.get("frames_used") != row.get("frames_used"):
            continue
        attention = np.load(args.dynin_dir / row["attention_file"])["vision_attention"][0].astype(np.float64)
        maps[key] = {n: layer_mean(attention, n) for n in (args.dynin_layers, 0)}
    keys = sorted(maps)
    lines += [f"Check 1: Dynin-Omni's attention against your labels ({len(keys)} labelled videos, sinks left out)", ""]
    for n, name in ((args.dynin_layers, f"last {args.dynin_layers} layers"), (0, "all layers")):
        real, control = [], []
        for key in keys:
            mask = label_mask(dynin_labels[key])
            mean, keep = maps[key][n]
            if not mask[keep].any():
                continue
            real.append(auroc(mean[keep], mask[keep]))
            others = [k for k in keys if k != key and (k[0], twin(k[1])) != key]
            scores = []
            for other in random.Random(f"{args.seed}{key}").sample(others, min(CONTROL_VIDEOS, len(others))):
                other_mean, other_keep = maps[other][n]
                both = keep & other_keep
                scores.append(auroc(other_mean[both], mask[both]))
            control.append(float(np.nanmean(scores)))
        diffs = np.array(real) - np.array(control)
        lines.append(f"{name:<16} AUROC {np.nanmean(real):.3f}, control {np.nanmean(control):.3f}, difference "
                     f"{np.nanmean(diffs):+.3f}, real above control in {int(np.sum(diffs > 0))} of {len(diffs)}, "
                     f"sign test p = {sign_test(diffs[~np.isnan(diffs)]):.3g}")

    # Check 2: does Qwen3-VL attend to Dynin-Omni's target more when it answers correctly?
    rows = []
    for n, key in enumerate(sorted(set(dynin) & set(qwen)), start=1):
        if n % 500 == 0:
            print(f"Check 2: {n} videos read", flush=True)
        d, q = dynin[key], qwen[key]
        try:
            d_attention = np.load(args.dynin_dir / d["attention_file"])["vision_attention"][0].astype(np.float64)
            q_attention = np.load(args.qwen_dir / q["attention_file"])["vision_attention"][0].astype(np.float64)
        except FileNotFoundError:
            continue
        cap = cv2.VideoCapture(str(args.data_dir / "video" / key[0] / key[0] / key[1]))
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        target_map, d_keep = layer_mean(d_attention, args.dynin_layers)
        target_map = np.where(d_keep, target_map, 0)
        mapped, covered = to_qwen_grid(target_map, d["frames_used"], q_attention.shape[1:], crop_window(width, height, 224))
        if not covered.any():
            continue
        threshold = np.quantile(mapped[covered], 1 - args.target_fraction)
        target = covered & (mapped >= threshold)
        row = {"category": key[0], "video": key[1], "qwen_correct": bool(q["correct"]), "dynin_correct": bool(d["correct"])}
        for n, name in ((4, "last4"), (0, "all")):
            q_mean, q_keep = layer_mean(q_attention, n)
            scored = covered & q_keep
            row[f"score_{name}"] = auroc(q_mean[scored], target[scored])
            label = labels.get(key)
            row[f"label_{name}"] = auroc(q_mean[q_keep], label_mask(label)[q_keep]) if label else float("nan")
        rows.append(row)

    lines += ["", f"Check 2: Qwen3-VL's attention on Dynin-Omni's target (top {100 * args.target_fraction:.0f}% of Dynin-Omni's "
              f"last {args.dynin_layers} layers), right against wrong answers", ""]
    for key, name in (("score_last4", "Qwen3-VL last 4 layers"), ("score_all", "Qwen3-VL all layers"),
                      ("label_last4", "For comparison, against your labels, last 4"),
                      ("label_all", "For comparison, against your labels, all layers")):
        lines.append(name)
        for category in sorted({r["category"] for r in rows}) + ["overall"]:
            chosen = [r for r in rows if category in ("overall", r["category"]) and not math.isnan(r[key])]
            if not chosen:
                continue
            right = [r[key] for r in chosen if r["qwen_correct"]]
            wrong = [r[key] for r in chosen if not r["qwen_correct"]]
            sep, p = separation([r[key] for r in chosen], [r["qwen_correct"] for r in chosen], rng)
            lines.append(f"  {category:<22}{len(chosen):>5} videos   score when right {np.mean(right) if right else float('nan'):.3f}, "
                         f"wrong {np.mean(wrong) if wrong else float('nan'):.3f}   tells right from wrong: AUROC {sep:.3f} "
                         f"(p = {p:.3g})")
        lines.append("  within pairs: " + pair_lines(rows, key))
        lines.append("")
    lines += ["AUROC 0.5 = no signal. The score must be higher when Qwen3-VL is right (separation above 0.5) to work as a reward."]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    write_csv(args.out_dir / "per_video.csv", rows, list(rows[0]))
    print("\n".join(lines))
    print(f"\nFiles written to {args.out_dir}")


if __name__ == "__main__":
    main()
