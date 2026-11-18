"""
Summarize the TabAlign-for-video runs in the shape of TabAlign's tables.

    A  Main results (Table 4): accuracy per category with binomial standard deviations (Table 9), the macro average,
       pair accuracy, and paired McNemar tests against Qwen3-VL answering directly
    B  Planning and reasoning (Tables 5 and 6a): no plan, Qwen3-VL's plan, LLaDA-8B-Instruct's plan; tool calls and
       seconds per video
    C  Reward signal (Table 6b): no feedback, Qwen3-VL's masks, Dynin-Omni's masks, 20%-noise human masks, human
       masks, on the labelled test videos
    D  Execution errors (Table 20): parse errors, tool errors, plan bypass (final answer before the plan is done),
       stagnation halts
    E  Plan quality (Table 8): parsed plans and target hallucination, LLaDA-8B-Instruct against Qwen3-VL
    F  The frame gap: Dynin-Omni's attention against the human labels and its own accuracy, with 5 centre-cropped
       frames and with one frame per Qwen3-VL time step (centre-cropped or whole)
    G  Does R measure grounding (D.4)? R against a density-preserving shuffled mask, and how well R tells right from
       wrong trajectories, per mask source, from the run without feedback

Writes <out-dir>/summary.txt and per_video.csv.

Usage (main environment):
    python summarize_tabalign.py
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import tabalign_video as tv

CONFIGS = [
    ("none_none", "Qwen3-VL with tools, no plan (LLM reasoning)"),
    ("qwen_dynin", "Qwen3-VL plan + Qwen3-VL + TabAttn (LLM planned)"),
    ("llada_none", "LLaDA plan + Qwen3-VL, no verifier"),
    ("llada_dynin", "LLaDA plan + Qwen3-VL + TabAttn (TabAlign)"),
    ("llada_qwen", "LLaDA plan + Qwen3-VL + TabAttn, Qwen3-VL masks"),
    ("llada_oracle_noise", "LLaDA plan + Qwen3-VL + TabAttn, human masks, 20% noise"),
    ("llada_oracle", "LLaDA plan + Qwen3-VL + TabAttn, human masks (oracle)"),
]
SHORT = {"event_sequence": "Event order", "moving_direction": "Moving dir.", "reversible_dynamics": "Reversible"}


def accuracy_row(name, correct, keys, reference=None):
    """One table row: per-category accuracy +- binomial sd, macro average +- SE, pairs, McNemar against reference."""
    cells, means, variances = [], [], []
    for category in tv.CATEGORIES:
        values = [correct[k] for k in keys if k[0] == category and k in correct]
        if not values:
            cells.append(f"{'—':>15}")
            continue
        p = float(np.mean(values))
        means.append(p)
        variances.append(p * (1 - p) / len(values))
        cells.append(f"{100 * p:5.1f} ±{100 * tv.binomial_sd(p, len(values)):4.1f} ({len(values):>3})")
    macro = f"{100 * np.mean(means):5.1f} ±{100 * math.sqrt(sum(variances)) / len(variances):3.1f}" if means else "—"
    found = [k for k in keys if k in correct]
    overall = f"{100 * np.mean([correct[k] for k in found]):5.1f}" if found else "—"
    pairs = {}
    for category in tv.CATEGORIES:
        right, total = tv.pair_accuracy({k[1]: correct[k] for k in found if k[0] == category})
        pairs[category] = (right, total)
    right, total = sum(r for r, _ in pairs.values()), sum(t for _, t in pairs.values())
    pair_text = f"{100 * right / total:5.1f}" if total else "—"
    test = ""
    if reference is not None:
        shared = [k for k in found if k in reference]
        only_new, only_ref, p = tv.mcnemar([correct[k] for k in shared], [reference[k] for k in shared])
        test = f"  +{only_new}/-{only_ref} p={p:.2g}"
    return f"{name:<58}" + " ".join(cells) + f"   {macro:>12}   {overall:>6}   {pair_text:>6}{test}"


def header():
    return (f"{'':<58}" + " ".join(f"{SHORT[c]:>15}" for c in tv.CATEGORIES)
            + f"   {'Avg (macro)':>12}   {'All':>6}   {'Pairs':>6}  vs direct")


def label_masks(paths):
    labels = {}
    for path in paths:
        for label in json.load(open(path))["labels"]:
            if label["done"] and not label["unsure"] and label.get("approved", True):
                mask = tv.label_cells_mask(label, label["grid"])
                if mask is not None and mask.any():
                    labels[(label["category"], label["video"])] = mask
    return labels


def map_auroc(grid_map, all_map, mask):
    """AUROC of a map on Qwen3-VL's grid against a human mask, sinks left out; squares the model did not see rank
    lowest."""
    sinks = tv.sink_squares(all_map)
    keep = ~sinks
    scores = np.nan_to_num(grid_map, nan=np.nanmin(grid_map) - 1 if np.isfinite(grid_map).any() else 0.0)
    return tv.auroc(scores[keep], mask[keep])


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="test")
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--direct", type=Path, default=Path("outputs/qwen3vl_8b/results.jsonl"))
    parser.add_argument("--dynin-5", type=Path, default=Path("outputs/dynin_omni"), help="The 5-frame Dynin-Omni run")
    parser.add_argument("--labels", type=Path, nargs="*", default=sorted(Path("labels").glob("labels_peter*.json")))
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    args = parser.parse_args()

    sets = tv.load_sets(args.out_dir / "sets.json")
    keys = sets[args.set]
    direct_rows = tv.load_rows(args.direct)
    direct = {k: r["correct"] for k, r in direct_rows.items()}
    runs = {}
    for name, _ in CONFIGS:
        rows = tv.load_rows(args.out_dir / "runs" / args.set / name / "traces.jsonl")
        if rows:
            runs[name] = rows
    lines = [f"TabAlign for video on DyBench, set '{args.set}': {len(keys)} videos "
             f"({', '.join(f'{sum(k[0] == c for k in keys)} {SHORT[c].lower()}' for c in tv.CATEGORIES)}).",
             "Accuracy in % ± binomial standard deviation (videos); Avg = mean of the categories ± its standard error; "
             "Pairs = a video and its reversed twin both right; +a/-b = right only here / right only directly, exact "
             "McNemar p.", ""]

    # A. Main results
    dynin_rows = {mode: tv.load_rows(args.out_dir / "dynin" / f"results_{mode}.jsonl") for mode in ("crop", "squeeze")}
    dynin5 = tv.load_rows(args.dynin_5 / "results.jsonl")
    lines += ["A. Main results (TabAlign Table 4)", header(),
              accuracy_row("Qwen3-VL-8B, directly", direct, keys)]
    for label, rows in (("Dynin-Omni, directly, 5 frames centre-cropped", dynin5),
                        ("Dynin-Omni, directly, 1 frame per time step, cropped", dynin_rows["crop"]),
                        ("Dynin-Omni, directly, 1 frame per time step, whole", dynin_rows["squeeze"])):
        if rows:
            lines.append(accuracy_row(label, {k: r["correct"] for k, r in rows.items()}, keys))
    if "llada_dynin" in runs:
        lines.append(accuracy_row("TabAlign: LLaDA plan + Qwen3-VL + TabAttn (Dynin)",
                                  {k: r["correct"] for k, r in runs["llada_dynin"].items()}, keys, direct))
    lines.append("")

    # B, C. Configurations
    lines += ["B. Planning and reasoning (TabAlign Tables 5 and 6a), and C. reward signal (Table 6b)", header()]
    lines.append(accuracy_row("Qwen3-VL-8B, directly (no tools, no plan)", direct, keys))
    for name, label in CONFIGS:
        if name in runs:
            lines.append(accuracy_row(label, {k: r["correct"] for k, r in runs[name].items()}, keys, direct))
    labels = label_masks(args.labels)
    labelled = [k for k in keys if k in labels]
    if labelled:
        lines += ["", f"   On the {len(labelled)} labelled test videos only (where human masks exist):", header()]
        lines.append(accuracy_row("Qwen3-VL-8B, directly", direct, labelled))
        for name, label in CONFIGS:
            if name in runs:
                lines.append(accuracy_row(label, {k: r["correct"] for k, r in runs[name].items()}, labelled, direct))
    lines.append("")

    # D. Execution
    lines += ["D. Execution (TabAlign Table 20 and E.5)",
              f"{'':<58}{'videos':>7}{'calls':>7}{'parse err':>11}{'tool err':>10}{'bypass':>8}{'halted':>8}"
              f"{'fallback':>10}{'s/video':>9}"]
    for name, label in CONFIGS:
        if name not in runs:
            continue
        rows = [r for k, r in runs[name].items() if k in set(keys)]
        turns = sum(len(r["turns"]) for r in rows)
        lines.append(f"{label:<58}{len(rows):>7}{np.mean([r['n_calls'] for r in rows]):>7.2f}"
                     f"{100 * sum(r['parse_errors'] for r in rows) / max(turns, 1):>10.1f}%"
                     f"{100 * sum(r['tool_errors'] for r in rows) / max(turns, 1):>9.1f}%"
                     f"{100 * np.mean([r['bypass'] for r in rows]):>7.1f}%{100 * np.mean([r['halted'] for r in rows]):>7.1f}%"
                     f"{100 * np.mean([r['fallback_final'] for r in rows]):>9.1f}%{np.mean([r['seconds'] for r in rows]):>9.1f}")
    lines += ["   Errors are per turn; bypass = answered before finishing the plan; halted = stopped by the stagnation "
              "rule; fallback = final letter only from the direct-answer fallback.", ""]

    # E. Plan quality
    lines += ["E. Plan quality (TabAlign Table 8)"]
    for planner in ("llada", "qwen"):
        plans = {k: r for k, r in tv.load_rows(args.out_dir / f"plans_{planner}.jsonl").items() if k in set(keys)}
        if not plans:
            continue
        parsed = [r for r in plans.values() if r.get("steps")]
        steps = [(r, s) for r in parsed for s in r["steps"]]
        tagged = [(r, s) for r, s in steps if s.get("target")]
        halluc = [h for h in (tv.target_hallucinated(s["target"], r) for r, s in tagged) if h is not None]
        bad_plans = [any(tv.target_hallucinated(s.get("target"), r) for s in r["steps"]) for r in parsed]
        copied = sum([s.get("target") for s in r["steps"]] == tv.EXAMPLE_TARGETS for r in parsed)
        lines.append(f"   {('LLaDA-8B-Instruct' if planner == 'llada' else 'Qwen3-VL-8B'):<20} {len(parsed)}/{len(plans)} "
                     f"plans parsed, {np.mean([len(r['steps']) for r in parsed]):.1f} steps each, "
                     f"{100 * len(tagged) / max(len(steps), 1):.0f}% of steps tagged, "
                     f"{100 * sum(1 for _, s in tagged if s['steps']) / max(len(tagged), 1):.0f}% with time steps; "
                     f"hallucinated targets {100 * np.mean(halluc) if halluc else 0:.1f}% of steps, "
                     f"{100 * np.mean(bad_plans) if bad_plans else 0:.1f}% of plans; {copied} copy the example")
    lines.append("")

    # F. Frame gap
    lines += ["F. The frame gap: Dynin-Omni's attention against the human labels (last 4 layers, sinks left out, squares "
              "it did not see rank lowest)"]
    aurocs = {"5 frames, cropped": {}, "1 per time step, cropped": {}, "1 per time step, whole": {}}
    for key, mask in labels.items():
        grid = tuple(mask.shape)
        row5 = dynin5.get(key)
        if row5 and (args.dynin_5 / row5["attention_file"]).exists():
            import cv2
            attention = np.load(args.dynin_5 / row5["attention_file"])["vision_attention"][0].astype(np.float64)
            cap = cv2.VideoCapture(str(tv.video_path(args.data_dir, *key)))
            width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            crop = tv.crop_window(width, height, 224)
            steps = [p // 2 for p in row5["frames_used"]]
            aurocs["5 frames, cropped"][key] = map_auroc(
                tv.place_on_grid(attention[-tv.ATTN_LAYERS:].mean(axis=0), grid, crop, steps),
                tv.place_on_grid(attention.mean(axis=0), grid, crop, steps), mask)
        for mode, name in (("crop", "1 per time step, cropped"), ("squeeze", "1 per time step, whole")):
            path = args.out_dir / "dynin" / mode / key[0] / (Path(key[1]).stem + ".npz")
            if path.exists():
                data = np.load(path)
                if data["last4"].shape[1] == grid[0]:
                    crop = tuple(data["crop"].astype(np.float64))
                    aurocs[name][key] = map_auroc(tv.place_on_grid(data["last4"][0], grid, crop),
                                                  tv.place_on_grid(data["all"][0], grid, crop), mask)
    shared = set.intersection(*[set(v) for v in aurocs.values() if v]) if any(aurocs.values()) else set()
    for name, values in aurocs.items():
        if values:
            on_shared = [values[k] for k in shared if not math.isnan(values[k])]
            lines.append(f"   {name:<28} AUROC {np.nanmean(list(values.values())):.3f} over {len(values)} videos"
                         + (f"; {np.mean(on_shared):.3f} on the {len(shared)} videos all settings share" if shared else ""))
    names = [n for n in aurocs if aurocs[n]]
    for a, b in zip(names, names[1:]):
        diffs = [aurocs[b][k] - aurocs[a][k] for k in shared]
        up, n, p = tv.sign_test(diffs)
        if n:
            lines.append(f"   {b} minus {a}: {np.nanmean(diffs):+.3f}, higher in {up} of {n}, sign test p = {p:.2g}")
    lines.append("")

    # G. Does R measure grounding?
    source_run = runs.get("llada_none")
    if source_run:
        lines += ["G. Does R measure grounding? (TabAlign D.4) From the run without feedback; null = the same number of "
                  "squares per time step placed at random (20 draws)",
                  f"   {'mask source':<16}{'steps':>7}{'R':>8}{'null':>8}{'R/null':>8}{'R > null':>16}"
                  f"{'tells right/wrong (AUROC, trajectory mean R)':>48}"]
        sources = sorted({s for r in source_run.values() for t in r["turns"] for s in (t.get("R") or {})})
        for source in sources:
            real, null, per_traj = [], [], []
            for k, r in source_run.items():
                if k not in set(keys):
                    continue
                values = [(t["R"][source], (t.get("R_null") or {}).get(source)) for t in r["turns"]
                          if (t.get("R") or {}).get(source) is not None]
                real += [v for v, _ in values]
                null += [n for _, n in values]
                if values:
                    per_traj.append((np.mean([v for v, _ in values]), r["correct"]))
            if not real:
                continue
            pairs = [(v, n) for v, n in zip(real, null) if n is not None]
            up, n, p = tv.sign_test([v - m for v, m in pairs])
            sep = tv.auroc([m for m, _ in per_traj], [c for _, c in per_traj])
            lines.append(f"   {source:<16}{len(real):>7}{np.mean(real):>8.3f}{np.mean([m for _, m in pairs]):>8.3f}"
                         f"{np.mean(real) / max(np.mean([m for _, m in pairs]), 1e-9):>8.2f}"
                         f"{f'{up}/{n} p={p:.1g}':>16}{sep:>48.3f}")
        lines.append("")

    calibration = args.out_dir / "calibration.json"
    if calibration.exists():
        cal = json.load(open(calibration))
        lines += [f"TabAttn calibration (validation, {cal['trajectories']} trajectories, {100 * cal['prior']:.1f}% right); "
                  f"Dynin-Omni masks from: {cal.get('dynin_choice')}"]
        for source, fit in cal["sources"].items():
            lines.append(f"   {source:<16} a {fit['a']:+.2f}  b {fit['b']:+.2f}  held-out BCE {fit['inner_bce']:.4f} "
                         f"(prior only {fit['inner_bce_prior_only']:.4f}); R tells right from wrong: AUROC "
                         f"{fit['step_auroc']:.3f} per step, {fit['trajectory_auroc']:.3f} per trajectory")

    text = "\n".join(lines) + "\n"
    (args.out_dir / "summary.txt").write_text(text)
    print(text)

    with open(args.out_dir / "per_video.csv", "w", newline="") as f:
        writer = csv.writer(f)
        names = [n for n, _ in CONFIGS if n in runs]
        writer.writerow(["category", "video", "direct"] + names)
        for k in keys:
            writer.writerow([k[0], k[1], direct.get(k)] + [runs[n][k]["correct"] if k in runs[n] else "" for n in names])
    print(f"Summary: {args.out_dir / 'summary.txt'}\nPer video: {args.out_dir / 'per_video.csv'}")


if __name__ == "__main__":
    main()
