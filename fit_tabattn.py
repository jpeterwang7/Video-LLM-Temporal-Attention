"""
Fit TabAttn's calibration: p(correct) = sigmoid(a * R + b), on the validation split.

TabAlign fits its verifier's two parameters by binary cross-entropy against terminal trajectory correctness, 80/20 with
the inner 20% only for early stopping (Section 4.2, A). Here every step of every validation trajectory carried out
without feedback (execute_tabalign.py --set calib --reward none) gives one example: its R (Eq. 1) for a mask source,
labelled with whether the trajectory ended right. Steps without a mask are left out (TabAlign's uninformative steps).
One fit per mask source. The Dynin-Omni frame setting that makes the masks (whole frame squeezed, or centre crop as
Dynin-Omni was trained) is chosen here, on validation, by the lower held-out cross-entropy.

Writes <out-dir>/calibration.json and prints, per source, the fit and how well R tells right from wrong
trajectories (AUROC 0.5 = no signal).

Usage (main environment):
    python fit_tabattn.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

import tabalign_video as tv


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=tv.OUT_DIR / "runs" / "calib" / "llada_none",
                        help="Validation run carried out without feedback")
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = list(tv.load_rows(args.run / "traces.jsonl").values())
    if not rows:
        raise SystemExit(f"No finished trajectories in {args.run}")
    sources = sorted({s for r in rows for t in r["turns"] for s in (t.get("R") or {})})
    out = {"run": str(args.run), "trajectories": len(rows), "prior": float(np.mean([r["correct"] for r in rows])),
           "sources": {}}
    lines = [f"{len(rows)} validation trajectories, {100 * out['prior']:.1f}% right", ""]
    for source in sources:
        xs, ys, groups, per_traj = [], [], [], []
        for r in rows:
            values = [t["R"][source] for t in r["turns"] if (t.get("R") or {}).get(source) is not None]
            xs += values
            ys += [int(r["correct"])] * len(values)
            groups += [tv.pair_name(r["video"])] * len(values)
            if values:
                per_traj.append((float(np.mean(values)), r["correct"]))
        if len(set(ys)) < 2 or len(xs) < 10:
            lines.append(f"{source}: too few scored steps ({len(xs)})")
            continue
        fit = tv.fit_calibration(xs, ys, groups, seed=args.seed)
        fit.update(step_auroc=tv.auroc(xs, ys), trajectory_auroc=tv.auroc([m for m, _ in per_traj], [c for _, c in per_traj]),
                   mean_R=float(np.mean(xs)), steps=len(xs))
        out["sources"][source] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in fit.items()}
        lines.append(f"{source:<14} {len(xs):>5} steps  mean R {fit['mean_R']:.3f}  a {fit['a']:+.2f} b {fit['b']:+.2f}  "
                     f"held-out BCE {fit['inner_bce']:.4f} (prior only {fit['inner_bce_prior_only']:.4f})  "
                     f"R tells right from wrong: AUROC {fit['step_auroc']:.3f} per step, {fit['trajectory_auroc']:.3f} "
                     f"per trajectory")
    dynin = {s: f for s, f in out["sources"].items() if s.startswith("dynin_")}
    if dynin:
        out["dynin_choice"] = min(dynin, key=lambda s: (dynin[s]["inner_bce"], s != "dynin_squeeze"))
        lines += ["", f"Dynin-Omni masks for the test runs: {out['dynin_choice']} (lower held-out cross-entropy)"]
    else:
        out["dynin_choice"] = "dynin_squeeze"
        lines += ["", "No Dynin-Omni scores in the validation run; defaulting to dynin_squeeze"]
    path = args.out_dir / "calibration.json"
    json.dump(out, open(path, "w"), indent=1)
    print("\n".join(lines))
    print(f"\nCalibration: {path}")


if __name__ == "__main__":
    main()
