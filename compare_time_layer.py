"""
Tracking the labelled object, for several models side by side on one colour and height scale.

plot_time_layer.py draws each model on its own scale, so the same red means different values in two models'
charts. This draws plot_time_layer.py's tracking_the_object view (attention on the labelled boxes over relative
time and layers) with a column per model and a row per category, all on one scale. Each model takes its run
directory and its labels (the converted ones for models that see other frames, as in run_labelled_models.sh):

    python compare_time_layer.py --model outputs/qwen3vl_8b labels/labels_peter*.json \
        --model outputs/dynin_omni outputs/dynin_omni/labels/labels_peter*.json --skip video_14052_reversed.mp4

Writes outputs/model_comparison/tracking_qwen_vs_dynin.png (or --out).
"""
import argparse
from pathlib import Path

from plot_time_layer import MODELS, draw, load_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", nargs="+", action="append", required=True, metavar=("RUN_DIR", "LABELS"),
                        help="A run directory, then its labels files; repeat for each model")
    parser.add_argument("--bins", type=int, default=5, help="Relative-time bins (default 5: fifths of the video)")
    parser.add_argument("--skip", nargs="*", default=[], help="Videos to leave out")
    parser.add_argument("--out", type=Path, default=Path("outputs/model_comparison/tracking_qwen_vs_dynin.png"))
    args = parser.parse_args()

    models = []
    for run_dir, *label_paths in args.model:
        run_dir = Path(run_dir)
        if not label_paths:
            parser.error(f"--model {run_dir}: give its labels files after the run directory")
        groups, missing = load_groups(run_dir, [Path(p) for p in label_paths], args.bins, args.skip)
        name, note = MODELS.get(run_dir.name, (run_dir.name, ""))
        total = sum(len(v) for v in groups.values())
        print(f"{name}: {total} labelled videos" + (f"; WARNING: {missing} have no result and were left out" if missing else ""))
        models.append((name, note, groups, total))

    categories = [c for c in models[0][2] if all(c in m[2] for m in models)]
    panels = {f"{name} · {category}": groups[category] for category in categories for name, _, groups, _ in models}
    totals = {m[3] for m in models}
    count = (f"{totals.pop()} labelled videos per model" if len(totals) == 1 else
             "labelled videos: " + ", ".join(f"{m[0]} {m[3]}" for m in models))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw(panels, "lift",
         "Tracking the labelled object: " + " vs ".join(m[0] for m in models),
         "Attention on the labelled boxes over relative time and layers, relative to the boxes' share of the frame at "
         "that moment (1 = no better than even; red = more, blue = less). All panels share one colour and height "
         f"scale. Sinks left out. {count}." + "".join(m[1] for m in models),
         "Lift on boxes", args.out, args.bins, ncols=len(models))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
