"""
Build a labelling page for marking which time steps, and which regions in them, a person needs
to see to answer each question.

The page shows each video's frames exactly as the model saw them (same sampling and size as
run_attention.py), grouped into the model's time steps of 2 frames. It runs in a browser with no
server: open <out-dir>/index.html. Labels save in the browser as you go, and "Download labels"
writes them to a JSON file. The folder is self-contained (videos included), so it can be zipped
and sent to a second labeller.

By default it picks 8 time-reversed pairs (16 videos): 2 per combination of category (moving
direction, reversible dynamics) and outcome (both right, one right).
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2

from show_sink_tokens import PATCH, resized_size, sample_indices

CATEGORIES = ["moving_direction", "reversible_dynamics"]
OUTCOMES = ["both_right", "one_right"]
TITLES = {"moving_direction": "Moving direction", "reversible_dynamics": "Reversible dynamics"}


def pick_pairs(run_dir, per_group, seed):
    with open(run_dir / "analysis" / "per_pair.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["twin_type"] == "time_reversed"]
    rng = random.Random(seed)
    picks = []
    for category in CATEGORIES:
        for outcome in OUTCOMES:
            group = sorted(r["video"] for r in rows if r["category"] == category and r["outcome"] == outcome)
            picks += [(category, video) for video in rng.sample(group, per_group)]
    return picks


def build_item(category, name, question, data_dir, out_dir, args):
    src = data_dir / "video" / category / category / name
    cap = cv2.VideoCapture(str(src))
    native_fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    indices = sample_indices(total, native_fps, args.fps, args.max_frames)
    h_bar, w_bar = resized_size(len(indices), height, width, args.max_vision_tokens * PATCH * PATCH * 2)

    wanted, frames = set(indices), {}
    for i in range(indices[-1] + 1):
        if not cap.grab():
            break
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames[i] = cv2.resize(frame, (w_bar, h_bar), interpolation=cv2.INTER_AREA)
    cap.release()
    if len(frames) != len(indices):
        raise RuntimeError(f"Could not read all sampled frames of {category}/{name}")

    stem = Path(name).stem
    (out_dir / "frames" / stem).mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    shutil.copy(src, out_dir / "videos" / name)
    steps = []
    for t in range(len(indices) // 2):
        pair = []
        for j in (2 * t, 2 * t + 1):
            rel = f"frames/{stem}/{j:02d}.jpg"
            cv2.imwrite(str(out_dir / rel), frames[indices[j]], [cv2.IMWRITE_JPEG_QUALITY, 85])
            pair.append(rel)
        time = (indices[2 * t] + indices[2 * t + 1]) / 2 / native_fps
        steps.append({"t": t, "time": round(time, 2), "frames": pair})
    return {
        "id": f"{category}/{name}",
        "category": category,
        "title": TITLES[category],
        "video": name,
        "video_path": f"videos/{name}",
        "question": question["question"],
        "candidates": question["candidates"],
        "answer": question["answer"],
        "grid": [len(steps), h_bar // PATCH, w_bar // PATCH],
        "frame_size": [h_bar, w_bar],
        "patch": PATCH,
        "steps": steps,
    }


def order_apart(items, seed):
    """Shuffle so a video and its reversed twin are never shown back to back."""
    rng = random.Random(seed)
    base = lambda item: item["video"].replace("_reversed.mp4", ".mp4")
    for _ in range(1000):
        rng.shuffle(items)
        if all(base(a) != base(b) for a, b in zip(items, items[1:])):
            return items
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/labeling"))
    parser.add_argument("--pairs-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    args = parser.parse_args()

    questions = {}
    for category in CATEGORIES:
        with open(args.data_dir / "json" / f"{category}.json") as f:
            questions.update({(category, q["video"]): q for q in json.load(f)})

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    items = []
    for category, name in pick_pairs(args.run_dir, args.pairs_per_group, args.seed):
        for video in (name, name.replace(".mp4", "_reversed.mp4")):
            items.append(build_item(category, video, questions[(category, video)], args.data_dir, args.out_dir, args))
    items = order_apart(items, args.seed)

    template = (Path(__file__).parent / "labeling_page.html").read_text()
    page = template.replace("__ITEMS__", json.dumps(items))
    (args.out_dir / "index.html").write_text(page)
    print(f"Wrote {len(items)} videos to {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
