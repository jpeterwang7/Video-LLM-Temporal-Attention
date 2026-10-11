"""
Run Qwen3-VL on DyBench and record its attention over the video for every question.

For each video this saves <out-dir>/<category>/<video>.npz holding
    vision_attention  [steps, layers, T, H, W]  attention from each generated token
                                                to every video token, averaged over heads
    frame_times       [T]  time in seconds of each temporal group (2 merged frames)
    frame_indices     [2T] indices of the sampled frames in the original video
    frame_size        [2]  height and width of the resized frames, in pixels
    gen_token_ids / gen_tokens  [steps]  the generated tokens
and appends one line per video to <out-dir>/results.jsonl.

Each video token covers a 32x32 pixel square of the resized frame.
"""
import argparse
import json
import logging
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from omnitrace import OmniTracer
from omnitrace.constants import QWEN3VL_MODEL_PATH, QWEN3VL_PIXELS_PER_TOKEN
from omnitrace.core import GenerationConfig

CATEGORIES = ["event_sequence", "moving_direction", "reversible_dynamics"]
LETTERS = "ABCDEFGH"


def build_prompt(item):
    lines = [item["question"], "Options:"]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(item["candidates"])]
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def parse_letter(response, candidates):
    allowed = LETTERS[: len(candidates)]
    match = re.search(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", response)
    if match:
        return match.group(1)
    # Fall back to an option written out in full
    for i, candidate in enumerate(candidates):
        if candidate.lower() in response.lower():
            return LETTERS[i]
    return None


def find_video(data_dir, category, name):
    for path in (
        data_dir / "video" / category / category / name,
        data_dir / "video" / category / name,
    ):
        if path.exists():
            return path
    raise FileNotFoundError(f"Video not found: {category}/{name}")


def load_done(results_path):
    done = {}
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row:
                    done[(row["category"], row["video"])] = row
    return done


def summarize(rows):
    for category in CATEGORIES:
        by_video = {r["video"]: r["correct"] for r in rows if r["category"] == category}
        if not by_video:
            continue
        accuracy = sum(by_video.values()) / len(by_video)
        # DyBench pair accuracy: the original and its reversed twin must both be right
        pairs = [
            (correct, by_video[video.replace(".mp4", "_reversed.mp4")])
            for video, correct in by_video.items()
            if video.replace(".mp4", "_reversed.mp4") in by_video
        ]
        line = f"{category}: accuracy {accuracy:.3f} over {len(by_video)} videos"
        if pairs:
            pair_accuracy = sum(a and b for a, b in pairs) / len(pairs)
            line += f", pair accuracy {pair_accuracy:.3f} over {len(pairs)} pairs"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--category", default="all", choices=CATEGORIES + ["all"])
    parser.add_argument("--limit", type=int, default=None, help="Only the first N videos per category")
    parser.add_argument("--model", default=QWEN3VL_MODEL_PATH)
    parser.add_argument("--method", default="attmean", choices=["attmean", "attraw"])
    parser.add_argument("--fps", type=float, default=2.0, help="Frames sampled per second of video")
    parser.add_argument("--max-frames", type=int, default=32, help="Cap on sampled frames per video")
    parser.add_argument(
        "--max-vision-tokens",
        type=int,
        default=3072,
        help="Cap on video tokens per video; lower this if the GPU runs out of memory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    parser.add_argument("--verbose", action="store_true", help="Show OmniTrace's per-video log")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("omnitrace").setLevel(logging.WARNING)

    categories = CATEGORIES if args.category == "all" else [args.category]
    items = []
    for category in categories:
        with open(args.data_dir / "json" / f"{category}.json") as f:
            # Sorted by name so a video and its "_reversed" twin stay together under --limit
            rows = sorted(json.load(f), key=lambda row: row["video"])
        items += [(category, row) for row in rows[: args.limit]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    done = load_done(results_path)
    todo = [(c, row) for c, row in items if (c, row["video"]) not in done]
    print(f"{len(items)} videos selected, {len(items) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        summarize(list(done.values()))
        return

    tracer = OmniTracer(model_name="qwen3vl", method=args.method, model_path=args.model)
    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens)
    use_cuda = torch.cuda.is_available()

    for n, (category, item) in enumerate(todo, start=1):
        video = item["video"]
        row = {
            "category": category,
            "video": video,
            "question": item["question"],
            "candidates": item["candidates"],
            "answer": item["answer"],
            "answer_letter": LETTERS[item["candidates"].index(item["answer"])],
        }
        start = time.time()
        try:
            result = tracer.trace(
                modality="video",
                prompt=build_prompt(item),
                video=str(find_video(args.data_dir, category, video)),
                video_fps=args.fps,
                video_max_frames=args.max_frames,
                video_max_pixels=args.max_vision_tokens * QWEN3VL_PIXELS_PER_TOKEN,
                generation_config=generation_config,
                return_raw_attention=True,
            )
            raw = result["raw_attention"]
            attention = raw["vision_attention"]

            attention_file = Path(category) / (Path(video).stem + ".npz")
            (args.out_dir / category).mkdir(exist_ok=True)
            np.savez_compressed(
                args.out_dir / attention_file,
                vision_attention=attention,
                frame_times=np.array(raw["frame_times"], dtype=np.float32),
                frame_indices=np.array(raw["frame_indices"], dtype=np.int64),
                frame_size=np.array(raw["frame_size"], dtype=np.int64),
                gen_token_ids=np.array(raw["gen_token_ids"], dtype=np.int64),
                gen_tokens=np.array(raw["gen_tokens"]),
            )

            pred = parse_letter(result["response"], item["candidates"])
            row.update(
                response=result["response"],
                pred_letter=pred,
                correct=pred == row["answer_letter"],
                attention_file=str(attention_file),
                grid_thw=raw["grid_thw"],
                frame_times=raw["frame_times"],
                gen_tokens=raw["gen_tokens"],
                # Share of the first generated token's attention that lands on the video
                vision_share=float(attention[0].sum(axis=(1, 2, 3)).mean()),
                attribution=result["attribution"]["per_chunk_attribution"],
            )
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if not isinstance(e, torch.cuda.OutOfMemoryError):
                traceback.print_exc()
            if use_cuda:
                torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - start, 2)

        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        status = row.get("error") or (
            f"{row['response']!r} ({'right' if row['correct'] else 'wrong'}, answer {row['answer_letter']}) "
            f"grid {row['grid_thw']}"
        )
        memory = f" peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if use_cuda else ""
        print(f"[{n}/{len(todo)}] {category}/{video}: {status} {row['seconds']}s{memory}", flush=True)

    summarize(list(load_done(results_path).values()))
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
