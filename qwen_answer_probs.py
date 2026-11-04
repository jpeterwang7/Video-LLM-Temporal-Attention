"""
Record how strongly Qwen3-VL favours each option, for answer distillation into Dynin-Omni.

One forward pass per video with exactly the input run_attention.py gave Qwen3-VL (same frames, sizes and prompt),
reading the logits of each option's letter at the position that emits the answer. The argmax letter should match the
answer Qwen3-VL gave in outputs/qwen3vl_8b/results.jsonl; the summary reports how often it does.

Usage:
    python qwen_answer_probs.py --split outputs/distillation/split.json

Appends to <out> (default outputs/distillation/qwen_answers.jsonl): category, video, letter_logits (one per option),
letter_mass (share of the full next-token probability on the option letters) and pred_letter. Resumes where it stopped.
"""
import argparse
import json
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from omnitrace.backends.qwen3vl import VIDEO_INFO_KEY, prepare_inputs_qwen3vl
from omnitrace.constants import QWEN3VL_MODEL_PATH, QWEN3VL_PIXELS_PER_TOKEN
from omnitrace.modalities.video import build_video_messages
from run_attention import CATEGORIES, LETTERS, build_prompt, find_video


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, help="split.json: only its train and validation videos")
    parser.add_argument("--limit", type=int, help="Only this many videos, picked at random (for a quick check)")
    parser.add_argument("--model", default=QWEN3VL_MODEL_PATH)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--saved", type=Path, default=Path("outputs/qwen3vl_8b/results.jsonl"),
                        help="Qwen3-VL's earlier answers, to check against")
    parser.add_argument("--out", type=Path, default=Path("outputs/distillation/qwen_answers.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    questions = {}
    for category in CATEGORIES:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})
    wanted = sorted(questions)
    if args.split:
        split = json.load(open(args.split))
        wanted = sorted(tuple(k) for k in split["train"] + split["val"])
    if args.limit:
        wanted = sorted(random.Random(args.seed).sample(wanted, min(args.limit, len(wanted))))
    saved = {}
    if args.saved.exists():
        for line in open(args.saved):
            row = json.loads(line)
            if "error" not in row:
                saved[(row["category"], row["video"])] = row.get("pred_letter")

    done = set()
    if args.out.exists():
        for line in open(args.out):
            row = json.loads(line)
            if "error" not in row:
                done.add((row["category"], row["video"]))
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(done & set(wanted))} already done, {len(todo)} to run")

    if todo:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype="auto", device_map=device,
                                                                attn_implementation="sdpa")
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model)
        bundle = SimpleNamespace(model=model, processor=processor)
        letter_ids = [processor.tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
        args.out.parent.mkdir(parents=True, exist_ok=True)

        for n, key in enumerate(todo, start=1):
            category, video = key
            item = questions[key]
            row = {"category": category, "video": video}
            start = time.time()
            try:
                messages = build_video_messages(str(find_video(args.data_dir, category, video)), build_prompt(item),
                                                video_max_pixels=args.max_vision_tokens * QWEN3VL_PIXELS_PER_TOKEN,
                                                video_fps=args.fps, video_max_frames=args.max_frames)
                inputs = prepare_inputs_qwen3vl(bundle, messages)
                inputs.pop(VIDEO_INFO_KEY)
                with torch.inference_mode():
                    logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
                options = len(item["candidates"])
                letter_logits = logits[letter_ids[:options]]
                row.update(letter_logits=[round(float(x), 4) for x in letter_logits],
                           letter_mass=float(torch.softmax(logits, -1)[letter_ids[:options]].sum()),
                           pred_letter=LETTERS[int(letter_logits.argmax())])
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - start, 2)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            if n % 100 == 0 or n == len(todo) or "error" in row:
                print(f"[{n}/{len(todo)}] {category}/{video}: {row.get('error') or row['pred_letter']} {row['seconds']}s",
                      flush=True)

    rows = {}
    for line in open(args.out):
        row = json.loads(line)
        if "error" not in row:
            rows[(row["category"], row["video"])] = row
    rows = {k: r for k, r in rows.items() if k in set(wanted)}
    checked = [k for k in rows if k in saved]
    agree = sum(rows[k]["pred_letter"] == saved[k] for k in checked)
    print(f"{len(rows)} of {len(wanted)} videos recorded; mean share of probability on the option letters "
          f"{sum(r['letter_mass'] for r in rows.values()) / max(len(rows), 1):.3f}; top letter matches Qwen3-VL's saved "
          f"answer on {agree} of {len(checked)}")


if __name__ == "__main__":
    main()
