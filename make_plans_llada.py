"""
Ask LLaDA-8B-Instruct, TabAlign's planner, for a step-by-step plan for DyBench questions (TabAlign step 1).

Like TabAlign's planner, LLaDA-8B-Instruct never sees the raw input: it gets the question, the options and that the
video has 5 frames, and writes numbered steps, each ending in [frames: ...] and [object: ...] tags, the video counterpart
of TabAlign's [target: col]. A worked example in the prompt shows the format, and the reply is started with "1." so it
begins a numbered list. Decoding follows TabAlign's planner: 128 denoising steps, blocks of 32, greedy,
lowest-confidence remasking.

It runs a sample of labelled videos for checking by hand and writes to <out-dir>:
    plans.jsonl   the plan as written, the parsed steps, and the frames (1-5) the label marks
    index.html    for each video its 5 frames (labelled frames outlined), the question, the options and the plan
and prints how many plans parse, how many copy the example, and how well the planned frames cover the labelled ones.
The 5 frames are the ones Dynin-Omni sees (picked evenly from Qwen3-VL's frames, centre-cropped).

Runs in the Dynin-Omni environment; the model (about 16 GB) downloads on first use.
"""
import argparse
import html
import json
import random
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from run_attention_dynin import CATEGORIES, crop_window, pick_frames
from run_attention_vidlada import LETTERS, sample_frames

MASK_ID = 126336

INSTRUCTIONS = """The video shows 5 frames in order, numbered 1 to 5.
Do not answer the question yet. Write a short plan for answering it: 2 to 4 numbered steps, one per line.
End every step with the frames it needs and the object to look at, in this form:
<step number>. <what to do> [frames: <frame numbers>] [object: <object>]

Example, for a different video and the question "In which direction is the dog moving?":
1. Find where the dog is at the start [frames: 1, 2] [object: dog]
2. Find where the dog is at the end [frames: 4, 5] [object: dog]
3. Compare the two positions to get the direction [frames: 1, 5] [object: dog]"""
EXAMPLE_FRAMES = [[1, 2], [4, 5], [1, 5]]

STEP = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*?)\s*\[\s*frames?\s*:\s*([^\]]*)\]\s*\[\s*object\s*:\s*([^\]]*)\]", re.I)


def build_plan_prompt(item):
    lines = [item["question"], "Options:"] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(item["candidates"])]
    return "\n".join(lines) + "\n\n" + INSTRUCTIONS


def parse_frames(text):
    """"1, 2", "1 and 5" or "3-5" -> sorted frame numbers, keeping only 1-5."""
    frames = set()
    for start, end in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", text):
        frames.update(range(int(start), int(end or start) + 1))
    return sorted(f for f in frames if 1 <= f <= 5)


def parse_plan(text):
    """Numbered steps with both tags, as {"text", "frames", "object"}; frames outside 1-5 are dropped."""
    steps = []
    for line in text.splitlines():
        match = STEP.match(line)
        if match:
            steps.append({"text": match.group(2), "frames": parse_frames(match.group(3)), "object": match.group(4).strip()})
    return steps


def sample_videos(label_paths, per_category, seed, data_dir):
    """A fixed sample of finished, approved labelled videos: (videos, labels, questions)."""
    labels = {}
    for path in label_paths:
        for label in json.load(open(path))["labels"]:
            if label["done"] and not label["unsure"] and label.get("approved", True):
                labels[(label["category"], label["video"])] = label
    rng = random.Random(seed)
    wanted = []
    for category in CATEGORIES:
        pool = sorted(k for k in labels if k[0] == category)
        wanted += sorted(rng.sample(pool, min(per_category, len(pool))))
    questions = {}
    for category in {c for c, _ in wanted}:
        questions.update({(category, q["video"]): q for q in json.load(open(data_dir / "json" / f"{category}.json"))})
    return wanted, labels, questions


def save_frames(key, label, data_dir, thumbs_dir, count=5, resolution=224):
    """Save the picked frames, centre-cropped as Dynin-Omni sees them; return which of them (1-5) the label marks."""
    category, video = key
    frames, _, _ = sample_frames(data_dir / "video" / category / category / video)
    if 2 * label["grid"][0] != len(frames):
        raise RuntimeError(f"{len(frames)} frames read, but the label has {label['grid'][0]} time steps")
    picked = pick_frames(len(frames), count)
    height, width = frames[0].shape[:2]
    x0, y0, x1, y1 = crop_window(width, height, resolution)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(picked):
        crop = frames[p][int(round(y0 * height)):int(round(y1 * height)), int(round(x0 * width)):int(round(x1 * width))]
        cv2.imwrite(str(thumbs_dir / f"{k + 1}.jpg"), cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_RGB2BGR))
    return [k + 1 for k, p in enumerate(picked) if label["boxes"].get(str(p // 2))]


def load_done(path):
    done = {}
    if path.exists():
        for line in open(path):
            row = json.loads(line)
            if "error" not in row:
                done[(row["category"], row["video"])] = row
    return done


def write_page(rows, out_dir):
    cards = []
    for row in rows:
        thumbs = "".join(
            f'<figure class="{"marked" if k + 1 in row.get("label_frames", []) else ""}">'
            f'<img src="frames/{html.escape(row["frames_dir"])}/{k + 1}.jpg" alt="frame {k + 1}"><figcaption>{k + 1}</figcaption></figure>'
            for k in range(5))
        options = "".join(f"<li{' class=answer' if c == row['answer'] else ''}>{LETTERS[i]}. {html.escape(c)}</li>"
                          for i, c in enumerate(row["candidates"]))
        steps = "".join(f"<tr><td>{html.escape(s['text'])}</td><td>{', '.join(map(str, s['frames']))}</td>"
                        f"<td>{html.escape(s['object'])}</td></tr>" for s in row.get("steps", []))
        table = (f"<table><tr><th>Step</th><th>Frames</th><th>Object</th></tr>{steps}</table>" if steps
                 else '<p class="warn">No step could be parsed.</p>')
        cards.append(f"""<section><h2>{html.escape(row['category'])} / {html.escape(row['video'])}</h2>
<div class="frames">{thumbs}</div><p class="note">Outlined: frames your label marks.</p>
<p><b>{html.escape(row['question'])}</b></p><ol class="options">{options}</ol>
<h3>Plan as written</h3><pre>{html.escape(row.get('plan') or row.get('error', ''))}</pre><h3>Parsed</h3>{table}</section>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLaDA-8B-Instruct plans</title><style>
body{{font-family:system-ui,sans-serif;margin:0 auto;padding:16px;max-width:980px;background:#fcfcfb;color:#1a1a19}}
section{{border:1px solid #e3e2de;border-radius:8px;padding:12px 16px;margin:16px 0;background:#fff}}
h2{{font-size:15px;margin:4px 0 8px}} h3{{font-size:13px;margin:12px 0 4px;color:#52514e}}
.frames{{display:flex;gap:6px;flex-wrap:wrap}} figure{{margin:0;text-align:center;border:3px solid transparent;border-radius:4px}}
figure.marked{{border-color:#2a78d6}} img{{width:150px;height:150px;display:block}} figcaption{{font-size:12px;color:#52514e}}
.note{{font-size:12px;color:#898781;margin:4px 0}} .options li.answer{{font-weight:600}}
pre{{white-space:pre-wrap;background:#f4f3f0;padding:8px;border-radius:4px;font-size:13px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #e3e2de;padding:4px 8px;text-align:left}} .warn{{color:#b3261e}}
</style></head><body><h1>LLaDA-8B-Instruct plans ({len(rows)} videos)</h1>{''.join(cards)}</body></html>"""
    (out_dir / "index.html").write_text(page)


@torch.no_grad()
def generate(model, prompt_ids, gen_length, steps, block_length):
    """LLaDA's sampler: start from all-masked answer slots and, block by block, fix the most confident ones."""
    x = torch.full((1, prompt_ids.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=prompt_ids.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids
    blocks = gen_length // block_length
    per_block = steps // blocks
    for b in range(blocks):
        start, end = prompt_ids.shape[1] + b * block_length, prompt_ids.shape[1] + (b + 1) * block_length
        masked = int((x[0, start:end] == MASK_ID).sum())
        counts = [masked // per_block + (i < masked % per_block) for i in range(per_block)]
        for count in counts:
            is_masked = x == MASK_ID
            logits = model(x).logits
            guess = logits.argmax(dim=-1)
            confidence = torch.softmax(logits.float(), dim=-1).gather(-1, guess[..., None])[..., 0]
            confidence[:, end:] = -float("inf")
            confidence = torch.where(is_masked, confidence, torch.full_like(confidence, -float("inf")))
            chosen = confidence[0].topk(count).indices
            x[0, chosen] = guess[0, chosen]
    return x[:, prompt_ids.shape[1]:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files to sample videos from")
    parser.add_argument("--per-category", type=int, default=10, help="Videos per category")
    parser.add_argument("--prefill", default="1.", help="Fixed start of the reply ('' for none)")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plans_llada"))
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128, help="TabAlign's planner setting")
    parser.add_argument("--block-length", type=int, default=32, help="TabAlign's planner setting")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    wanted, labels, questions = sample_videos(args.labels, args.per_category, args.seed, args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plans_path = args.out_dir / "plans.jsonl"
    done = load_done(plans_path)
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(done)} already done, {len(todo)} to run")

    if todo:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device)
        model.eval()
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        for n, key in enumerate(todo, start=1):
            item = questions[key]
            row = {"category": key[0], "video": key[1], "question": item["question"], "candidates": item["candidates"],
                   "answer": item["answer"], "frames_dir": f"{key[0]}_{Path(key[1]).stem}"}
            start = time.time()
            try:
                row["label_frames"] = save_frames(key, labels[key], args.data_dir, args.out_dir / "frames" / row["frames_dir"])
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": build_plan_prompt(item)}],
                                                       add_generation_prompt=True, tokenize=False) + args.prefill
                prompt_ids = torch.tensor(tokenizer(prompt)["input_ids"], device=args.device)[None]
                generated = generate(model, prompt_ids, args.gen_length, args.steps, args.block_length)[0].tolist()
                if eot_id in generated:
                    generated = generated[:generated.index(eot_id)]
                row["plan"] = (args.prefill + tokenizer.decode(generated, skip_special_tokens=True)).strip()
                row["steps"] = parse_plan(row["plan"])
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - start, 2)
            with open(plans_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[key] = row
            status = row.get("error") or f"{len(row['steps'])} steps parsed"
            print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: {status} ({row['seconds']}s)", flush=True)

    rows = [done[k] for k in wanted if k in done]
    write_page(rows, args.out_dir)
    parsed = [r for r in rows if r.get("steps")]
    copied = sum([s["frames"] for s in r["steps"]] == EXAMPLE_FRAMES for r in parsed)
    recall = [len(set(r["label_frames"]) & {f for s in r["steps"] for f in s["frames"]}) / len(r["label_frames"])
              for r in parsed if r["label_frames"]]
    print(f"\n{len(parsed)} of {len(rows)} plans have at least one parsed step; "
          f"steps per plan {np.mean([len(r['steps']) for r in parsed]) if parsed else 0:.1f}; {copied} copy the example's "
          f"frames exactly; planned frames cover {100 * np.mean(recall) if recall else 0:.0f}% of the labelled frames on average")
    print(f"Plans: {plans_path}\nReview page: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
