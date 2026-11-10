"""
Ask Dynin-Omni for a step-by-step plan for DyBench questions, as TabAlign's diffusion planner writes one for tables.

Dynin-Omni sees each video as run_attention_dynin.py feeds it (5 frames, centre-cropped to 224x224), numbered 1 to 5
in the prompt, with the question and options. It is asked for a plan, not an answer: numbered steps, each ending in
[frames: ...] and [object: ...] tags, the video counterpart of TabAlign's [target: col], so that each step can later
become a mask over the video. Decoding uses Dynin-Omni's own video-to-text settings (128 tokens, 128 steps, one block,
lowest-confidence remasking, temperature 0). The reply is started with "1." (--prefill) so it begins a numbered list;
--no-options leaves the answer options out, and --text-only leaves the video out, as TabAlign's planner saw only a
summary of the table rather than the table itself.

This is for checking the plans by hand. It runs a sample of labelled videos and writes to <out-dir>:
    plans.jsonl   the prompt's answer as text, the parsed steps, and the frames (1-5) the label marks
    index.html    for each video the 5 frames as Dynin-Omni saw them (labelled frames outlined), the question,
                  the options and the plan; open it by double-clicking
and prints how many plans parse, how many steps they have, how many copy the example's frames, and how well the
planned frames cover the labelled ones.

Needs the Dynin-Omni environment (see README).
"""
import argparse
import html
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from run_attention_dynin import CATEGORIES, SPECIAL_TOKENS, crop_window, pick_frames
from run_attention_vidlada import LETTERS, sample_frames

INSTRUCTIONS = """The video shows 5 frames in order, numbered 1 to 5.
Do not answer the question yet. Write a short plan for answering it: 2 to 4 numbered steps, one per line.
End every step with the frames it needs and the object to look at, in this form:
<step number>. <what to do> [frames: <frame numbers>] [object: <object>]"""

EXAMPLE = """
Example, for a different video and the question "In which direction is the dog moving?":
1. Find where the dog is at the start [frames: 1, 2] [object: dog]
2. Find where the dog is at the end [frames: 4, 5] [object: dog]
3. Compare the two positions to get the direction [frames: 1, 5] [object: dog]"""
EXAMPLE_FRAMES = [[1, 2], [4, 5], [1, 5]]

STEP = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*?)\s*\[\s*frames?\s*:\s*([^\]]*)\]\s*\[\s*object\s*:\s*([^\]]*)\]", re.I)


def build_plan_prompt(item, example=True, options=True):
    lines = [item["question"]]
    if options:
        lines += ["Options:"] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(item["candidates"])]
    return "\n".join(lines) + "\n\n" + INSTRUCTIONS + (EXAMPLE if example else "")


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
            frames = parse_frames(match.group(3))
            steps.append({"text": match.group(2), "frames": frames, "object": match.group(4).strip()})
    return steps


def label_frames(label, picked):
    """Which of the 5 frames (1-5) fall in a time step the label marks with a box."""
    return [k + 1 for k, p in enumerate(picked) if label["boxes"].get(str(p // 2))]


def write_page(rows, out_dir):
    cards = []
    for row in rows:
        thumbs = "".join(
            f'<figure class="{"marked" if k + 1 in row.get("label_frames", []) else ""}">'
            f'<img src="frames/{html.escape(row["frames_dir"])}/{k + 1}.jpg" alt="frame {k + 1}"><figcaption>{k + 1}</figcaption></figure>'
            for k in range(5))
        options = "".join(f"<li{' class=answer' if c == row['answer'] else ''}>{LETTERS[i]}. {html.escape(c)}</li>"
                          for i, c in enumerate(row["candidates"]))
        steps = "".join(f"<tr><td>{s['text'] and html.escape(s['text'])}</td><td>{', '.join(map(str, s['frames']))}</td>"
                        f"<td>{html.escape(s['object'])}</td></tr>" for s in row.get("steps", []))
        table = (f"<table><tr><th>Step</th><th>Frames</th><th>Object</th></tr>{steps}</table>" if steps
                 else '<p class="warn">No step could be parsed.</p>')
        cards.append(f"""<section><h2>{html.escape(row['category'])} / {html.escape(row['video'])}</h2>
<div class="frames">{thumbs}</div><p class="note">Outlined: frames your label marks.</p>
<p><b>{html.escape(row['question'])}</b></p><ol class="options">{options}</ol>
<h3>Plan as written</h3><pre>{html.escape(row.get('plan') or row.get('error', ''))}</pre><h3>Parsed</h3>{table}</section>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dynin-Omni plans</title><style>
body{{font-family:system-ui,sans-serif;margin:0 auto;padding:16px;max-width:980px;background:#fcfcfb;color:#1a1a19}}
section{{border:1px solid #e3e2de;border-radius:8px;padding:12px 16px;margin:16px 0;background:#fff}}
h2{{font-size:15px;margin:4px 0 8px}} h3{{font-size:13px;margin:12px 0 4px;color:#52514e}}
.frames{{display:flex;gap:6px;flex-wrap:wrap}} figure{{margin:0;text-align:center;border:3px solid transparent;border-radius:4px}}
figure.marked{{border-color:#2a78d6}} img{{width:150px;height:150px;display:block}} figcaption{{font-size:12px;color:#52514e}}
.note{{font-size:12px;color:#898781;margin:4px 0}} .options li.answer{{font-weight:600}}
pre{{white-space:pre-wrap;background:#f4f3f0;padding:8px;border-radius:4px;font-size:13px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #e3e2de;padding:4px 8px;text-align:left}} .warn{{color:#b3261e}}
</style></head><body><h1>Dynin-Omni plans ({len(rows)} videos)</h1>{''.join(cards)}</body></html>"""
    (out_dir / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files to sample videos from")
    parser.add_argument("--per-category", type=int, default=10, help="Videos per category")
    parser.add_argument("--no-example", action="store_true", help="Leave the worked example out of the prompt")
    parser.add_argument("--no-options", action="store_true", help="Leave the answer options out of the prompt")
    parser.add_argument("--text-only", action="store_true",
                        help="Leave the video out, as TabAlign's planner saw only a summary of the table")
    parser.add_argument("--prefill", default="1.", help="Fixed start of the reply ('' for none)")
    parser.add_argument("--dynin-repo", type=Path, default=Path.home() / "Dynin-Omni")
    parser.add_argument("--model", default="snu-aidas/Dynin-Omni")
    parser.add_argument("--vq-model", default="snu-aidas/magvitv2")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plans"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--gen-length", type=int, default=128, help="Dynin-Omni's video-to-text default")
    parser.add_argument("--steps", type=int, default=128, help="Dynin-Omni's video-to-text default")
    parser.add_argument("--block-length", type=int, default=128, help="Dynin-Omni's video-to-text default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.dynin_repo))
    from PIL import Image
    from transformers import AutoTokenizer
    from models import DyninOmniModelLM, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform

    labels = {}
    for path in args.labels:
        for label in json.load(open(path))["labels"]:
            if label["done"] and not label["unsure"] and label.get("approved", True):
                labels[(label["category"], label["video"])] = label
    rng = random.Random(args.seed)
    wanted = []
    for category in CATEGORIES:
        pool = sorted(k for k in labels if k[0] == category)
        wanted += sorted(rng.sample(pool, min(args.per_category, len(pool))))
    questions = {}
    for category in {c for c, _ in wanted}:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plans_path = args.out_dir / "plans.jsonl"
    done = {}
    if plans_path.exists():
        for line in open(plans_path):
            row = json.loads(line)
            if "error" not in row:
                done[(row["category"], row["video"])] = row
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(done)} already done, {len(todo)} to run; example {not args.no_example}, "
          f"options {not args.no_options}, video {not args.text_only}, reply starts with {args.prefill!r}")

    if todo:
        device = args.device
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
        uni_prompting = UniversalPrompting(tokenizer, max_text_len=2048, special_tokens=SPECIAL_TOKENS, ignore_id=-100,
                                           cond_dropout_prob=0.0, use_reserved_token=True)
        vq_model = MAGVITv2.from_pretrained(args.vq_model).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        model = DyninOmniModelLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        model.eval()
        special = {name: torch.tensor([[int(uni_prompting.sptids_dict[name])]], device=device)
                   for name in ("<|v2t|>", "<|soi|>", "<|eoi|>", "<|sot|>")}
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        for n, (category, video) in enumerate(todo, start=1):
            item = questions[(category, video)]
            row = {"category": category, "video": video, "question": item["question"], "candidates": item["candidates"],
                   "answer": item["answer"], "frames_dir": f"{category}_{Path(video).stem}"}
            start = time.time()
            try:
                path = args.data_dir / "video" / category / category / video
                frames, _, _ = sample_frames(path)
                picked = pick_frames(len(frames), args.frames)
                if 2 * labels[(category, video)]["grid"][0] != len(frames):
                    raise RuntimeError(f"{len(frames)} frames read, but the label has {labels[(category, video)]['grid'][0]} time steps")
                row["label_frames"] = label_frames(labels[(category, video)], picked)

                height, width = frames[0].shape[:2]
                x0, y0, x1, y1 = crop_window(width, height, args.resolution)
                thumbs = args.out_dir / "frames" / row["frames_dir"]
                thumbs.mkdir(parents=True, exist_ok=True)
                for k, p in enumerate(picked):
                    crop = frames[p][int(round(y0 * height)):int(round(y1 * height)), int(round(x0 * width)):int(round(x1 * width))]
                    cv2.imwrite(str(thumbs / f"{k + 1}.jpg"), cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_RGB2BGR))

                pixels = torch.stack([image_transform(Image.fromarray(frames[p]), resolution=args.resolution)
                                      for p in picked]).to(device)
                prompt = ("<|start_header_id|>user<|end_header_id|>\n"
                          + build_plan_prompt(item, not args.no_example, not args.no_options)
                          + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n" + args.prefill)
                query_ids = torch.tensor(uni_prompting.text_tokenizer([prompt])["input_ids"], device=device)
                with torch.inference_mode():
                    if args.text_only:
                        input_ids = query_ids
                    else:
                        video_tokens = (vq_model.get_code(pixels) + len(uni_prompting.text_tokenizer)).view(1, -1)
                        input_ids = torch.cat([special["<|v2t|>"], special["<|soi|>"], video_tokens, special["<|eoi|>"],
                                               special["<|sot|>"], query_ids], dim=1)
                    output = model.mmu_generate(input_ids, max_new_tokens=args.gen_length, steps=args.steps,
                                                block_length=args.block_length)
                generated = output[0, input_ids.shape[1]:].tolist()
                if eot_id in generated:
                    generated = generated[:generated.index(eot_id)]
                row["plan"] = (args.prefill + uni_prompting.text_tokenizer.decode(generated, skip_special_tokens=True)).strip()
                row["steps"] = parse_plan(row["plan"])
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - start, 2)
            with open(plans_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[(category, video)] = row
            status = row.get("error") or f"{len(row['steps'])} steps parsed"
            print(f"[{n}/{len(todo)}] {category}/{video}: {status} ({row['seconds']}s)", flush=True)

    rows = [done[k] for k in wanted if k in done]
    write_page(rows, args.out_dir)
    ok = [r for r in rows if "error" not in r]
    parsed = [r for r in ok if r["steps"]]
    copied = sum([s["frames"] for s in r["steps"]] == EXAMPLE_FRAMES for r in parsed)
    recall = [len(set(r["label_frames"]) & {f for s in r["steps"] for f in s["frames"]}) / len(r["label_frames"])
              for r in parsed if r["label_frames"]]
    print(f"\n{len(parsed)} of {len(ok)} plans have at least one parsed step; "
          f"steps per plan {np.mean([len(r['steps']) for r in parsed]) if parsed else 0:.1f}; "
          f"{copied} copy the example's frames exactly; planned frames cover "
          f"{100 * np.mean(recall) if recall else 0:.0f}% of the labelled frames on average")
    print(f"Plans: {plans_path}\nReview page: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
