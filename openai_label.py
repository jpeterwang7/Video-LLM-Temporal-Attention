"""
Label DyBench videos with the OpenAI API, the way the hand-labelled videos were labelled.

Each request sends the labelling rules (labels/labeling_rules.md), then a few hand-labelled videos of the same
category as worked examples (the video's time steps, then the hand label as the answer: the steps picked and
the boxes), then the new video, and asks for its label in the same form. Frames, time steps and frame sizes
are exactly those of the labelling page; each step is sent as one image with its two frames side by side.

Answers are saved as they arrive in <out-dir>/responses.jsonl, with token counts, so a stopped run resumes
where it left off. Every run then rebuilds:
    <out-dir>/review/index.html   the labelling page, pre-filled with the API's labels, with an "Approved"
                                  tick per video: check each label, fix it if needed, tick Approved, and
                                  "Download labels" (the file records which videos you approved)
    <out-dir>/labels_openai.json  the API's labels as they came back, in the labelling page's format

Needs `pip install openai` and OPENAI_API_KEY set in the environment. Start small:
    python openai_label.py --list-models              # the models this key can use
    python openai_label.py --model MODEL --dry-run     # build one request without sending it
    python openai_label.py --model MODEL --limit 20    # label 20 videos, then review them
"""
import argparse
import base64
import json
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from make_labeling_page import build_item, write_page

CATEGORIES = ["moving_direction", "reversible_dynamics"]
LETTERS = "ABCDEFGH"
# What the API must return (strict JSON schema): steps are numbered from 1, boxes are fractions of the left frame
SCHEMA = {
    "type": "object",
    "properties": {
        "needed_steps": {"type": "array", "items": {"type": "integer"}},
        "boxes": {"type": "array", "items": {
            "type": "object",
            "properties": {"step": {"type": "integer"}, "x0": {"type": "number"}, "y0": {"type": "number"},
                           "x1": {"type": "number"}, "y1": {"type": "number"}},
            "required": ["step", "x0", "y0", "x1", "y1"], "additionalProperties": False}},
        "unsure": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["needed_steps", "boxes", "unsure", "note"],
    "additionalProperties": False,
}


def step_image(page_dir, step, frames):
    """One time step as a JPEG data URL: the step's frames side by side (frames="pair") or the first alone."""
    images = [cv2.imread(str(page_dir / rel)) for rel in (step["frames"] if frames == "pair" else step["frames"][:1])]
    if len(images) == 2:
        gap = np.full((images[0].shape[0], 8, 3), 255, dtype=np.uint8)
        images = [images[0], gap, images[1]]
    ok, data = cv2.imencode(".jpg", np.hstack(images), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode()


def video_content(item, page_dir, frames, detail):
    """The user message for one video: question, options, answer, then one image per time step."""
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["candidates"]))
    layout = ("Each image is one time step: left half = first frame (box coordinates refer to this frame), "
              "right half = second frame." if frames == "pair" else "Each image is the first frame of one time step.")
    content = [{"type": "text", "text": f"Category: {item['title']}\nQuestion: {item['question']}\nOptions:\n{options}\n"
                                        f"Correct answer: {item['answer']}\nThe video has {len(item['steps'])} time steps. {layout}"}]
    for step in item["steps"]:
        content.append({"type": "text", "text": f"Step {step['t'] + 1} ({step['time']:.1f} s):"})
        content.append({"type": "image_url", "image_url": {"url": step_image(page_dir, step, frames), "detail": detail}})
    return content


def as_answer(label):
    """A hand label in the form the API is asked to return (steps from 1)."""
    boxes = [{"step": int(t) + 1, "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}
             for t, bs in sorted(label["boxes"].items(), key=lambda kv: int(kv[0])) for b in bs]
    return json.dumps({"needed_steps": [t + 1 for t in label["needed_steps"]], "boxes": boxes,
                       "unsure": label["unsure"], "note": label["note"]})


def cells_for(item, boxes):
    """Token-grid cells a set of boxes covers (the labelling page's cellsFor)."""
    (h, w), p = item["frame_size"], item["patch"]
    rows, cols, out = h // p, w // p, set()
    for x0, y0, x1, y1 in boxes:
        for r in range(int(y0 * h // p), min(rows - 1, int((y1 * h - 1e-6) // p)) + 1):
            for c in range(int(x0 * w // p), min(cols - 1, int((x1 * w - 1e-6) // p)) + 1):
                out.add((r, c))
    return sorted([list(x) for x in out])


def to_label(item, answer, source):
    """The API's answer as a labelling-page label. Boxes are clipped to the frame and put in order; a step with a
    box counts as picked; steps outside the video are dropped."""
    steps = len(item["steps"])
    boxes = {}
    for b in answer["boxes"]:
        t = b["step"] - 1
        x0, x1 = sorted(min(max(v, 0.0), 1.0) for v in (b["x0"], b["x1"]))
        y0, y1 = sorted(min(max(v, 0.0), 1.0) for v in (b["y0"], b["y1"]))
        if 0 <= t < steps and x1 - x0 > 0.005 and y1 - y0 > 0.005:
            boxes.setdefault(str(t), []).append([round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)])
    needed = sorted({s - 1 for s in answer["needed_steps"] if 1 <= s <= steps} | {int(t) for t in boxes})
    return {"id": item["id"], "category": item["category"], "video": item["video"], "grid": item["grid"],
            "frame_size": item["frame_size"], "needed_steps": needed, "boxes": boxes,
            "cells": {str(t): cells_for(item, boxes.get(str(t), [])) for t in needed}, "unsure": answer["unsure"],
            "note": answer["note"], "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "done": answer["unsure"] or (bool(needed) and all(str(t) in boxes for t in needed)),
            "approved": False, "source": source}


def ask(client, model, messages):
    response = client.chat.completions.create(
        model=model, messages=messages,
        response_format={"type": "json_schema", "json_schema": {"name": "attention_label", "strict": True, "schema": SCHEMA}})
    usage = response.usage
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return response.choices[0].message.content, {"prompt": usage.prompt_tokens, "cached": cached,
                                                  "completion": usage.completion_tokens}


def rebuild(out_dir, responses, model_note):
    """Review page and labels file from every answer so far."""
    done = [r for r in responses.values() if "error" not in r]
    items = [r["item"] for r in done]
    labels = {r["item"]["id"]: r["label"] for r in done}
    prefill = {i: {"needed": l["needed_steps"], "boxes": l["boxes"], "unsure": l["unsure"], "note": l["note"]}
               for i, l in labels.items()}
    write_page(items, out_dir / "review", prefill=prefill, review=True)
    json.dump({"annotator": model_note, "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "videos": len(labels), "labels": list(labels.values())}, open(out_dir / "labels_openai.json", "w"), indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="OpenAI model (see --list-models)")
    parser.add_argument("--list-models", action="store_true", help="Print the models this API key can use, then stop")
    parser.add_argument("--examples", type=Path, nargs="+",
                        default=[Path("labels/labels_peter.json"), Path("labels/labels_peter_labeling_round2.json")],
                        help="Hand labels: worked examples, and videos not to label again")
    parser.add_argument("--rules", type=Path, default=Path("labels/labeling_rules.md"))
    parser.add_argument("--per-request", type=int, default=2, help="Worked examples in each request (same category)")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument("--limit", type=int, help="Label at most this many new videos in this run")
    parser.add_argument("--frames", choices=["pair", "first"], default="pair",
                        help="Send both frames of each step side by side (default), or only the first")
    parser.add_argument("--detail", choices=["high", "low", "auto"], default="high", help="Image detail for the API")
    parser.add_argument("--dry-run", action="store_true", help="Build the first request and save it; send nothing")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/openai_labels"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(max_retries=5)
        if args.list_models:
            print("\n".join(sorted(m.id for m in client.models.list())))
            return
    if not args.model and not args.dry_run:
        parser.error("--model is required (run --list-models to see the choices)")
    model = args.model or "dry-run"

    frame_args = SimpleNamespace(fps=2.0, max_frames=32, max_vision_tokens=3072)    # the labelling page's defaults
    questions = {}
    for category in CATEGORIES:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})
    hand = {}
    for path in args.examples:
        for label in json.load(open(path))["labels"]:
            hand[(label["category"], label["video"])] = label

    # Worked examples: a fixed set per category, so every request starts the same way (the API can then reuse it)
    rng = random.Random(args.seed)
    examples_dir = args.out_dir / "examples"
    examples = {}
    for category in args.categories:
        usable = sorted(k for k, l in hand.items() if k[0] == category and l["done"] and not l["unsure"])
        picked = rng.sample(usable, min(args.per_request, len(usable)))
        if not picked:
            raise SystemExit(f"No hand-labelled {category} videos to use as examples")
        examples[category] = [(build_item(c, v, questions[(c, v)], args.data_dir, examples_dir, frame_args, copy_video=False),
                               hand[(c, v)]) for c, v in picked]
    rules = args.rules.read_text().split("---", 1)[-1].strip()

    def messages_for(item, page_dir):
        messages = [{"role": "system", "content": rules}]
        for example_item, label in examples[item["category"]]:
            messages.append({"role": "user", "content": video_content(example_item, examples_dir, args.frames, args.detail)})
            messages.append({"role": "assistant", "content": as_answer(label)})
        content = video_content(item, page_dir, args.frames, args.detail)
        content[0]["text"] += "\nLabel this video the way the examples are labelled."
        messages.append({"role": "user", "content": content})
        return messages

    args.out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.out_dir / "review"
    responses_path = args.out_dir / "responses.jsonl"
    responses = {}
    if responses_path.exists():
        for line in open(responses_path):
            row = json.loads(line)
            responses[(row["category"], row["video"])] = row
    todo = [k for k in questions if k[0] in args.categories and k not in hand
            and ("error" in responses.get(k, {"error": 1}))]
    random.Random(args.seed).shuffle(todo)
    todo = todo[:args.limit] if args.limit else todo
    print(f"{len(todo)} videos to label with {model}; {sum('error' not in r for r in responses.values())} already labelled; "
          f"{args.per_request} worked examples per request")

    if args.dry_run:
        category, video = todo[0]
        item = build_item(category, video, questions[(category, video)], args.data_dir, review_dir, frame_args, copy_video=False)
        messages = messages_for(item, review_dir)
        images = sum(1 for m in messages if isinstance(m["content"], list) for c in m["content"] if c["type"] == "image_url")
        dry = args.out_dir / "dry_run"
        dry.mkdir(exist_ok=True)
        n = 0
        for m in messages:
            for c in (m["content"] if isinstance(m["content"], list) else []):
                if c["type"] == "image_url":
                    (dry / f"image_{n:03d}.jpg").write_bytes(base64.b64decode(c["image_url"]["url"].split(",", 1)[1]))
                    c["image_url"]["url"] = f"image_{n:03d}.jpg"
                    n += 1
        json.dump(messages, open(dry / "messages.json", "w"), indent=1)
        print(f"One request for {category}/{video}: {len(messages)} messages, {images} images. Saved to {dry}")
        return

    totals = {"prompt": 0, "cached": 0, "completion": 0}
    try:
        for n, (category, video) in enumerate(todo, start=1):
            start = time.time()
            row = {"category": category, "video": video, "model": model}
            try:
                item = build_item(category, video, questions[(category, video)], args.data_dir, review_dir, frame_args,
                                  copy_video=False)
                text, usage = ask(client, model, messages_for(item, review_dir))
                label = to_label(item, json.loads(text), f"openai:{model}")
                row.update(item=item, label=label, raw=text, usage=usage)
                for k in totals:
                    totals[k] += usage[k]
                status = (f"{len(label['needed_steps'])}/{len(item['steps'])} steps, "
                          f"{sum(len(b) for b in label['boxes'].values())} boxes" + (", unsure" if label["unsure"] else ""))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                status = row["error"]
            responses[(category, video)] = row
            with open(responses_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{n}/{len(todo)}] {category}/{video}: {status} ({time.time() - start:.1f}s)", flush=True)
    finally:
        rebuild(args.out_dir, responses, f"openai:{model}")
        print(f"\nTokens this run: {totals['prompt']} prompt ({totals['cached']} of them cached), "
              f"{totals['completion']} completion")
        print(f"Review page: {review_dir / 'index.html'}\nLabels: {args.out_dir / 'labels_openai.json'}")


if __name__ == "__main__":
    main()
