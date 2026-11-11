"""
Have Qwen3-VL carry out LLaDA-8B-Instruct's plans step by step on the video (TabAlign's stepwise execution).

Qwen3-VL gets the video exactly as run_attention.py gave it (same frames, sizes and prompt format), then the question and
options, then the plan's steps one at a time as a conversation: each step's text, its object, and the times in the video
of the frames it names (the plan numbers the 5 frames Dynin-Omni sees, which are among Qwen3-VL's frames). After the
last step it answers the question with a letter. Decoding is greedy. The step-by-step answers are the reasoning trace
Dynin-Omni will later learn from.

For checking by hand it writes to <out-dir>:
    traces.jsonl   per video: each step's prompt and Qwen3-VL's answer, the final answer, and whether it is right
    index.html     the plan's frames (from the planner's page), the question, the trace and the final answer
and prints the accuracy of the stepwise answers next to Qwen3-VL's direct answers on the same videos.

Usage (in the main environment, after make_plans_llada.py):
    python execute_plans_qwen.py
"""
import argparse
import html
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from make_plans_llada import load_done
from omnitrace.backends.qwen3vl import VIDEO_INFO_KEY, prepare_inputs_qwen3vl
from omnitrace.constants import QWEN3VL_MODEL_PATH, QWEN3VL_PIXELS_PER_TOKEN
from omnitrace.modalities.video import build_video_messages
from run_attention import LETTERS, build_prompt, find_video, parse_letter
from run_attention_dynin import pick_frames
from run_attention_vidlada import sample_frames

INTRO = "Work through the following steps one at a time before answering. Answer each step in one sentence."
FINAL = "Based on your answers to the steps, which option is correct? Answer with the option's letter from the given choices directly."


def step_prompt(number, step, times):
    when = " and ".join(f"{t:.1f}s" for t in times) if times else "the whole video"
    return f"Step {number}: {step['text'].rstrip('.')}. Look at the {step['object']} around {when}."


def write_page(rows, out_dir, plans_dir):
    cards = []
    for row in rows:
        thumbs = "".join(f'<figure><img src="{html.escape(str(plans_dir))}/frames/{html.escape(row["frames_dir"])}/{k + 1}.jpg" '
                         f'alt="frame {k + 1}"><figcaption>{k + 1} ({t:.1f}s)</figcaption></figure>'
                         for k, t in enumerate(row.get("frame_times", [])))
        options = "".join(f"<li{' class=answer' if c == row['answer'] else ''}>{LETTERS[i]}. {html.escape(c)}</li>"
                          for i, c in enumerate(row["candidates"]))
        turns = "".join(f"<tr><td>{html.escape(t['prompt'])}</td><td>{html.escape(t['reply'])}</td></tr>"
                        for t in row.get("turns", []))
        verdict = ("right" if row.get("correct") else "wrong") if "final" in row else ""
        direct = {True: "right", False: "wrong"}.get(row.get("direct_correct"), "not available")
        cards.append(f"""<section><h2>{html.escape(row['category'])} / {html.escape(row['video'])}</h2>
<div class="frames">{thumbs}</div><p><b>{html.escape(row['question'])}</b></p><ol class="options">{options}</ol>
<table><tr><th>Step</th><th>Qwen3-VL</th></tr>{turns}</table>
<p class="final">Final answer: <b>{html.escape(row.get('final') or row.get('error', ''))}</b> ({verdict}). Answering directly, without the
plan: {direct}.</p></section>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-VL stepwise execution</title><style>
body{{font-family:system-ui,sans-serif;margin:0 auto;padding:16px;max-width:980px;background:#fcfcfb;color:#1a1a19}}
section{{border:1px solid #e3e2de;border-radius:8px;padding:12px 16px;margin:16px 0;background:#fff}}
h2{{font-size:15px;margin:4px 0 8px}} .frames{{display:flex;gap:6px;flex-wrap:wrap}} figure{{margin:0;text-align:center}}
img{{width:150px;height:150px;display:block}} figcaption{{font-size:12px;color:#52514e}} .options li.answer{{font-weight:600}}
table{{border-collapse:collapse;font-size:13px;width:100%}} td,th{{border:1px solid #e3e2de;padding:4px 8px;text-align:left;vertical-align:top}}
td:first-child{{width:45%;color:#52514e}} .final{{font-size:14px}}
</style></head><body><h1>Qwen3-VL stepwise execution ({len(rows)} videos)</h1>{''.join(cards)}</body></html>"""
    (out_dir / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", type=Path, default=Path("outputs/plans_llada/plans.jsonl"))
    parser.add_argument("--direct", type=Path, default=Path("outputs/qwen3vl_8b/results.jsonl"),
                        help="Qwen3-VL's direct answers, for comparison")
    parser.add_argument("--model", default=QWEN3VL_MODEL_PATH)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    parser.add_argument("--step-tokens", type=int, default=64, help="Most tokens for each step's answer")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/qwen_execution"))
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    plans = [r for r in load_done(args.plans).values() if r.get("steps")]
    direct = {(r["category"], r["video"]): r["correct"] for r in load_done(args.direct).values()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = args.out_dir / "traces.jsonl"
    done = load_done(traces_path)
    todo = [p for p in plans if (p["category"], p["video"]) not in done]
    print(f"{len(plans)} plans, {len(done)} already done, {len(todo)} to run")

    if todo:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype="auto", device_map=device,
                                                                attn_implementation="sdpa")
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model)
        bundle = SimpleNamespace(model=model, processor=processor)

        def reply(messages, max_new_tokens):
            inputs = prepare_inputs_qwen3vl(bundle, messages)
            inputs.pop(VIDEO_INFO_KEY)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        for n, plan in enumerate(todo, start=1):
            key = (plan["category"], plan["video"])
            row = {k: plan[k] for k in ("category", "video", "question", "candidates", "answer", "frames_dir")}
            row["direct_correct"] = direct.get(key)
            start = time.time()
            try:
                path = find_video(args.data_dir, *key)
                frames, indices, native_fps = sample_frames(path)
                picked = pick_frames(len(frames), 5)
                row["frame_times"] = [indices[p] / native_fps for p in picked]
                item = {"question": plan["question"], "candidates": plan["candidates"]}
                messages = build_video_messages(str(path), "", video_max_pixels=args.max_vision_tokens * QWEN3VL_PIXELS_PER_TOKEN,
                                                video_fps=args.fps, video_max_frames=args.max_frames)
                first = True
                row["turns"] = []
                for number, step in enumerate(plan["steps"], start=1):
                    prompt = step_prompt(number, step, [row["frame_times"][f - 1] for f in step["frames"]])
                    if first:
                        messages[0]["content"][1]["text"] = build_prompt(item).rsplit("\n", 1)[0] + "\n\n" + INTRO + "\n\n" + prompt
                        first = False
                    else:
                        messages.append({"role": "user", "content": prompt})
                    answer = reply(messages, args.step_tokens)
                    messages.append({"role": "assistant", "content": answer})
                    row["turns"].append({"prompt": prompt, "reply": answer})
                messages.append({"role": "user", "content": FINAL})
                row["final"] = reply(messages, 8)
                letter = parse_letter(row["final"], plan["candidates"])
                row["pred_letter"] = letter
                row["correct"] = letter == LETTERS[plan["candidates"].index(plan["answer"])]
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - start, 2)
            with open(traces_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[key] = row
            status = row.get("error") or f"{len(row['turns'])} steps, final {row['final']!r} ({'right' if row['correct'] else 'wrong'})"
            print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: {status} ({row['seconds']}s)", flush=True)

    rows = [done[(p["category"], p["video"])] for p in plans if (p["category"], p["video"]) in done]
    write_page(rows, args.out_dir, Path("..") / args.plans.parent.name)
    finished = [r for r in rows if "final" in r]
    compared = [r for r in finished if r.get("direct_correct") is not None]
    gained = sum(r["correct"] and not r["direct_correct"] for r in compared)
    lost = sum(r["direct_correct"] and not r["correct"] for r in compared)
    print(f"\n{len(finished)} of {len(rows)} traces finished. Stepwise accuracy {sum(r['correct'] for r in finished)} / "
          f"{len(finished)}; answering directly {sum(r['direct_correct'] for r in compared)} / {len(compared)} on the same "
          f"videos ({gained} now right that were wrong, {lost} now wrong that were right)")
    print(f"Traces: {traces_path}\nReview page: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
