"""
Ask LLaDA-8B-Instruct, TabAlign's planner, for a step-by-step plan for DyBench questions.

Like TabAlign's planner, LLaDA-8B-Instruct never sees the raw input: it gets the question, the options and that the
video has 5 frames, and with --captions also a one-sentence description of each frame (from caption_frames_dynin.py),
the video counterpart of TabAlign's compact table metadata. The prompt and the [frames: ...] [object: ...] plan format
are make_plans_dynin.py's, so the plans compare directly; the reply is started with "1." (--prefill). Decoding follows
TabAlign's planner: 128 denoising steps, blocks of 32, greedy, lowest-confidence remasking.

Writes plans.jsonl and index.html to <out-dir>, as make_plans_dynin.py does, for the same sample of labelled videos.
Runs in the Dynin-Omni environment; the model (about 16 GB) downloads on first use.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from make_plans_dynin import build_plan_prompt, load_done, load_frames, parse_plan, report, sample_videos

MASK_ID = 126336


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
    parser.add_argument("--captions", type=Path, help="frame_captions.jsonl from caption_frames_dynin.py")
    parser.add_argument("--per-category", type=int, default=10, help="Videos per category")
    parser.add_argument("--no-example", action="store_true", help="Leave the worked example out of the prompt")
    parser.add_argument("--no-options", action="store_true", help="Leave the answer options out of the prompt")
    parser.add_argument("--prefill", default="1.", help="Fixed start of the reply ('' for none)")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plans_llada"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=224, help="For the review page's frames")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128, help="TabAlign's planner setting")
    parser.add_argument("--block-length", type=int, default=32, help="TabAlign's planner setting")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    wanted, labels, questions = sample_videos(args.labels, args.per_category, args.seed, args.data_dir)
    captions = {}
    if args.captions:
        captions = {k: r["captions"] for k, r in load_done(args.captions).items()}
        missing = [k for k in wanted if k not in captions]
        if missing:
            raise SystemExit(f"{len(missing)} videos have no captions in {args.captions}, such as {missing[0][1]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plans_path = args.out_dir / "plans.jsonl"
    done = load_done(plans_path)
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(done)} already done, {len(todo)} to run; example {not args.no_example}, "
          f"options {not args.no_options}, frame descriptions {bool(captions)}, reply starts with {args.prefill!r}")

    if todo:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device)
        model.eval()
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        for n, key in enumerate(todo, start=1):
            item = questions[key]
            row = {"category": key[0], "video": key[1], "question": item["question"], "candidates": item["candidates"],
                   "answer": item["answer"], "frames_dir": f"{key[0]}_{Path(key[1]).stem}", "captions": captions.get(key)}
            start = time.time()
            try:
                _, _, row["label_frames"] = load_frames(key, labels[key], args.data_dir, args.frames, args.resolution,
                                                        args.out_dir / "frames" / row["frames_dir"])
                content = build_plan_prompt(item, not args.no_example, not args.no_options, captions.get(key))
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                       tokenize=False) + args.prefill
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

    report([done[k] for k in wanted if k in done], args.out_dir, "LLaDA-8B-Instruct plans")


if __name__ == "__main__":
    main()
