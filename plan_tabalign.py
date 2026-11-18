"""
TabAlign's understanding step for video: write a plan for each question, without seeing the video.

The planner gets the question, the options and compact metadata (the video's length and its number of time steps,
the counterpart of a table's column names), and writes 2 to 4 steps, each naming one of six operations and ending in
[target: <object or event>; steps: <time steps>], the counterpart of TabAlign's [target: col] (Section 4.1, B.2).

    --planner llada   LLaDA-8B-Instruct, decoded as TabAlign's planner: a 256-token canvas, blocks of 32, classifier-free
                      guidance 1.5, greedy, lowest-confidence remasking, and 256 denoising steps, TabAlign's setting
                      "for higher-quality plans" (B.2). Its default of 128 steps fixes 2 tokens per step, and in the
                      smoke run that fused neighbouring words ("movesove or changesver") and broke target tags.
                      Decoding stops after the block in which the plan ends: blocks are fixed left to right and
                      everything after the end token is dropped anyway, so the plan is the same and costs less
                      (Dynin-Omni environment)
    --planner qwen    Qwen3-VL-8B as the autoregressive planner of TabAlign's Table 5, same prompt, greedy, text only
                      (main environment)

The reply is started with "1." so it opens a numbered list. A video and its reversed twin share their question,
options and metadata, so their prompt is planned once. Writes <out-dir>/plans_<planner>.jsonl and prints how many
plans parse, how many copy the prompt's example, and how many targets name nothing in the question (TabAlign's
column-hallucination rate, Table 8).

Usage:
    python plan_tabalign.py --planner llada --set calib test
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import tabalign_video as tv

MASK_ID = 126336


@torch.no_grad()
def llada_generate(model, prompt_ids, gen_length=256, steps=256, block_length=32, cfg_scale=1.5, stop_ids=()):
    """LLaDA's sampler as TabAlign runs it: fully masked canvas, semi-autoregressive blocks, in each step the most
    confident masked tokens of the current block are fixed (greedy), with classifier-free guidance against the same
    canvas with the prompt masked out. Stops after a block containing one of stop_ids (later blocks stay masked)."""
    length = prompt_ids.shape[1]
    x = torch.full((1, length + gen_length), MASK_ID, dtype=torch.long, device=prompt_ids.device)
    x[:, :length] = prompt_ids
    prompt_index = x != MASK_ID
    blocks = gen_length // block_length
    per_block = steps // blocks
    for b in range(blocks):
        start, end = length + b * block_length, length + (b + 1) * block_length
        masked = int((x[0, start:end] == MASK_ID).sum())
        counts = [masked // per_block + (i < masked % per_block) for i in range(per_block)]
        for count in counts:
            if count == 0:
                continue
            is_masked = x == MASK_ID
            if cfg_scale > 0:
                unconditional = x.clone()
                unconditional[prompt_index] = MASK_ID
                logits, un_logits = torch.chunk(model(torch.cat([x, unconditional])).logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            guess = logits.argmax(dim=-1)
            confidence = torch.softmax(logits.float(), dim=-1).gather(-1, guess[..., None])[..., 0]
            confidence[:, end:] = -float("inf")
            confidence = torch.where(is_masked, confidence, torch.full_like(confidence, -float("inf")))
            chosen = confidence[0].topk(count).indices
            x[0, chosen] = guess[0, chosen]
        if any(int(t) in stop_ids for t in x[0, start:end].tolist()):
            break
    return x[0, length:].tolist()


class LladaPlanner:
    def __init__(self, args):
        from transformers import AutoModel, AutoTokenizer
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.llada_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(args.llada_model, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).to(args.device).eval()
        self.eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.eos = self.tokenizer.eos_token_id

    def __call__(self, prompt):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                  tokenize=False) + self.args.prefill
        ids = torch.tensor(self.tokenizer(text)["input_ids"], device=self.args.device)[None]
        stops = tuple(s for s in (self.eot, self.eos) if s is not None)
        out = llada_generate(self.model, ids, self.args.gen_length, self.args.steps, self.args.block_length,
                             self.args.cfg_scale, stops)
        cut = min([out.index(s) for s in stops if s in out] + [len(out)])
        out = [t for t in out[:cut] if t != MASK_ID]
        return (self.args.prefill + self.tokenizer.decode(out, skip_special_tokens=True)).strip()


class QwenPlanner:
    def __init__(self, args):
        from omnitrace.constants import QWEN3VL_MODEL_PATH
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        path = args.qwen_model or QWEN3VL_MODEL_PATH
        self.args = args
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype="auto", device_map=args.device,
                                                                     attn_implementation="sdpa").eval()
        self.processor = AutoProcessor.from_pretrained(path)

    @torch.no_grad()
    def __call__(self, prompt):
        text = self.processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                                                  add_generation_prompt=True, tokenize=False) + self.args.prefill
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.args.gen_length, do_sample=False)
        reply = self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return (self.args.prefill + reply).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", choices=["llada", "qwen"], required=True)
    parser.add_argument("--set", nargs="+", default=["calib", "test"], help="Sets from sets.json to plan")
    parser.add_argument("--limit", type=int, help="Only the first N videos of each set")
    parser.add_argument("--exclude-set", nargs="*", default=[], help="Leave out the videos of these sets first")
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--llada-model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--qwen-model", help="Default: the Qwen3-VL run_attention.py used")
    parser.add_argument("--prefill", default="1.", help="Fixed start of the reply")
    parser.add_argument("--gen-length", type=int, default=256, help="TabAlign's canvas length")
    parser.add_argument("--steps", type=int, default=256, help="Denoising steps (TabAlign's higher-quality setting)")
    parser.add_argument("--block-length", type=int, default=32, help="TabAlign's block length")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="TabAlign's classifier-free guidance")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sets = tv.load_sets(args.out_dir / "sets.json")
    wanted = []
    for name in args.set:        # in the executor's order, so --limit N plans the videos it runs first
        wanted += [k for k in tv.select_videos(sets, name, args.exclude_set, args.limit) if k not in wanted]
    questions = tv.load_questions(args.data_dir)
    path = args.out_dir / f"plans_{args.planner}.jsonl"
    done = tv.load_rows(path)
    by_prompt = {r["prompt_hash"]: r for r in done.values() if "plan" in r}
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(wanted) - len(todo)} already planned, {len(todo)} to plan", flush=True)

    planner = None
    for n, key in enumerate(todo, start=1):
        item = questions[key]
        row = {"category": key[0], "video": key[1], "question": item["question"], "candidates": item["candidates"],
               "answer": item["answer"], "planner": args.planner}
        start = time.time()
        try:
            count, duration, times = tv.qwen_time_steps(tv.video_path(args.data_dir, *key))
            prompt = tv.plan_prompt(item, count, duration)
            digest = hashlib.sha1(prompt.encode()).hexdigest()
            row.update(time_steps=count, duration=round(duration, 2), prompt_hash=digest)
            if digest in by_prompt:
                row["plan"], row["reused_from"] = by_prompt[digest]["plan"], by_prompt[digest]["video"]
            else:
                if planner is None:
                    planner = (LladaPlanner if args.planner == "llada" else QwenPlanner)(args)
                    print("Planner loaded", flush=True)
                row["plan"] = planner(prompt)
                by_prompt[digest] = row
            row["steps"] = tv.parse_plan(row["plan"], count, item)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - start, 2)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        done[key] = row
        status = row.get("error") or f"{len(row['steps'])} steps" + (" (reused)" if "reused_from" in row else "")
        print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: {status} ({row['seconds']}s)", flush=True)
        if "plan" in row and "reused_from" not in row:
            print("    " + row["plan"].replace("\n", "\n    "), flush=True)

    rows = [done[k] for k in wanted if k in done]
    parsed = [r for r in rows if r.get("steps")]
    copied = sum([s.get("target") for s in r["steps"]] == tv.EXAMPLE_TARGETS for r in parsed)
    steps = [s for r in parsed for s in r["steps"]]
    tagged = [s for s in steps if s.get("target")]
    questions_of = {(r["category"], r["video"]): r for r in rows}
    halluc = [tv.target_hallucinated(s["target"], questions_of[(r["category"], r["video"])])
              for r in parsed for s in r["steps"] if s.get("target")]
    halluc = [h for h in halluc if h is not None]
    print(f"\n{len(parsed)} of {len(rows)} plans have a parsed step; {np.mean([len(r['steps']) for r in parsed]) if parsed else 0:.1f} "
          f"steps per plan; {copied} copy the example's targets; {len(tagged)} of {len(steps)} steps have a target, "
          f"{sum(1 for s in tagged if s['steps'])} with readable time steps; "
          f"{100 * np.mean(halluc) if halluc else 0:.0f}% of targets name nothing in the question or options")
    print(f"Plans: {path}")


if __name__ == "__main__":
    main()
