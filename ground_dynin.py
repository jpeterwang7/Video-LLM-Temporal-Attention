"""
Ground each plan target in the video with Dynin-Omni's attention, on the same frames Qwen3-VL sees.

TabAlign compiles a plan step's [target: col] into a cell mask by matching the column name, which needs no vision. A
video target ("the person", "crack egg") has to be found in the frames, and the planner (LLaDA-8B-Instruct) never sees
them. Dynin-Omni, a masked-diffusion model on the same LLaDA-8B backbone that does see video, does the grounding: its
bidirectional attention found the labelled objects (check_dynin_attention.py), as TabAlign's pilot found for LLaDA on
tables. For each target, one forward pass over [video + "Find the <target> in the video." + masked answer canvas]
gives each layer's attention from the first masked slot, as run_attention_dynin.py measures it.

The frame gap. Dynin-Omni was run on 5 frames centre-cropped to 224x224, while Qwen3-VL sees up to 32 frames (merged
in pairs into up to 16 time steps) of the whole frame, so a Dynin-Omni map covered a third of Qwen3-VL's time steps
and cut the sides off wide videos. Here Dynin-Omni sees one frame per Qwen3-VL time step (the first of each merged
pair; up to 16 frames, 3,136 tokens, inside its 4,096-token context), in two ways:
    squeeze  the whole frame resized to 224x224, so its 14x14 grid covers exactly what Qwen3-VL's grid covers
    crop     resized and centre-cropped to 224x224 as Dynin-Omni was trained
Every target is grounded both ways; which one makes the masks is chosen on the validation split (fit_tabattn.py).
For the frame-gap comparison, the question itself is also run both ways and answered (Dynin-Omni's own multiple-choice
decoding), next to the 5-frame run in outputs/dynin_omni.

Writes <out-dir>/dynin/<mode>/<category>/<video>.npz (queries, last4 and all: [queries, T, 14, 14] attention averaged
over the last 4 and over all layers, frame_times, crop) and <out-dir>/dynin/results_<mode>.jsonl.

Usage (Dynin-Omni environment, after plan_tabalign.py):
    python ground_dynin.py --set calib test
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import tabalign_video as tv
from attention_rows import SdpaRowRecorder
from run_attention_dynin import SPECIAL_TOKENS, crop_window
from run_attention_vidlada import MASK_ID, build_prompt, parse_letter, sample_frames

MODES = ["squeeze", "crop"]


def grounding_prompt(target):
    return f"Find the {target} in the video."


@torch.inference_mode()
def decode_answer(model, canvas, prompt_len, gen_length, first_logits):
    """Dynin-Omni's mmu_generate as its multiple-choice evaluation runs it (a gen_length canvas, gen_length steps, one
    block, greedy, lowest-confidence remasking, no guidance, full vocabulary), but with the softmax taken over the
    answer slots only: mmu_generate takes it over the whole sequence in float64, which with 16 frames needs more
    memory than the card has left. The first step reuses the logits of the attention pass, which ran on the same
    canvas."""
    canvas = canvas.clone()
    logits = first_logits
    for step in range(gen_length):
        if step:
            logits = model(canvas).logits[0, prompt_len:prompt_len + gen_length].float()
        confidence, tokens = torch.softmax(logits, dim=-1).max(dim=-1)
        open_slots = canvas[0, prompt_len:prompt_len + gen_length] == MASK_ID
        confidence = torch.where(open_slots, confidence, torch.full_like(confidence, -1.0))
        slot = int(confidence.argmax())
        canvas[0, prompt_len + slot] = tokens[slot]
    return canvas[:, prompt_len:prompt_len + gen_length]


def plan_targets(out_dir):
    """Every plan target of every video, from both planners' plans."""
    targets = {}
    for path in sorted(out_dir.glob("plans_*.jsonl")):
        for key, row in tv.load_rows(path).items():
            for step in row.get("steps", []):
                if step.get("target"):
                    targets.setdefault(key, [])
                    if step["target"] not in targets[key]:
                        targets[key].append(step["target"])
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="+", default=["calib", "test"])
    parser.add_argument("--limit", type=int, help="Only the first N videos of each set")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--dynin-repo", type=Path, default=Path.home() / "Dynin-Omni")
    parser.add_argument("--model", default="snu-aidas/Dynin-Omni")
    parser.add_argument("--vq-model", default="snu-aidas/magvitv2")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--gen-length", type=int, default=2, help="Answer canvas (Dynin-Omni's multiple-choice setting)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(args.dynin_repo))
    from PIL import Image
    from transformers import AutoTokenizer
    from models import DyninOmniModelLM, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform

    sets = tv.load_sets(args.out_dir / "sets.json")
    wanted = []
    for name in args.set:
        wanted += [k for k in sets[name][:args.limit] if k not in wanted]
    questions = tv.load_questions(args.data_dir)
    targets = plan_targets(args.out_dir)
    root = args.out_dir / "dynin"
    root.mkdir(parents=True, exist_ok=True)
    results = {mode: root / f"results_{mode}.jsonl" for mode in args.modes}
    done = {mode: tv.load_rows(path) for mode, path in results.items()}

    def finished(key, mode):
        row = done[mode].get(key)
        return row is not None and set(targets.get(key, [])) <= set(row.get("queries", []))

    todo = [k for k in wanted if not all(finished(k, m) for m in args.modes)]
    print(f"{len(wanted)} videos, {len(wanted) - len(todo)} already grounded, {len(todo)} to run", flush=True)
    if not todo:
        return

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=2048, special_tokens=SPECIAL_TOKENS, ignore_id=-100,
                                       cond_dropout_prob=0.0, use_reserved_token=True)
    vq_model = MAGVITv2.from_pretrained(args.vq_model).to(device).eval()
    vq_model.requires_grad_(False)
    model = DyninOmniModelLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    blocks = model.model.transformer.blocks
    for block in blocks:
        block.flash_attn_func = None           # route attention through torch's sdpa, where the recorder sits
    recorder = SdpaRowRecorder()
    side = args.resolution // 16                 # MAGVIT-v2 downsamples by 16
    special = {name: torch.tensor([[int(uni_prompting.sptids_dict[name])]], device=device)
               for name in ("<|v2t|>", "<|soi|>", "<|eoi|>", "<|sot|>")}
    print(f"Dynin-Omni loaded: {len(blocks)} layers", flush=True)

    def wrap(text):
        return "<|start_header_id|>user<|end_header_id|>\n" + text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    for n, key in enumerate(todo, start=1):
        item = questions[key]
        frames, indices, native_fps = sample_frames(tv.video_path(args.data_dir, *key))
        steps = len(frames) // 2
        picked = [2 * t for t in range(steps)]
        height, width = frames[0].shape[:2]
        queries = ["question"] + targets.get(key, [])
        for mode in args.modes:
            if finished(key, mode):
                continue
            start = time.time()
            row = {"category": key[0], "video": key[1], "mode": mode, "answer": item["answer"],
                   "answer_letter": tv.LETTERS[item["candidates"].index(item["answer"])], "queries": queries}
            try:
                images = []
                for p in picked:
                    image = Image.fromarray(frames[p])
                    if mode == "squeeze":
                        image = image.resize((args.resolution, args.resolution), Image.BICUBIC)
                    images.append(image_transform(image, resolution=args.resolution))
                with torch.inference_mode():
                    video_tokens = (vq_model.get_code(torch.stack(images).to(device)) + len(uni_prompting.text_tokenizer)).view(1, -1)
                    if video_tokens.shape[1] != steps * side * side:
                        raise RuntimeError(f"{video_tokens.shape[1]} video tokens for {steps} frames")
                    head = torch.cat([special["<|v2t|>"], special["<|soi|>"], video_tokens, special["<|eoi|>"],
                                      special["<|sot|>"]], dim=1)
                    last4, every = [], []
                    for q in queries:
                        text = build_prompt(item) if q == "question" else grounding_prompt(q)
                        ids = torch.tensor(uni_prompting.text_tokenizer([wrap(text)])["input_ids"], device=device)
                        input_ids = torch.cat([head, ids], dim=1)
                        prompt_len = input_ids.shape[1]
                        canvas = torch.cat([input_ids, torch.full((1, args.gen_length), MASK_ID, device=device)], dim=1)
                        with recorder.record(prompt_len):
                            first = model(canvas).logits[0, prompt_len:prompt_len + args.gen_length].float()
                        rows = torch.stack(recorder.rows).numpy()
                        if rows.shape[0] != len(blocks):
                            raise RuntimeError(f"Recorded {rows.shape[0]} attention calls for {len(blocks)} layers")
                        video_rows = rows[:, 2:2 + video_tokens.shape[1]].reshape(len(blocks), steps, side, side)
                        last4.append(video_rows[-tv.ATTN_LAYERS:].mean(axis=0))
                        every.append(video_rows.mean(axis=0))
                        if q == "question":
                            answer = decode_answer(model, canvas, prompt_len, args.gen_length, first)
                            response = uni_prompting.text_tokenizer.batch_decode(answer, skip_special_tokens=True)[0].strip()
                            pred = parse_letter(response, item["candidates"])
                            row.update(response=response, pred_letter=pred, correct=pred == row["answer_letter"])
                out = root / mode / key[0] / (Path(key[1]).stem + ".npz")
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, queries=np.array(queries), last4=np.stack(last4).astype(np.float16),
                                    all=np.stack(every).astype(np.float16),
                                    frame_times=np.array([indices[p] / native_fps for p in picked], dtype=np.float32),
                                    crop=np.array(crop_window(width, height, args.resolution) if mode == "crop"
                                                  else (0.0, 0.0, 1.0, 1.0), dtype=np.float32))
                row.update(file=str(out.relative_to(root)), time_steps=steps)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - start, 2)
            with open(results[mode], "a") as f:
                f.write(json.dumps(row) + "\n")
            if "error" not in row:
                done[mode][key] = row
            status = row.get("error") or f"{len(queries)} queries, answer {row['response']!r} ({'right' if row['correct'] else 'wrong'})"
            print(f"[{n}/{len(todo)}] {mode} {key[0]}/{key[1]}: {status} ({row['seconds']}s, peak GPU "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)

    for mode in args.modes:
        rows = [r for r in tv.load_rows(results[mode]).values() if (r["category"], r["video"]) in set(wanted)]
        right = sum(r["correct"] for r in rows)
        print(f"Dynin-Omni answering directly, one frame per Qwen3-VL time step ({mode}): {right} / {len(rows)}")


if __name__ == "__main__":
    main()
