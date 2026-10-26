"""
Record Dream-VL-7B's attention on DyBench videos, measured the way TabAlign measured LLaDA.

Dream-VL is a diffusion vision-language model: its language model is Dream-7B, a masked-diffusion LLM with
bidirectional attention initialised from Qwen2.5-7B, and its vision encoder is Qwen2-VL's. Like Qwen3-VL it
merges each pair of frames into one time step, with each video token a 28x28 pixel square. Each frame Qwen3-VL
saw is resized so the grid of squares has exactly Qwen3-VL's rows and columns, so Dream-VL gets the same
frames, time steps and token grid, and the labels apply unchanged.

Following TabAlign, one forward pass over [video + question + masked answer canvas] gives each layer's
attention (mean over heads) from the position that emits the first answer token. Dream keeps Qwen2.5's shift
(the token at position i is predicted from position i-1), so that is the last prompt position, which here
attends to the whole canvas in both directions. The answer is decoded the way Dream-VL's own diffusion_generate
does it, with its multiple-choice setting (3 tokens, 1 step), but scoring only the answer slots: the original
scores every position against the whole vocabulary, which runs out of GPU memory on the longer videos.

Writes <out-dir>/results.jsonl and <out-dir>/<category>/<video>.npz (vision_attention [1, layers, T, H, W],
same layout as run_attention.py), so score_attention.py, plot_time_layer.py and compare_models.py run on it
with the original labels files.

Needs its own environment (see README).
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from attention_rows import SdpaRowRecorder
from run_attention_vidlada import LETTERS, build_prompt, parse_letter, sample_frames

SQUARE = 28  # pixels per video token side: 14px patches merged 2x2


def decode_answer(model, embeds, position_ids, prompt_len, length, steps, mask_id, eps=1e-3):
    """Dream-VL's diffusion_generate for one sequence (alg maskgit_plus, temperature 0, no EOS penalty), scoring
    only the answer slots. As in Dream, the token at position i is read from position i-1's output."""
    embeds = embeds.clone()
    ids = [mask_id] * length
    timesteps = torch.linspace(1, eps, steps + 1)
    for i in range(steps):
        hidden = model.model(inputs_embeds=embeds, position_ids=position_ids, use_cache=False)[0]
        logits = model.lm_head(hidden[:, prompt_len - 1:prompt_len - 1 + length])[0]
        confidence, tokens = torch.softmax(logits, dim=-1).max(dim=-1)
        open_slots = torch.tensor([j for j in range(length) if ids[j] == mask_id])
        t, s = timesteps[i], timesteps[i + 1]
        count = int(len(open_slots) * (1 - s / t)) if i < steps - 1 else len(open_slots)
        if count > 0:
            for j in open_slots[torch.topk(confidence[open_slots], count).indices].tolist():
                ids[j] = int(tokens[j])
                embeds[:, prompt_len + j] = model.get_input_embeddings()(tokens[j:j + 1])[0]
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files: which videos to run, and their token grid")
    parser.add_argument("--model", default="Dream-org/Dream-VL-7B")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dreamvl_7b"))
    parser.add_argument("--max-new-tokens", type=int, default=3, help="Answer canvas length (Dream-VL's multiple-choice setting)")
    parser.add_argument("--steps", type=int, default=1, help="Diffusion steps (Dream-VL's multiple-choice setting)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    grids = {}
    for path in args.labels:
        for label in json.load(open(path))["labels"]:
            grids[(label["category"], label["video"])] = label["grid"]
    wanted = sorted(grids)
    questions = {}
    for category in {c for c, _ in wanted}:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in open(results_path):
            row = json.loads(line)
            if "error" not in row:
                done.add((row["category"], row["video"]))
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(done)} already done, {len(todo)} to run")

    device = args.device
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                      attn_implementation="sdpa").to(device)
    model.eval()
    # Frames arrive already sized to the grid, so the processor must leave them as they are
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, min_pixels=SQUARE * SQUARE,
                                              max_pixels=SQUARE * SQUARE * 100000)
    layers = len(model.model.layers)
    mask_id, video_id = model.config.mask_token_id, model.config.video_token_id
    stop_ids = {model.config.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|im_end|>")}
    recorder = SdpaRowRecorder()
    print(f"Model loaded: {layers} layers")

    for n, (category, video) in enumerate(todo, start=1):
        item = questions[(category, video)]
        row = {"category": category, "video": video, "question": item["question"], "candidates": item["candidates"],
               "answer": item["answer"], "answer_letter": LETTERS[item["candidates"].index(item["answer"])]}
        start = time.time()
        try:
            steps, rows_per_step, cols_per_step = grids[(category, video)]
            frames, indices, native_fps = sample_frames(args.data_dir / "video" / category / category / video)
            if len(frames) != 2 * steps:
                raise RuntimeError(f"{len(frames)} frames read, but the label has {steps} time steps")
            size = (cols_per_step * SQUARE, rows_per_step * SQUARE)
            clip = [Image.fromarray(frame).resize(size, Image.BICUBIC) for frame in frames]
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": build_prompt(item)}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], videos=[clip], padding=True, return_tensors="pt").to(device)
            grid_thw = inputs["video_grid_thw"]
            if grid_thw[0].tolist() != [steps, 2 * rows_per_step, 2 * cols_per_step]:
                raise RuntimeError(f"Video grid {grid_thw[0].tolist()}, expected {[steps, 2 * rows_per_step, 2 * cols_per_step]}")
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]
            canvas = torch.cat([input_ids, torch.full((1, args.max_new_tokens), mask_id, device=device)], dim=1)

            with torch.inference_mode():
                embeds = model.get_input_embeddings()(canvas)
                video_embeds = torch.cat(model.get_video_features(inputs["pixel_values_videos"], grid_thw))
                is_video = canvas == video_id
                if int(is_video.sum()) != video_embeds.shape[0]:
                    raise RuntimeError(f"{int(is_video.sum())} video tokens for {video_embeds.shape[0]} video features")
                embeds[is_video] = model.projector(video_embeds).to(embeds.dtype)
                position_ids, _ = model.get_rope_index(canvas, None, grid_thw, None)
                with recorder.record(prompt_len - 1):
                    model.model(inputs_embeds=embeds, position_ids=position_ids, use_cache=False)
                rows = torch.stack(recorder.rows).numpy()                          # [layers, sequence]
                if rows.shape[0] != layers:
                    raise RuntimeError(f"Recorded {rows.shape[0]} attention calls for {layers} layers")
                answer_ids = decode_answer(model, embeds, position_ids, prompt_len, args.max_new_tokens, args.steps, mask_id)
            stops = [i for i, token in enumerate(answer_ids) if token in stop_ids]
            answer_ids = answer_ids[:stops[0]] if stops else answer_ids
            response = processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            video_start = int(is_video[0].nonzero()[0])
            video_len = steps * rows_per_step * cols_per_step
            video_rows = rows[:, video_start:video_start + video_len]
            grid = video_rows.reshape(rows.shape[0], steps, rows_per_step, cols_per_step)
            times = [(indices[2 * t] + indices[2 * t + 1]) / 2 / native_fps for t in range(steps)]

            attention_file = Path(category) / (Path(video).stem + ".npz")
            (args.out_dir / category).mkdir(exist_ok=True)
            np.savez_compressed(args.out_dir / attention_file, vision_attention=grid[None].astype(np.float32),
                                frame_times=np.array(times, dtype=np.float32), frame_indices=np.array(indices))
            pred = parse_letter(response, item["candidates"])
            row.update(response=response, pred_letter=pred, correct=pred == row["answer_letter"],
                       attention_file=str(attention_file), grid_thw=[steps, rows_per_step, cols_per_step],
                       vision_share=float(video_rows.sum(axis=1).mean()))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - start, 2)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        status = row.get("error") or f"{row['response']!r} ({'right' if row['correct'] else 'wrong'}, answer {row['answer_letter']})"
        print(f"[{n}/{len(todo)}] {category}/{video}: {status} {row['seconds']}s "
              f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
