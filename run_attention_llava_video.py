"""
Record LLaVA-Video-7B's attention on DyBench videos: the autoregressive counterpart of VidLaDA.

LLaVA-Video-7B-Qwen2 and VidLaDA are built from the same LLaVA-NeXT code: a SigLIP vision encoder at
384px, 2x2 bilinear pooling to a 14x14 grid per frame, one end-of-row token per row, and the same video
token layout. What differs is the language model: Qwen2-7B reads left to right, VidLaDA's LLaDA-8B is a
diffusion model with bidirectional attention. Comparing the two isolates how the answer is decoded.

As for Qwen3-VL, attention is read from the position that emits the answer (the last prompt token), per
layer, mean over heads, during the first step of greedy decoding. It is fed exactly the frames Qwen3-VL saw
(2 per second, at most 32, evenly spaced), and pairs of frames are summed into Qwen3-VL's time steps, as
for VidLaDA.

Writes <out-dir>/results.jsonl and <out-dir>/<category>/<video>.npz (vision_attention
[1, layers, T, 14, 14], same layout as run_attention.py), plus the labels converted to this grid in
<out-dir>/labels/, so score_attention.py and plot_time_layer.py run on it unchanged.

Runs in the VidLaDA environment, with the LLaVA-NeXT repository (see README).
"""
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

from attention_rows import SdpaRowRecorder
from run_attention_vidlada import LETTERS, build_prompt, convert_label, parse_letter, sample_frames


def greedy_answer(model, embeds, recorder, query, max_new_tokens, stop_ids):
    """Greedy decoding (what generate does with do_sample=False), recording attention on the first step."""
    backbone = model.get_model()
    with recorder.record(query):
        out = backbone(inputs_embeds=embeds, past_key_values=DynamicCache(), use_cache=True)
    rows = torch.stack(recorder.rows).numpy()                                  # [layers, sequence]
    ids = []
    for _ in range(max_new_tokens):
        token = int(model.lm_head(out.last_hidden_state[:, -1]).argmax(dim=-1))
        if token in stop_ids:
            break
        ids.append(token)
        step = backbone.embed_tokens(torch.tensor([[token]], device=embeds.device))
        out = backbone(inputs_embeds=step, past_key_values=out.past_key_values, use_cache=True)
    return rows, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files: which videos to run, and to convert")
    parser.add_argument("--llava-repo", type=Path, default=Path.home() / "LLaVA-NeXT")
    parser.add_argument("--model", default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/llava_video_7b"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    sys.path.insert(0, str(args.llava_repo))
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    labels = {}
    for path in args.labels:
        data = json.load(open(path))
        labels[path.name] = data
    wanted = sorted({(l["category"], l["video"]) for d in labels.values() for l in d["labels"]})
    questions = {}
    for category in {c for c, _ in wanted}:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "labels").mkdir(exist_ok=True)
    rows_per_frame, cols_per_frame = 14, 14
    for name, data in labels.items():
        converted = dict(data, labels=[convert_label(l, rows_per_frame, cols_per_frame) for l in data["labels"]])
        json.dump(converted, open(args.out_dir / "labels" / name, "w"), indent=1)

    results_path = args.out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in open(results_path):
            row = json.loads(line)
            if "error" not in row:
                done.add((row["category"], row["video"]))
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(done)} already done, {len(todo)} to run")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model, None, "llava_qwen", device_map="cuda:0", torch_dtype="bfloat16",
        attn_implementation="sdpa", overwrite_config={"mm_spatial_pool_stride": 2, "mm_spatial_pool_mode": "bilinear"})
    model.eval()
    layers = len(model.get_model().layers)
    recorder = SdpaRowRecorder()
    per_frame = rows_per_frame * (cols_per_frame + 1)  # one end-of-row token per row
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    print(f"Model loaded: {layers} layers")

    for n, (category, video) in enumerate(todo, start=1):
        item = questions[(category, video)]
        row = {"category": category, "video": video, "question": item["question"], "candidates": item["candidates"],
               "answer": item["answer"], "answer_letter": LETTERS[item["candidates"].index(item["answer"])]}
        start = time.time()
        try:
            frames, indices, native_fps = sample_frames(args.data_dir / "video" / category / category / video)
            pixels = image_processor.preprocess(np.stack(frames), return_tensors="pt")["pixel_values"].to("cuda:0", torch.bfloat16)
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + build_prompt(item))
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).to("cuda:0")
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            with torch.inference_mode():
                _, _, _, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, None, [pixels], ["video"], image_sizes=None)
                prompt_len = embeds.shape[1]
                video_start = int((input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
                video_len = prompt_len - (input_ids.shape[1] - 1)
                if video_len != len(frames) * per_frame:
                    raise RuntimeError(f"{video_len} video tokens for {len(frames)} frames (expected {per_frame} each)")
                rows, answer_ids = greedy_answer(model, embeds, recorder, prompt_len - 1, args.max_new_tokens, stop_ids)
                if rows.shape[0] != layers:
                    raise RuntimeError(f"Recorded {rows.shape[0]} attention calls for {layers} layers")
            response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            del embeds, pixels

            video_rows = rows[:, video_start:video_start + video_len]
            grid = video_rows.reshape(rows.shape[0], len(frames), rows_per_frame, cols_per_frame + 1)[..., :cols_per_frame]
            steps = grid.reshape(rows.shape[0], len(frames) // 2, 2, rows_per_frame, cols_per_frame).sum(axis=2)
            times = [(indices[2 * t] + indices[2 * t + 1]) / 2 / native_fps for t in range(len(frames) // 2)]

            attention_file = Path(category) / (Path(video).stem + ".npz")
            (args.out_dir / category).mkdir(exist_ok=True)
            np.savez_compressed(args.out_dir / attention_file, vision_attention=steps[None].astype(np.float32),
                                frame_times=np.array(times, dtype=np.float32), frame_indices=np.array(indices))
            pred = parse_letter(response, item["candidates"])
            row.update(response=response, pred_letter=pred, correct=pred == row["answer_letter"],
                       attention_file=str(attention_file), grid_thw=[len(frames) // 2, rows_per_frame, cols_per_frame],
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

    print(f"Results: {results_path}\nConverted labels: {args.out_dir / 'labels'}")


if __name__ == "__main__":
    main()
