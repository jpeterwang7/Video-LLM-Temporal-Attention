"""
Record VidLaDA-8B's attention on DyBench videos, measured the way TabAlign measured LLaDA.

VidLaDA is a diffusion video LLM (LLaDA backbone, bidirectional attention). Following TabAlign:
one forward pass over [video + question + masked answer canvas], reading each layer's attention
(mean over heads) from the first masked answer position, i.e. the position that emits the answer.
It is fed exactly the frames Qwen3-VL saw (2 per second, at most 32, evenly spaced), and pairs of
frames are summed into Qwen3-VL's time steps so both models share one time axis. Each frame is a
14x14 grid of video tokens (SigLIP2 at 384px, pooled 2x2); the end-of-row tokens are dropped.

Writes <out-dir>/results.jsonl and <out-dir>/<category>/<video>.npz (vision_attention
[1, layers, T, 14, 14], same layout as run_attention.py), plus the labels converted to this grid in
<out-dir>/labels/, so score_attention.py and plot_time_layer.py run on it unchanged.

Needs its own environment with the VidLaDA repository (see README).
"""
import argparse
import copy
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

MASK_ID = 126336
LETTERS = "ABCDEFGH"


def sample_frames(path, fps=2.0, max_frames=32):
    """The same frames Qwen3-VL was given: an even number, evenly spaced, 2 per second, at most 32."""
    cap = cv2.VideoCapture(str(path))
    native_fps, total = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num = min(max(int(total / native_fps * fps), 4), max_frames, total)
    num -= num % 2
    wanted = np.linspace(0, total - 1, num).round().astype(int).tolist()
    frames, indices = [], []
    for i in range(wanted[-1] + 1):
        if not cap.grab():
            break
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                indices.append(i)
    cap.release()
    if len(frames) % 2:
        frames, indices = frames[:-1], indices[:-1]
    return frames, indices, native_fps


def build_prompt(item):
    lines = [item["question"], "Options:"] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(item["candidates"])]
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def parse_letter(response, candidates):
    allowed = LETTERS[: len(candidates)]
    match = re.search(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", response)
    if match:
        return match.group(1)
    for i, candidate in enumerate(candidates):
        if candidate.lower() in response.lower():
            return LETTERS[i]
    return None


def convert_label(label, rows, cols):
    """Re-express a labelling-page label on a rows x cols grid per time step (same time steps)."""
    cells = {}
    for t, boxes in label["boxes"].items():
        out = set()
        for x0, y0, x1, y1 in boxes:
            for r in range(math.floor(y0 * rows), min(rows - 1, math.floor(y1 * rows - 1e-6)) + 1):
                for c in range(math.floor(x0 * cols), min(cols - 1, math.floor(x1 * cols - 1e-6)) + 1):
                    out.add((r, c))
        cells[t] = sorted([list(x) for x in out])
    converted = dict(label)
    converted["grid"] = [label["grid"][0], rows, cols]
    converted["cells"] = cells
    return converted


class AnswerRowRecorder:
    """Pre-hooks on every LLaDA attention module: recompute the attention row of one query position
    (the first masked answer slot) from the module's own projections, mean over heads."""

    def __init__(self, layers):
        self.query = None
        self.active = False
        self.rows = [None] * len(layers)
        self.handles = [layer.self_attn.register_forward_pre_hook(self._hook(i), with_kwargs=True)
                        for i, layer in enumerate(layers)]

    def _hook(self, index):
        def hook(module, args, kwargs):
            if not self.active:
                return
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            position_ids, mask = kwargs.get("position_ids"), kwargs.get("attention_mask")
            ops = sys.modules[type(module).__module__]
            bsz, n, _ = hidden.shape
            q = self.query
            keys = module.k_proj(hidden).view(bsz, n, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            query = module.q_proj(hidden[:, q:q + 1]).view(bsz, 1, module.num_heads, module.head_dim).transpose(1, 2)
            cos, sin = module.rotary_emb(keys, position_ids)
            keys, _ = ops.apply_rotary_pos_emb(keys, keys, cos, sin)
            query, _ = ops.apply_rotary_pos_emb(query, query, cos[:, q:q + 1], sin[:, q:q + 1])
            keys = ops.repeat_kv(keys, module.num_key_value_groups)
            scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(module.head_dim)
            if mask is not None:
                scores = scores + mask[:, :, q:q + 1, :n]
            self.rows[index] = torch.softmax(scores.float(), dim=-1)[0, :, 0].mean(dim=0).cpu()
        return hook


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files: which videos to run, and to convert")
    parser.add_argument("--vidlada-repo", type=Path, default=Path.home() / "VidLaDA")
    parser.add_argument("--model", default="ziHoHe/VidLaDA-8B")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/vidlada_8b"))
    parser.add_argument("--gen-length", type=int, default=2, help="Answer canvas length (VidLaDA's default for multiple choice)")
    args = parser.parse_args()

    sys.path.insert(0, str(args.vidlada_repo / "train"))
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
        args.model, None, "llava_llada_video", device_map="cuda:0", torch_dtype="bfloat16",
        attn_implementation="sdpa", overwrite_config={"mm_spatial_pool_stride": 2, "mm_spatial_pool_mode": "bilinear"})
    model.eval()
    backbone = model.get_model()
    recorder = AnswerRowRecorder(backbone.layers)
    per_frame = rows_per_frame * (cols_per_frame + 1)  # one end-of-row token per row
    print(f"Model loaded: {len(backbone.layers)} layers")

    for n, (category, video) in enumerate(todo, start=1):
        item = questions[(category, video)]
        row = {"category": category, "video": video, "question": item["question"], "candidates": item["candidates"],
               "answer": item["answer"], "answer_letter": LETTERS[item["candidates"].index(item["answer"])]}
        start = time.time()
        try:
            frames, indices, native_fps = sample_frames(args.data_dir / "video" / category / category / video)
            pixels = image_processor.preprocess(np.stack(frames), return_tensors="pt")["pixel_values"].to("cuda:0", torch.bfloat16)
            conv = copy.deepcopy(conv_templates["llava_llada"])
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
                mask = backbone.embed_tokens(torch.tensor([MASK_ID], device="cuda:0"))
                canvas = torch.cat([embeds, mask[None].expand(1, args.gen_length, -1).to(embeds.dtype)], dim=1)
                recorder.query, recorder.active = prompt_len, True
                backbone(inputs_embeds=canvas)
                recorder.active = False
                rows = torch.stack(recorder.rows).numpy()                          # [layers, sequence]

                answer_ids = model.generate(input_ids, attention_mask=attention_mask, images=[pixels], modalities=["video"],
                                            steps=args.gen_length, gen_length=args.gen_length, block_length=args.gen_length,
                                            temperature=0, cfg_scale=0.0, remasking="low_confidence", tokenizer=tokenizer,
                                            stopping_criteria=[conv.sep], use_cache=False)
            response = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()

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
            recorder.active = False
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
