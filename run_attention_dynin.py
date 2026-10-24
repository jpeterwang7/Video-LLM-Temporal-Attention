"""
Record Dynin-Omni's attention on DyBench videos, measured the way TabAlign measured LLaDA.

Dynin-Omni is a masked-diffusion omni model on a LLaDA-8B backbone. Unlike VidLaDA it has no vision
encoder: each frame is turned into discrete MAGVIT-v2 image tokens that share the text vocabulary. It was
trained and evaluated on 5 frames per video, each resized to 224px on the short side and centre-cropped to
224x224, which gives a 14x14 grid of tokens per frame. It is run that way here, with the 5 frames picked
evenly from the frames Qwen3-VL saw, so each one sits inside a labelled time step; each frame is its own time
step in the output. Following TabAlign, one forward pass over [video + question + masked answer canvas]
gives each layer's attention (mean over heads) from the first masked answer position. The answer is decoded
the way Dynin-Omni's own multiple-choice evaluation does it (a 2-token canvas, 2 steps, lowest-confidence
remasking).

Writes <out-dir>/results.jsonl and <out-dir>/<category>/<video>.npz (vision_attention
[1, layers, 5, 14, 14], same layout as run_attention.py), plus the labels moved onto these 5 frames and the
cropped 14x14 grid in <out-dir>/labels/ (boxes outside the crop are cut), so score_attention.py and
plot_time_layer.py run on it unchanged.

Needs its own environment with the Dynin-Omni repository (see README).
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from attention_rows import SdpaRowRecorder
from run_attention_vidlada import LETTERS, MASK_ID, build_prompt, convert_label, parse_letter, sample_frames

SPECIAL_TOKENS = ("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
                  "<|i2i|>", "<|ti2ti|>", "<|v2t|>", "<|v2s|>", "<|s2t|>", "<|t2s|>", "<|s2s|>", "<|soa|>", "<|eoa|>",
                  "<think>", "</think>")


def pick_frames(num_frames, count):
    """Positions (into the frames Qwen3-VL saw) of `count` evenly spread frames."""
    return np.linspace(0, num_frames - 1, count).round().astype(int).tolist()


def crop_window(width, height, resolution):
    """The part of the frame (as fractions x0, y0, x1, y1) that torchvision's Resize(resolution) followed by
    CenterCrop(resolution) keeps, as in Dynin-Omni's image_transform."""
    if width <= height:
        new_w, new_h = resolution, int(resolution * height / width)
    else:
        new_w, new_h = int(resolution * width / height), resolution
    left, top = int(round((new_w - resolution) / 2.0)), int(round((new_h - resolution) / 2.0))
    return left / new_w, top / new_h, (left + resolution) / new_w, (top + resolution) / new_h


def convert_label_dynin(label, picked, crop, rows, cols):
    """Move a label onto the picked frames (frame k takes the boxes of the time step it falls in) and into the
    crop, then onto a rows x cols grid. Boxes are clipped to the crop and dropped if nothing is left."""
    x0, y0, x1, y1 = crop
    steps_used = [p // 2 for p in picked]
    boxes, cut = {}, 0
    for k, step in enumerate(steps_used):
        kept = []
        for bx0, by0, bx1, by1 in label["boxes"].get(str(step), []):
            box = [min(max((bx0 - x0) / (x1 - x0), 0), 1), min(max((by0 - y0) / (y1 - y0), 0), 1),
                   min(max((bx1 - x0) / (x1 - x0), 0), 1), min(max((by1 - y0) / (y1 - y0), 0), 1)]
            if box[2] - box[0] > 1e-6 and box[3] - box[1] > 1e-6:
                kept.append([round(v, 4) for v in box])
            else:
                cut += 1
        if kept:
            boxes[str(k)] = kept
    moved = dict(label, grid=[len(picked), rows, cols], boxes=boxes, frames_used=picked, steps_used=steps_used,
                 crop=[round(v, 4) for v in crop],
                 needed_steps=[k for k, step in enumerate(steps_used) if step in label.get("needed_steps", [])])
    return convert_label(moved, rows, cols), cut


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files: which videos to run, and to convert")
    parser.add_argument("--dynin-repo", type=Path, default=Path.home() / "Dynin-Omni")
    parser.add_argument("--model", default="snu-aidas/Dynin-Omni")
    parser.add_argument("--vq-model", default="snu-aidas/magvitv2")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dynin_omni"))
    parser.add_argument("--frames", type=int, default=5, help="Frames per video (Dynin-Omni was trained on 5)")
    parser.add_argument("--resolution", type=int, default=224, help="Frame size (Dynin-Omni's video setting)")
    parser.add_argument("--gen-length", type=int, default=2, help="Answer canvas length (Dynin-Omni's default for multiple choice)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(args.dynin_repo))
    from PIL import Image
    from transformers import AutoTokenizer
    from models import DyninOmniModelLM, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform

    labels = {}
    for path in args.labels:
        data = json.load(open(path))
        labels[path.name] = data
    wanted = sorted({(l["category"], l["video"]) for d in labels.values() for l in d["labels"]})
    questions = {}
    for category in {c for c, _ in wanted}:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})

    def video_path(category, video):
        return args.data_dir / "video" / category / category / video

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "labels").mkdir(exist_ok=True)
    rows_per_frame = cols_per_frame = args.resolution // 16    # MAGVIT-v2 downsamples by 16
    expected_frames, total_cut = {}, 0
    for name, data in labels.items():
        converted = []
        for label in data["labels"]:
            cap = cv2.VideoCapture(str(video_path(label["category"], label["video"])))
            width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            num_frames = 2 * label["grid"][0]
            expected_frames[(label["category"], label["video"])] = num_frames
            moved, cut = convert_label_dynin(label, pick_frames(num_frames, args.frames),
                                             crop_window(width, height, args.resolution), rows_per_frame, cols_per_frame)
            converted.append(moved)
            total_cut += cut
        json.dump(dict(data, labels=converted), open(args.out_dir / "labels" / name, "w"), indent=1)
    print(f"Labels converted; {total_cut} boxes fell outside the centre crop and were dropped")

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=2048, special_tokens=SPECIAL_TOKENS, ignore_id=-100,
                                       cond_dropout_prob=0.0, use_reserved_token=True)
    vq_model = MAGVITv2.from_pretrained(args.vq_model).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = DyninOmniModelLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    blocks = model.model.transformer.blocks
    for block in blocks:
        block.flash_attn_func = None           # route attention through torch's sdpa, where the recorder sits
    recorder = SdpaRowRecorder()
    per_frame = rows_per_frame * cols_per_frame
    special = {name: torch.tensor([[int(uni_prompting.sptids_dict[name])]], device=device)
               for name in ("<|v2t|>", "<|soi|>", "<|eoi|>", "<|sot|>")}
    print(f"Model loaded: {len(blocks)} layers")

    for n, (category, video) in enumerate(todo, start=1):
        item = questions[(category, video)]
        row = {"category": category, "video": video, "question": item["question"], "candidates": item["candidates"],
               "answer": item["answer"], "answer_letter": LETTERS[item["candidates"].index(item["answer"])]}
        start = time.time()
        try:
            frames, indices, native_fps = sample_frames(video_path(category, video))
            if len(frames) != expected_frames[(category, video)]:
                raise RuntimeError(f"{len(frames)} frames read, but the label has {expected_frames[(category, video)]}")
            picked = pick_frames(len(frames), args.frames)
            pixels = torch.stack([image_transform(Image.fromarray(frames[p]), resolution=args.resolution)
                                  for p in picked]).to(device)
            prompt = ("<|start_header_id|>user<|end_header_id|>\n" + build_prompt(item)
                      + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
            query_ids = torch.tensor(uni_prompting.text_tokenizer([prompt])["input_ids"], device=device)

            with torch.inference_mode():
                video_tokens = (vq_model.get_code(pixels) + len(uni_prompting.text_tokenizer)).view(1, -1)
                input_ids = torch.cat([special["<|v2t|>"], special["<|soi|>"], video_tokens, special["<|eoi|>"],
                                       special["<|sot|>"], query_ids], dim=1)
                video_start, video_len, prompt_len = 2, video_tokens.shape[1], input_ids.shape[1]
                if video_len != len(picked) * per_frame:
                    raise RuntimeError(f"{video_len} video tokens for {len(picked)} frames (expected {per_frame} each)")
                canvas = torch.cat([input_ids, torch.full((1, args.gen_length), MASK_ID, device=device)], dim=1)
                with recorder.record(prompt_len):
                    model(canvas)
                rows = torch.stack(recorder.rows).numpy()                          # [layers, sequence]
                if rows.shape[0] != len(blocks):
                    raise RuntimeError(f"Recorded {rows.shape[0]} attention calls for {len(blocks)} layers")
                output = model.mmu_generate(input_ids, max_new_tokens=args.gen_length, steps=args.gen_length,
                                            block_length=args.gen_length)
            response = uni_prompting.text_tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0].strip()

            video_rows = rows[:, video_start:video_start + video_len]
            steps = video_rows.reshape(rows.shape[0], len(picked), rows_per_frame, cols_per_frame)
            attention_file = Path(category) / (Path(video).stem + ".npz")
            (args.out_dir / category).mkdir(exist_ok=True)
            np.savez_compressed(args.out_dir / attention_file, vision_attention=steps[None].astype(np.float32),
                                frame_times=np.array([indices[p] / native_fps for p in picked], dtype=np.float32),
                                frame_indices=np.array([indices[p] for p in picked]))
            pred = parse_letter(response, item["candidates"])
            row.update(response=response, pred_letter=pred, correct=pred == row["answer_letter"],
                       attention_file=str(attention_file), grid_thw=[len(picked), rows_per_frame, cols_per_frame],
                       frames_used=picked, vision_share=float(video_rows.sum(axis=1).mean()))
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
