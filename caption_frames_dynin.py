"""
Describe each of the 5 frames Dynin-Omni sees in one sentence, as the summary a text planner reads.

TabAlign's planner never saw the table, only compact metadata about it. For video, the summary is a short
description of each frame: Dynin-Omni looks at the same 5 frames run_attention_dynin.py gives it (centre-cropped),
one at a time in its image-to-text mode at its image resolution, and describes each. make_plans_llada.py --captions
then plans from these. Videos are the same sample make_plans_dynin.py and make_plans_llada.py use.

Writes <out> (default outputs/frame_captions.jsonl): category, video, captions (5 sentences). Resumes where it stopped.
Needs the Dynin-Omni environment (see README).
"""
import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import torch

from make_plans_dynin import load_done, load_frames, sample_videos
from run_attention_dynin import SPECIAL_TOKENS

PROMPT = "Describe this image in one sentence."


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files to sample videos from")
    parser.add_argument("--per-category", type=int, default=10, help="Videos per category")
    parser.add_argument("--dynin-repo", type=Path, default=Path.home() / "Dynin-Omni")
    parser.add_argument("--model", default="snu-aidas/Dynin-Omni")
    parser.add_argument("--vq-model", default="snu-aidas/magvitv2")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out", type=Path, default=Path("outputs/frame_captions.jsonl"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=336, help="Dynin-Omni's image-to-text resolution")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.dynin_repo))
    from PIL import Image
    from transformers import AutoTokenizer
    from models import DyninOmniModelLM, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform

    wanted, labels, _ = sample_videos(args.labels, args.per_category, args.seed, args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

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
               for name in ("<|mmu|>", "<|soi|>", "<|eoi|>", "<|sot|>")}
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    prompt = "<|start_header_id|>user<|end_header_id|>\n" + PROMPT + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    query_ids = torch.tensor(uni_prompting.text_tokenizer([prompt])["input_ids"], device=device)

    for n, key in enumerate(todo, start=1):
        row = {"category": key[0], "video": key[1]}
        start = time.time()
        try:
            frames, picked, _ = load_frames(key, labels[key], args.data_dir, args.frames, args.resolution)
            captions = []
            for p in picked:
                pixels = image_transform(Image.fromarray(frames[p]), resolution=args.resolution).to(device)[None]
                with torch.inference_mode():
                    image_tokens = vq_model.get_code(pixels) + len(uni_prompting.text_tokenizer)
                    input_ids = torch.cat([special["<|mmu|>"], special["<|soi|>"], image_tokens, special["<|eoi|>"],
                                           special["<|sot|>"], query_ids], dim=1)
                    output = model.mmu_generate(input_ids, max_new_tokens=args.gen_length, steps=args.steps,
                                                block_length=args.gen_length)
                generated = output[0, input_ids.shape[1]:].tolist()
                if eot_id in generated:
                    generated = generated[:generated.index(eot_id)]
                text = uni_prompting.text_tokenizer.decode(generated, skip_special_tokens=True)
                captions.append(re.sub(r"<start of description>|\s+", " ", text).strip())
            row["captions"] = captions
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - start, 2)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: {row.get('error') or row['captions'][0][:80]} ({row['seconds']}s)",
              flush=True)
    print(f"Captions: {args.out}")


if __name__ == "__main__":
    main()
