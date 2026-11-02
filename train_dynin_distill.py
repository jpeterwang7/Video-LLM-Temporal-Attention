"""
Fine-tune Dynin-Omni on DyBench answers, optionally pulling its attention toward Qwen3-VL's.

    loss = cross entropy on the answer letter + attn_weight * KL(teacher attention || Dynin-Omni attention)

Dynin-Omni's attention is read as in run_attention_dynin.py: from the first masked answer slot, over the video, and
averaged over the trained layers. The attention term applies only to videos with a teacher map (those Qwen3-VL
answered correctly). --teacher shuffled gives each video another video's map, as a control.

Only the last --train-layers blocks are trained: the model's bfloat16 weights take about 16 GB, and gradients and
optimizer state for all of them would not fit on a 24 GB card. Those blocks keep float32 master weights and Adam
state on the CPU, so small updates are not lost to bfloat16's rounding. Weights go back to the GPU after each update.

Usage (after prepare_distillation.py, in the Dynin-Omni environment):
    python train_dynin_distill.py --out-dir outputs/distillation/qwen_attention --attn-weight 1

Writes <out-dir>/best.pt (the trained blocks at the best validation accuracy), which run_attention_dynin.py loads with
--checkpoint, and <out-dir>/log.jsonl. Video tokens are cached in <data>/video_tokens, shared between runs.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention_rows import SdpaRowRecorder
from run_attention_dynin import CATEGORIES, SPECIAL_TOKENS, pick_frames
from run_attention_vidlada import LETTERS, MASK_ID, build_prompt, parse_letter, sample_frames


def kl_to_teacher(row, teacher):
    """KL(teacher || student); row and teacher are attention over the same video tokens, row not yet scaled to 1."""
    student = row / row.sum()
    used = teacher > 0
    return (teacher[used] * (teacher[used].log() - student[used].clamp_min(1e-10).log())).sum()


class CpuAdam:
    """AdamW on float32 master copies held on the CPU; the model's parameters only hold their bfloat16 rounding."""

    def __init__(self, parameters, lr, weight_decay):
        self.parameters = list(parameters)
        self.masters = [torch.nn.Parameter(p.detach().float().cpu()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay)

    def step(self, max_norm):
        for parameter, master in zip(self.parameters, self.masters):
            master.grad = parameter.grad.float().cpu()
            parameter.grad = None
        norm = torch.nn.utils.clip_grad_norm_(self.masters, max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, master in zip(self.parameters, self.masters):
            parameter.data.copy_(master.data.to(parameter.dtype))
        return float(norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("outputs/distillation"),
                        help="Folder from prepare_distillation.py")
    parser.add_argument("--attn-weight", type=float, default=1.0, help="Weight of the attention term (0 = answers only)")
    parser.add_argument("--teacher", choices=["qwen", "shuffled"], default="qwen")
    parser.add_argument("--train-layers", type=int, default=7, help="Train this many of the last blocks")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1, help="Stop after this many epochs without a better validation score")
    parser.add_argument("--lr", type=float, default=2e-5, help="Dynin-Omni's own stage 3 learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--accumulate", type=int, default=8, help="Videos per update (Dynin-Omni's own setting)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="Use only this many train and validation videos, for a quick test")
    parser.add_argument("--dynin-repo", type=Path, default=Path.home() / "Dynin-Omni")
    parser.add_argument("--model", default="snu-aidas/Dynin-Omni")
    parser.add_argument("--vq-model", default="snu-aidas/magvitv2")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--gen-length", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.dynin_repo))
    from PIL import Image
    from transformers import AutoTokenizer
    from models import DyninOmniModelLM, MAGVITv2
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    split = json.load(open(args.data / "split.json"))
    train = [tuple(k) for k in split["train"]]
    val = [tuple(k) for k in split["val"]]
    if args.limit:
        train = sorted(random.Random(args.seed).sample(train, min(args.limit, len(train))))
        val = sorted(random.Random(args.seed).sample(val, min(args.limit, len(val))))
    questions = {}
    for category in CATEGORIES:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({k: str(v) for k, v in vars(args).items()}, open(args.out_dir / "args.json", "w"), indent=1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=2048, special_tokens=SPECIAL_TOKENS, ignore_id=-100,
                                       cond_dropout_prob=0.0, use_reserved_token=True)

    # Video tokens, made once: the MAGVIT-v2 encoder then leaves the GPU, and training never needs the videos
    cache = args.data / "video_tokens"
    todo = [k for k in train + val if not (cache / k[0] / (Path(k[1]).stem + ".npz")).exists()]
    if todo:
        vq_model = MAGVITv2.from_pretrained(args.vq_model).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        for n, (category, video) in enumerate(todo, start=1):
            frames, _, _ = sample_frames(args.data_dir / "video" / category / category / video)
            pixels = torch.stack([image_transform(Image.fromarray(frames[p]), resolution=args.resolution)
                                  for p in pick_frames(len(frames), args.frames)]).to(device)
            with torch.inference_mode():
                tokens = vq_model.get_code(pixels) + len(uni_prompting.text_tokenizer)
            out = cache / category / (Path(video).stem + ".npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, tokens=tokens.view(-1).cpu().numpy(), num_frames=len(frames))
            if n % 100 == 0:
                print(f"Video tokens: {n}/{len(todo)}", flush=True)
        del vq_model
        torch.cuda.empty_cache()

    special = {name: torch.tensor([[int(uni_prompting.sptids_dict[name])]], device=device)
               for name in ("<|v2t|>", "<|soi|>", "<|eoi|>", "<|sot|>")}

    def input_for(key):
        """(input_ids, video_len, num_frames); the same sequence run_attention_dynin.py builds, video from 2."""
        saved = np.load(cache / key[0] / (Path(key[1]).stem + ".npz"))
        video_tokens = torch.from_numpy(saved["tokens"]).long().to(device)[None]
        prompt = ("<|start_header_id|>user<|end_header_id|>\n" + build_prompt(questions[key])
                  + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
        query_ids = torch.tensor(uni_prompting.text_tokenizer([prompt])["input_ids"], device=device)
        ids = torch.cat([special["<|v2t|>"], special["<|soi|>"], video_tokens, special["<|eoi|>"], special["<|sot|>"],
                         query_ids], dim=1)
        return ids, video_tokens.shape[1], int(saved["num_frames"])

    # Teacher maps; --teacher shuffled hands each training video a map from another video
    def load_map(key):
        path = args.data / "maps" / key[0] / (Path(key[1]).stem + ".npz")
        return np.load(path) if path.exists() else None

    has_map = [k for k in train if load_map(k) is not None]
    donors = dict(zip(has_map, has_map))
    if args.teacher == "shuffled":
        shuffled = has_map[:]
        random.Random(args.seed).shuffle(shuffled)
        # Twins sit next to each other in has_map, so never hand a video its twin's map
        for i in range(len(has_map)):
            if shuffled[i][1].replace("_reversed", "") == has_map[i][1].replace("_reversed", ""):
                j = (i + 1) % len(has_map)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        donors = dict(zip(has_map, shuffled))
    print(f"{len(train)} train videos, {len(has_map)} with a teacher map; {len(val)} validation videos; "
          f"teacher {args.teacher}, attention weight {args.attn_weight}")

    model = DyninOmniModelLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.requires_grad_(False)
    blocks = model.model.transformer.blocks
    trained = list(range(len(blocks) - args.train_layers, len(blocks)))
    for index in trained:
        blocks[index].requires_grad_(True)
    for block in blocks:
        block.flash_attn_func = None           # route attention through torch's sdpa, where the recorder sits
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = CpuAdam(parameters, args.lr, args.weight_decay)
    recorder = SdpaRowRecorder(to_cpu=False)
    log_path = args.out_dir / "log.jsonl"
    print(f"Model loaded: training layers {trained[0] + 1}-{trained[-1] + 1} of {len(blocks)}, "
          f"{sum(p.numel() for p in parameters) / 1e6:.0f}M parameters")

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def validate():
        right = 0
        with torch.inference_mode():
            for key in val:
                item = questions[key]
                ids, _, _ = input_for(key)
                output = model.mmu_generate(ids, max_new_tokens=args.gen_length, steps=args.gen_length,
                                            block_length=args.gen_length)
                response = uni_prompting.text_tokenizer.batch_decode(output[:, ids.shape[1]:], skip_special_tokens=True)[0]
                right += parse_letter(response, item["candidates"]) == LETTERS[item["candidates"].index(item["answer"])]
        torch.cuda.empty_cache()
        return right / max(len(val), 1)

    def save_best():
        names = {id(p): n for n, p in model.named_parameters()}
        state = {names[id(p)]: p.detach().cpu() for p in parameters}
        torch.save({"state": state}, args.out_dir / "best.pt")

    best = validate()
    save_best()
    log({"epoch": 0, "val_accuracy": best})
    stale = 0
    for epoch in range(1, args.epochs + 1):
        order = train[:]
        random.Random(args.seed + epoch).shuffle(order)
        totals = {"answer": 0.0, "attention": 0.0, "with_map": 0}
        start = time.time()
        for n, key in enumerate(order, start=1):
            item = questions[key]
            ids, video_len, num_frames = input_for(key)
            canvas = torch.cat([ids, torch.full((1, args.gen_length), MASK_ID, device=device)], dim=1)
            with recorder.record(ids.shape[1]):
                logits = model(canvas).logits

            # The first slot holds the letter; the second holds "." or the end token, which doesn't change the answer
            letter = LETTERS[item["candidates"].index(item["answer"])]
            target = tokenizer.encode(letter, add_special_tokens=False)
            answer_loss = F.cross_entropy(logits[0, ids.shape[1]].float()[None], torch.tensor(target[:1], device=device))
            loss = answer_loss
            totals["answer"] += float(answer_loss)

            donor = donors.get(key)
            saved = load_map(donor) if donor is not None else None
            if saved is not None and args.attn_weight > 0 and int(load_map(key)["num_frames"]) == num_frames:
                row = torch.stack(recorder.rows[-args.train_layers:]).mean(dim=0)[2:2 + video_len]
                teacher = torch.from_numpy(saved["map"].ravel()).to(row)
                attention_loss = kl_to_teacher(row, teacher)
                loss = loss + args.attn_weight * attention_loss
                totals["attention"] += float(attention_loss)
                totals["with_map"] += 1
            recorder.rows = []

            (loss / args.accumulate).backward()
            if n % args.accumulate == 0 or n == len(order):
                optimizer.step(args.max_grad_norm)
            if n % 100 == 0 or n == len(order):
                print(f"epoch {epoch} [{n}/{len(order)}] answer loss {totals['answer'] / n:.4f}, attention KL "
                      f"{totals['attention'] / max(totals['with_map'], 1):.4f}, {(time.time() - start) / n:.2f}s per video, peak GPU "
                      f"{torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f} GB", flush=True)

        accuracy = validate()
        log({"epoch": epoch, "answer_loss": totals["answer"] / len(order),
             "attention_kl": totals["attention"] / max(totals["with_map"], 1), "val_accuracy": accuracy,
             "minutes": round((time.time() - start) / 60, 1)})
        if accuracy > best:
            best, stale = accuracy, 0
            save_best()
        else:
            stale += 1
            if stale >= args.patience:
                print(f"No better validation score for {stale} epoch(s); stopping")
                break
    print(f"Best validation accuracy {100 * best:.1f}%. Saved to {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
