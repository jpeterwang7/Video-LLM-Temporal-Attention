"""
Step 4: train Dynin-Omni to carry out LLaDA-8B-Instruct's plans the way Qwen3-VL did.

Training examples come from Qwen3-VL's traces on the training videos (run_step3.sh), keeping those that end in the
right answer and whose steps do not argue against it (see usable). Each turn of a trace is one example. Dynin-Omni
sees its 5 frames, then the same conversation Qwen3-VL had in execute_plans_qwen.py up to that turn: the question, the
earlier steps with Qwen3-VL's answers, and the current step. It learns to write Qwen3-VL's answer. The last turn asks
for the option's letter. Qwen3-VL's answers often give times ("action 1 starts at 0:03"), which Qwen3-VL reads off its
own frames, so Dynin-Omni's first message also says when its 5 frames are ("The video's 5 frames are at 0.0s, 7.0s,
...").

The loss is Dynin-Omni's own text loss from its stage 3 training (masked diffusion). The reply canvas is masked at a
random rate t, and the cross entropy of the masked tokens, each divided by t, is summed and divided by --step-tokens.
For a step, the canvas is the reply, then the end token up to --step-tokens. For the last turn, it is the letter and
the end token. As in Dynin-Omni's training, only the reply and its first end token are trained: the filler end tokens
are left out, and every trained token weighs the same. Training on the filler taught it to fill the canvas with end
tokens and answer every step with one letter. The learning rate rises linearly over the first --warmup updates.

Attention anchor: Dynin-Omni's attention finds the labelled objects (check_dynin_attention.py), so training should not
move where it looks. Before training, the untouched model's attention is recorded for every example in each trained
layer: from the first reply slot over the video, with the canvas fully masked (as run_attention_dynin.py measures
it). Each example then adds a pass with the canvas fully masked, with loss --attn-weight x KL(untouched || current),
averaged over those layers. --attn-weight 0 turns it off.

Only the last --train-layers blocks are trained, with float32 master weights and AdamW on the CPU: gradients and
optimizer state for the whole model do not fit on a 24 GB card.

Validation: Dynin-Omni carries out the plans of every planned validation video, step by step, whether Qwen3-VL got
the video right or wrong. Accuracy is the share it gets right. It is measured on the untouched model first and after
each epoch. The weights with the best validation accuracy are kept.

Usage (in the Dynin-Omni environment, after run_step3.sh):
    python train_dynin_reasoning.py --out-dir outputs/dynin_reasoning

Writes to <out-dir>:
    best.pt                   the trained blocks at the best validation accuracy
    log.jsonl                 losses and validation accuracy per epoch
    val_traces_<epoch>.jsonl  Dynin-Omni's steps and answers on the validation videos (epoch 0 is the untouched model)
Video tokens are cached in outputs/dynin_video_tokens.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention_rows import SdpaRowRecorder
from execute_plans_qwen import FINAL, first_step_only, opening, step_prompt
from make_plans_llada import load_done
from run_attention_dynin import CATEGORIES, SPECIAL_TOKENS, pick_frames
from run_attention_vidlada import LETTERS, MASK_ID, parse_letter, sample_frames

USER = "<|start_header_id|>user<|end_header_id|>\n"
ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n"
EOT = "<|eot_id|>"
# In reversible dynamics "nothing happened" is never the answer, so a step concluding it argues against the answer
NOTHING_HAPPENED = re.compile(
    r"\bneither\b|nothing happen|no (?:significant )?(?:change|movement|motion)|did not (?:move|change)|"
    r"does not (?:move|change)|not (?:moved|changed)|stay(?:s|ed)? still|"
    r"remain(?:s|ed)? (?:still|stationary|unchanged|the same)|(?:same|unchanged|consistent) throughout", re.I)


def plain(text):
    return re.sub(r"[^a-z ]", "", text.lower())


def usable(trace):
    """A trace to learn from: it ends in the right answer, and its steps do not argue against it (a reversible
    dynamics step saying nothing happened, or steps naming a wrong option but never the right one)."""
    if not trace.get("correct"):
        return False
    replies = [t["reply"] for t in trace["turns"]]
    if trace["category"] == "reversible_dynamics" and any(NOTHING_HAPPENED.search(r) for r in replies):
        return False
    said = plain(" ".join(replies))
    wrong = [c for c in trace["candidates"] if c != trace["answer"]]
    return not (any(plain(c) in said for c in wrong) and plain(trace["answer"]) not in said)


def conversation(item, times, prompts, replies):
    """Chat text up to the assistant's next reply: prompts[i] was answered by replies[i], the last prompt not yet."""
    when = ", ".join(f"{t:.1f}s" for t in times[:-1]) + f" and {times[-1]:.1f}s"
    text = ""
    for i, prompt in enumerate(prompts):
        first = f"The video's {len(times)} frames are at {when}.\n" + opening(item, prompt)
        text += USER + (first if i == 0 else prompt) + EOT + ASSISTANT
        if i < len(replies):
            text += replies[i] + EOT
    return text


class CpuAdam:
    """AdamW on float32 master copies held on the CPU; the model's parameters only hold their bfloat16 rounding."""

    def __init__(self, parameters, lr, weight_decay, warmup):
        self.parameters = list(parameters)
        self.masters = [torch.nn.Parameter(p.detach().float().cpu()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay)
        self.lr, self.warmup, self.steps = lr, warmup, 0

    def step(self, max_norm):
        self.steps += 1
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * min(1.0, self.steps / max(self.warmup, 1))
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
    parser.add_argument("--traces", type=Path, default=Path("outputs/traces_train/traces.jsonl"))
    parser.add_argument("--plans", type=Path, default=Path("outputs/plans_train/plans.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("outputs/split.json"))
    parser.add_argument("--direct", type=Path, default=Path("outputs/dynin_omni/results.jsonl"),
                        help="Dynin-Omni's direct answers, for comparison")
    parser.add_argument("--attn-weight", type=float, default=1.0, help="Weight of the attention anchor (0 = none)")
    parser.add_argument("--train-layers", type=int, default=7, help="Train this many of the last blocks")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1, help="Stop after this many epochs without a better validation score")
    parser.add_argument("--lr", type=float, default=2e-5, help="Dynin-Omni's own stage 3 learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--accumulate", type=int, default=32, help="Examples (turns) per update")
    parser.add_argument("--warmup", type=int, default=20, help="Updates over which the learning rate rises to --lr")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--step-tokens", type=int, default=64, help="Canvas for each step's answer")
    parser.add_argument("--step-steps", type=int, default=32, help="Denoising steps for each step's answer")
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Use only this many training videos, for a test")
    parser.add_argument("--val-limit", type=int, help="Use only this many validation videos, for a test")
    parser.add_argument("--dynin-repo", type=Path, default=Path.home() / "Dynin-Omni")
    parser.add_argument("--model", default="snu-aidas/Dynin-Omni")
    parser.add_argument("--vq-model", default="snu-aidas/magvitv2")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--tokens-dir", type=Path, default=Path("outputs/dynin_video_tokens"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=224)
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
    questions = {}
    for category in CATEGORIES:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})
    split = json.load(open(args.split))
    in_train, in_val = {tuple(k) for k in split["train"]}, {tuple(k) for k in split["val"]}
    traces, plans = load_done(args.traces), load_done(args.plans)
    if set(traces) & {tuple(k) for k in split["test"]}:
        raise RuntimeError("Some traces are of test videos; the split has changed since run_step3.sh")
    train = sorted(k for k, t in traces.items() if k in in_train and usable(t))
    val = sorted(k for k, p in plans.items() if k in in_val and p.get("steps"))
    print(f"{sum(k in in_train and t.get('correct') for k, t in traces.items())} training traces end in the right "
          f"answer; {len(train)} of them are usable. {len(val)} planned validation videos.")
    if args.limit:
        train = sorted(random.Random(args.seed).sample(train, min(args.limit, len(train))))
    if args.val_limit:
        val = sorted(random.Random(args.seed).sample(val, min(args.val_limit, len(val))))
    examples = [(key, turn) for key in train for turn in range(len(traces[key]["turns"]) + 1)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({k: str(v) for k, v in vars(args).items()}, open(args.out_dir / "args.json", "w"), indent=1)
    log_path = args.out_dir / "log.jsonl"
    log_path.write_text("")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=2048, special_tokens=SPECIAL_TOKENS, ignore_id=-100,
                                       cond_dropout_prob=0.0, use_reserved_token=True)
    eos_id = tokenizer.eos_token_id
    stop_ids = {eos_id, tokenizer.convert_tokens_to_ids(EOT)}

    # Video tokens, made once: the MAGVIT-v2 encoder then leaves the GPU
    def token_path(key):
        return args.tokens_dir / key[0] / (Path(key[1]).stem + ".npz")

    todo = [k for k in train + val if not token_path(k).exists()]
    if todo:
        vq_model = MAGVITv2.from_pretrained(args.vq_model).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        for n, (category, video) in enumerate(todo, start=1):
            frames, indices, native_fps = sample_frames(args.data_dir / "video" / category / category / video)
            picked = pick_frames(len(frames), args.frames)
            pixels = torch.stack([image_transform(Image.fromarray(frames[p]), resolution=args.resolution)
                                  for p in picked]).to(device)
            with torch.inference_mode():
                tokens = vq_model.get_code(pixels) + len(uni_prompting.text_tokenizer)
            token_path((category, video)).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(token_path((category, video)), tokens=tokens.view(-1).cpu().numpy(),
                                times=np.array([indices[p] / native_fps for p in picked]))
            if n % 100 == 0 or n == len(todo):
                print(f"Video tokens: {n}/{len(todo)}", flush=True)
        del vq_model
        torch.cuda.empty_cache()
    saved = {k: np.load(token_path(k)) for k in train + val}
    video_tokens = {k: s["tokens"] for k, s in saved.items()}
    frame_times = {k: s["times"].tolist() for k, s in saved.items()}
    special = {name: int(uni_prompting.sptids_dict[name]) for name in ("<|v2t|>", "<|soi|>", "<|eoi|>", "<|sot|>")}

    def prompt_ids(key, text):
        """The sequence run_attention_dynin.py builds: task token, video (from position 2), then the chat text."""
        text_ids = uni_prompting.text_tokenizer([text])["input_ids"][0]
        ids = ([special["<|v2t|>"], special["<|soi|>"]] + video_tokens[key].tolist() + [special["<|eoi|>"], special["<|sot|>"]]
               + text_ids)
        return torch.tensor([ids], device=device)

    def example(key, turn):
        """(prompt ids, target canvas, number of trained tokens) for one turn of a training trace."""
        trace, item = traces[key], questions[key]
        prompts = [t["prompt"] for t in trace["turns"]] + [FINAL]
        replies = [t["reply"] for t in trace["turns"]]
        ids = prompt_ids(key, conversation(item, frame_times[key], prompts[:turn + 1], replies[:turn]))
        if turn < len(replies):
            target = tokenizer(replies[turn], add_special_tokens=False)["input_ids"][:args.step_tokens - 1]
            length = args.step_tokens
        else:
            target = tokenizer(LETTERS[item["candidates"].index(item["answer"])], add_special_tokens=False)["input_ids"]
            length = 2
        return ids, torch.tensor(target + [eos_id] * (length - len(target)), device=device), len(target) + 1

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
    recorder = SdpaRowRecorder(to_cpu=False)
    video_len = len(next(iter(video_tokens.values())))
    print(f"Model loaded: training layers {trained[0] + 1}-{trained[-1] + 1} of {len(blocks)} "
          f"({sum(p.numel() for p in parameters) / 1e6:.0f}M parameters). {len(train)} training videos, "
          f"{len(examples)} examples; {len(val)} validation videos.", flush=True)

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def video_attention(ids, length):
        """Each trained layer's attention from the first reply slot over the video, canvas fully masked, scaled to 1."""
        canvas = torch.cat([ids, torch.full((1, length), MASK_ID, device=device)], dim=1)
        with recorder.record(ids.shape[1]):
            model(canvas)
        if len(recorder.rows) != len(blocks):
            raise RuntimeError(f"Recorded {len(recorder.rows)} attention calls for {len(blocks)} layers")
        rows = torch.stack(recorder.rows[-args.train_layers:])[:, 2:2 + video_len]
        recorder.rows = []
        return rows / rows.sum(dim=1, keepdim=True)

    def generate(key, text, length, steps):
        ids = prompt_ids(key, text)
        with torch.no_grad():
            output = model.mmu_generate(ids, max_new_tokens=length, steps=steps, block_length=min(length, args.block_length))
        reply = output[0, ids.shape[1]:].tolist()
        stops = [i for i, token in enumerate(reply) if token in stop_ids]
        return tokenizer.decode(reply[:stops[0]] if stops else reply, skip_special_tokens=True).strip()

    def validate(epoch):
        """Carry out each validation video's plan step by step; returns accuracy and writes the traces."""
        rows = []
        for key in val:
            item, plan = questions[key], plans[key]
            prompts, replies = [], []
            for number, step in enumerate(plan["steps"], start=1):
                prompts.append(step_prompt(number, step))
                text = conversation(item, frame_times[key], prompts, replies)
                replies.append(first_step_only(generate(key, text, args.step_tokens, args.step_steps)))
            prompts.append(FINAL)
            final = generate(key, conversation(item, frame_times[key], prompts, replies), 2, 2)
            letter = parse_letter(final, item["candidates"])
            rows.append({"category": key[0], "video": key[1], "question": item["question"],
                         "candidates": item["candidates"], "answer": item["answer"],
                         "turns": [{"prompt": p, "reply": r} for p, r in zip(prompts, replies)], "final": final,
                         "pred_letter": letter, "correct": letter == LETTERS[item["candidates"].index(item["answer"])]})
        torch.cuda.empty_cache()
        with open(args.out_dir / f"val_traces_{epoch}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        for r in rows[:3]:
            print(f"  {r['video']}: final {r['final']!r} ({'right' if r['correct'] else 'wrong'})\n" +
                  "\n".join(f"    {t['reply'][:150]}" for t in r["turns"]), flush=True)
        by_category = {c: f"{sum(r['correct'] for r in rows if r['category'] == c)}/{sum(r['category'] == c for r in rows)}"
                       for c in CATEGORIES}
        return sum(r["correct"] for r in rows) / max(len(rows), 1), by_category

    def save_best():
        names = {id(p): n for n, p in model.named_parameters()}
        torch.save({"state": {names[id(p)]: p.detach().cpu() for p in parameters}}, args.out_dir / "best.pt")

    # For comparison: Qwen3-VL following the same plans, and Dynin-Omni answering directly
    direct = load_done(args.direct)
    qwen = [traces[k]["correct"] for k in val if k in traces and "correct" in traces[k]]
    dynin = [direct[k]["correct"] for k in val if k in direct]
    print(f"Validation videos: Qwen3-VL following the plans {sum(qwen)}/{len(qwen)}; Dynin-Omni answering directly "
          f"{sum(dynin)}/{len(dynin)}", flush=True)

    # The untouched model's attention, the anchor's target
    anchors = {}
    if args.attn_weight > 0:
        start = time.time()
        with torch.no_grad():
            for n, (key, turn) in enumerate(examples, start=1):
                ids, target, _ = example(key, turn)
                anchors[(key, turn)] = video_attention(ids, len(target)).float().cpu()
                if n % 500 == 0 or n == len(examples):
                    print(f"Anchor targets: {n}/{len(examples)} ({(time.time() - start) / n:.2f}s each)", flush=True)

    start = time.time()
    best, by_category = validate(0)
    save_best()
    log({"epoch": 0, "val_accuracy": best, "val_by_category": by_category, "minutes": round((time.time() - start) / 60, 1)})
    optimizer = CpuAdam(parameters, args.lr, args.weight_decay, args.warmup)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        order = examples[:]
        random.Random(args.seed + epoch).shuffle(order)
        totals = {"text": 0.0, "anchor": 0.0}
        start = time.time()
        for n, (key, turn) in enumerate(order, start=1):
            ids, target, answer_len = example(key, turn)
            # Dynin-Omni's text loss: mask the canvas at a random rate t, predict the masked tokens of the reply and its
            # end token, weight them 1/t
            t = (1 - 1e-3) * random.random() + 1e-3
            masked = torch.rand(len(target), device=device) < t
            if not masked[:answer_len].any():
                masked[random.randrange(answer_len)] = True
            canvas = torch.where(masked, torch.full_like(target, MASK_ID), target)
            logits = model(torch.cat([ids, canvas[None]], dim=1)).logits[0, ids.shape[1]:ids.shape[1] + answer_len]
            scored = masked[:answer_len]
            text_loss = (F.cross_entropy(logits[scored].float(), target[:answer_len][scored], reduction="none") / t).sum() \
                / args.step_tokens
            (text_loss / args.accumulate).backward()
            totals["text"] += float(text_loss)
            del logits

            if args.attn_weight > 0:
                current = video_attention(ids, len(target))
                untouched = anchors[(key, turn)].to(device)
                anchor_loss = (untouched * (untouched.log() - current.float().clamp_min(1e-10).log())).sum(dim=1).mean()
                (args.attn_weight * anchor_loss / args.accumulate).backward()
                totals["anchor"] += float(anchor_loss)

            if n % args.accumulate == 0 or n == len(order):
                optimizer.step(args.max_grad_norm)
            if n % 200 == 0 or n == len(order):
                print(f"epoch {epoch} [{n}/{len(order)}] text loss {totals['text'] / n:.4f}, attention KL "
                      f"{totals['anchor'] / n:.5f}, {(time.time() - start) / n:.2f}s per example, peak GPU "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

        train_minutes = round((time.time() - start) / 60, 1)
        accuracy, by_category = validate(epoch)
        log({"epoch": epoch, "text_loss": totals["text"] / len(order), "attention_kl": totals["anchor"] / len(order),
             "val_accuracy": accuracy, "val_by_category": by_category, "train_minutes": train_minutes,
             "minutes": round((time.time() - start) / 60, 1)})
        if accuracy > best:
            best, stale = accuracy, 0
            save_best()
        else:
            stale += 1
            if stale >= args.patience:
                print(f"No better validation accuracy for {stale} epoch(s); stopping")
                break
    print(f"Best validation accuracy {100 * best:.1f}%. Saved to {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
