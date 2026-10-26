"""
Push attention onto the labelled object and see whether the answers change.

At the position that emits the answer (the one the run_attention scripts read attention from), each layer's
attention logits on the labelled video tokens get +bias, so their attention weights are multiplied by e^bias
before the row is renormalised. Nothing else in the model changes. As a control, the same bias goes to as many
random unlabelled video tokens, drawn from the same time steps, so a gain from looking at the video more is not
mistaken for a gain from looking at the object. The answer is read in one forward pass from that position's
scores for the option letters: the chosen letter and the probability of the correct one.

One model per run, in that model's environment (see README):
    python steer_attention.py qwen3vl labels/labels_peter.json labels/labels_peter_labeling_round2.json
    python steer_attention.py llava_video outputs/llava_video_7b/labels/labels_peter.json outputs/llava_video_7b/labels/labels_peter_labeling_round2.json
    python steer_attention.py vidlada outputs/vidlada_8b/labels/labels_peter.json outputs/vidlada_8b/labels/labels_peter_labeling_round2.json
    python steer_attention.py dynin outputs/dynin_omni/labels/labels_peter.json outputs/dynin_omni/labels/labels_peter_labeling_round2.json
    python steer_attention.py dreamvl labels/labels_peter.json labels/labels_peter_labeling_round2.json

Writes outputs/steering/<model>/results.jsonl: one line per video, target (none, object or random) and bias,
with the letter probabilities and the attention share the answer position gave the object, the target and the
whole video (mean over layers and heads). summarize_steering.py turns these into a table and a chart.
"""
import argparse
import copy
import json
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import torch

from attention_rows import SdpaSteerer
from run_attention_vidlada import LETTERS, MASK_ID, build_prompt, sample_frames


class Case:
    """One video, ready to run: sequence length, the answer-emitting row, the sequence position of every video
    token as [T, H, W, tokens per cell], and a forward pass returning that row's next-token scores."""

    def __init__(self, seq_len, row, positions, forward):
        self.seq_len, self.row, self.positions, self.forward = seq_len, row, positions, forward


def label_mask(label):
    """[T, H, W] cells the label marks (as score_attention.label_mask, which needs matplotlib to import)."""
    mask = np.zeros(label["grid"], dtype=bool)
    for t, cells in label["cells"].items():
        for row, col in cells:
            mask[int(t), row, col] = True
    return mask


def letter_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        if len(tokens) != 1:
            raise RuntimeError(f"Letter {letter} is {len(tokens)} tokens for this tokenizer")
        ids.append(tokens[0])
    return ids


def llava_positions(video_start, frames, rows, cols):
    """LLaVA-NeXT video layout (VidLaDA, LLaVA-Video): frame by frame, row by row, an end-of-row token after each
    row. Time step t covers frames 2t and 2t+1."""
    t, r, c, j = np.meshgrid(np.arange(frames // 2), np.arange(rows), np.arange(cols), np.arange(2), indexing="ij")
    return video_start + (2 * t + j) * rows * (cols + 1) + r * (cols + 1) + c


class Qwen3VL:
    name = "qwen3vl_8b"

    def __init__(self, args):
        from omnitrace.backends.qwen3vl import VIDEO_INFO_KEY, load_model_qwen3vl, prepare_inputs_qwen3vl
        from omnitrace.constants import QWEN3VL_PIXELS_PER_TOKEN
        from omnitrace.modalities.video import build_video_messages
        self.info_key, self.prepare_inputs, self.messages = VIDEO_INFO_KEY, prepare_inputs_qwen3vl, build_video_messages
        self.max_pixels = 3072 * QWEN3VL_PIXELS_PER_TOKEN       # run_attention.py's defaults
        self.bundle = load_model_qwen3vl(args.model)
        self.layers = self.bundle.model.config.text_config.num_hidden_layers
        self.letters = letter_ids(self.bundle.tokenizer)

    def prepare(self, item, path, label):
        messages = self.messages(str(path), build_prompt(item), video_max_pixels=self.max_pixels, video_fps=2.0,
                                 video_max_frames=32)
        inputs = self.prepare_inputs(self.bundle, messages)
        steps, rows, cols = inputs.pop(self.info_key)["grid_thw"]
        ids = inputs["input_ids"][0]
        positions = torch.nonzero(ids == self.bundle.vision_token_id).flatten().cpu().numpy().reshape(steps, rows, cols, 1)
        n = ids.shape[0]
        return Case(n, n - 1, positions, lambda: self.bundle.model(**inputs, logits_to_keep=1).logits[0, -1])


class LlavaFamily:
    """VidLaDA and LLaVA-Video share LLaVA-NeXT's input code; they differ in repository, chat template and
    where the answer comes from (the first masked slot, or the last prompt position)."""

    def __init__(self, args, repo, model_name, load_name, conv_name, diffusion):
        sys.path.insert(0, str(args.repo or Path.home() / repo))
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        self.image_token, self.image_index, self.tokenize = DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, tokenizer_image_token
        self.conv = conv_templates[conv_name]
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            args.model or model_name, None, load_name, device_map="cuda:0", torch_dtype="bfloat16",
            attn_implementation="sdpa", overwrite_config={"mm_spatial_pool_stride": 2, "mm_spatial_pool_mode": "bilinear"})
        self.model.eval()
        self.backbone = self.model.get_model()
        self.layers = len(self.backbone.layers)
        self.letters = letter_ids(self.tokenizer)
        self.diffusion = diffusion

    def prepare(self, item, path, label):
        frames, _, _ = sample_frames(path)
        steps, rows, cols = label["grid"]
        if len(frames) != 2 * steps:
            raise RuntimeError(f"{len(frames)} frames read, but the label has {steps} time steps")
        pixels = self.image_processor.preprocess(np.stack(frames), return_tensors="pt")["pixel_values"].to("cuda:0", torch.bfloat16)
        conv = copy.deepcopy(self.conv)
        conv.append_message(conv.roles[0], self.image_token + "\n" + build_prompt(item))
        conv.append_message(conv.roles[1], None)
        input_ids = self.tokenize(conv.get_prompt(), self.tokenizer, self.image_index, return_tensors="pt").unsqueeze(0).to("cuda:0")
        with torch.inference_mode():
            _, _, _, _, embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, torch.ones_like(input_ids, dtype=torch.bool), None, None, [pixels], ["video"], image_sizes=None)
            prompt_len = embeds.shape[1]
            if self.diffusion:   # VidLaDA: a 2-slot masked answer canvas; the first slot emits the answer
                mask = self.backbone.embed_tokens(torch.tensor([MASK_ID], device="cuda:0"))
                embeds = torch.cat([embeds, mask[None].expand(1, 2, -1).to(embeds.dtype)], dim=1)
        video_start = int((input_ids[0] == self.image_index).nonzero()[0])
        if prompt_len - (input_ids.shape[1] - 1) != len(frames) * rows * (cols + 1):
            raise RuntimeError("Video token count does not match the label grid")
        row = prompt_len if self.diffusion else prompt_len - 1
        forward = lambda: self.model.lm_head(self.backbone(inputs_embeds=embeds, use_cache=False)[0][:, row])[0]
        return Case(embeds.shape[1], row, llava_positions(video_start, len(frames), rows, cols), forward)


class LlavaVideo(LlavaFamily):
    name = "llava_video_7b"

    def __init__(self, args):
        super().__init__(args, "LLaVA-NeXT", "lmms-lab/LLaVA-Video-7B-Qwen2", "llava_qwen", "qwen_1_5", diffusion=False)


class VidLaDA(LlavaFamily):
    name = "vidlada_8b"

    def __init__(self, args):
        repo = args.repo or Path.home() / "VidLaDA"
        args.repo = repo / "train"
        super().__init__(args, "VidLaDA/train", "ziHoHe/VidLaDA-8B", "llava_llada_video", "llava_llada", diffusion=True)


class Dynin:
    name = "dynin_omni"

    def __init__(self, args):
        sys.path.insert(0, str(args.repo or Path.home() / "Dynin-Omni"))
        from PIL import Image
        from transformers import AutoTokenizer
        from models import DyninOmniModelLM, MAGVITv2
        from training.prompting_utils import UniversalPrompting
        from training.utils import image_transform
        from run_attention_dynin import SPECIAL_TOKENS, pick_frames
        self.image, self.transform, self.pick = Image, image_transform, pick_frames
        self.device = args.device
        tokenizer = AutoTokenizer.from_pretrained(args.model or "snu-aidas/Dynin-Omni", padding_side="left", trust_remote_code=True)
        self.prompting = UniversalPrompting(tokenizer, max_text_len=2048, special_tokens=SPECIAL_TOKENS, ignore_id=-100,
                                            cond_dropout_prob=0.0, use_reserved_token=True)
        self.vq = MAGVITv2.from_pretrained(args.vq_model or "snu-aidas/magvitv2").to(self.device).eval()
        self.model = DyninOmniModelLM.from_pretrained(args.model or "snu-aidas/Dynin-Omni", torch_dtype=torch.bfloat16).to(self.device).eval()
        for block in self.model.model.transformer.blocks:
            block.flash_attn_func = None
        self.layers = len(self.model.model.transformer.blocks)
        self.letters = letter_ids(tokenizer)
        self.special = {name: torch.tensor([[int(self.prompting.sptids_dict[name])]], device=self.device)
                        for name in ("<|v2t|>", "<|soi|>", "<|eoi|>", "<|sot|>")}

    def prepare(self, item, path, label):
        frames, _, _ = sample_frames(path)
        picked = self.pick(len(frames), label["grid"][0])
        if picked != label["frames_used"]:
            raise RuntimeError(f"Frames {picked} differ from the label's {label['frames_used']}")
        pixels = torch.stack([self.transform(self.image.fromarray(frames[p]), resolution=224) for p in picked]).to(self.device)
        prompt = ("<|start_header_id|>user<|end_header_id|>\n" + build_prompt(item)
                  + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
        query_ids = torch.tensor(self.prompting.text_tokenizer([prompt])["input_ids"], device=self.device)
        with torch.inference_mode():
            video_tokens = (self.vq.get_code(pixels) + len(self.prompting.text_tokenizer)).view(1, -1)
        input_ids = torch.cat([self.special["<|v2t|>"], self.special["<|soi|>"], video_tokens, self.special["<|eoi|>"],
                               self.special["<|sot|>"], query_ids], dim=1)
        prompt_len = input_ids.shape[1]
        canvas = torch.cat([input_ids, torch.full((1, 2), MASK_ID, device=self.device)], dim=1)
        steps, rows, cols = label["grid"]
        positions = 2 + np.arange(steps * rows * cols).reshape(steps, rows, cols, 1)
        return Case(canvas.shape[1], prompt_len, positions, lambda: self.model(canvas).logits[0, prompt_len])


class DreamVL:
    name = "dreamvl_7b"

    def __init__(self, args):
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
        from run_attention_dreamvl import SQUARE
        self.image, self.square, self.device = Image, SQUARE, args.device
        name = args.model or "Dream-org/Dream-VL-7B"
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                               attn_implementation="sdpa").to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, min_pixels=SQUARE * SQUARE,
                                                       max_pixels=SQUARE * SQUARE * 100000)
        self.layers = len(self.model.model.layers)
        self.letters = letter_ids(self.processor.tokenizer)

    def prepare(self, item, path, label):
        frames, _, _ = sample_frames(path)
        steps, rows, cols = label["grid"]
        if len(frames) != 2 * steps:
            raise RuntimeError(f"{len(frames)} frames read, but the label has {steps} time steps")
        clip = [self.image.fromarray(f).resize((cols * self.square, rows * self.square), self.image.BICUBIC) for f in frames]
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": build_prompt(item)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[clip], padding=True, return_tensors="pt").to(self.device)
        input_ids, grid = inputs["input_ids"], inputs["video_grid_thw"]
        prompt_len = input_ids.shape[1]
        canvas = torch.cat([input_ids, torch.full((1, 3), self.model.config.mask_token_id, device=self.device)], dim=1)
        with torch.inference_mode():
            embeds = self.model.get_input_embeddings()(canvas)
            is_video = canvas == self.model.config.video_token_id
            embeds[is_video] = self.model.projector(torch.cat(self.model.get_video_features(inputs["pixel_values_videos"], grid))).to(embeds.dtype)
            position_ids, _ = self.model.get_rope_index(canvas, None, grid, None)
        positions = torch.nonzero(is_video[0]).flatten().cpu().numpy().reshape(steps, rows, cols, 1)
        row = prompt_len - 1      # Dream reads the token at position i from position i-1
        forward = lambda: self.model.lm_head(self.model.model(inputs_embeds=embeds, position_ids=position_ids,
                                                              use_cache=False)[0][:, row])[0]
        return Case(canvas.shape[1], row, positions, forward)


MODELS = {"qwen3vl": Qwen3VL, "llava_video": LlavaVideo, "vidlada": VidLaDA, "dynin": Dynin, "dreamvl": DreamVL}


def random_like(mask, rng):
    """As many cells as the label marks in each time step, drawn at random from that step's unmarked cells."""
    out = np.zeros_like(mask)
    for t in range(mask.shape[0]):
        free = np.flatnonzero(~mask[t])
        picked = rng.choice(free, size=min(int(mask[t].sum()), free.size), replace=False)
        out[t].flat[picked] = True
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=sorted(MODELS), help="Model to steer")
    parser.add_argument("labels", type=Path, nargs="+", help="Labels files on this model's token grid")
    parser.add_argument("--biases", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--model-path", dest="model", help="Checkpoint (default: the one its run script uses)")
    parser.add_argument("--repo", type=Path, help="Model repository (VidLaDA, LLaVA-NeXT, Dynin-Omni) if not in ~")
    parser.add_argument("--vq-model", help="Dynin-Omni's image tokenizer (default snu-aidas/magvitv2)")
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/steering"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    labels = {}
    for path in args.labels:
        for label in json.load(open(path))["labels"]:
            if label["done"] and not label["unsure"]:
                labels[(label["category"], label["video"])] = label
    questions = {}
    for category in {c for c, _ in labels}:
        questions.update({(category, q["video"]): q for q in json.load(open(args.data_dir / "json" / f"{category}.json"))})
    conditions = [("none", 0.0)] + [(target, b) for b in args.biases for target in ("object", "random")]

    adapter = MODELS[args.which](args)
    out_dir = args.out_dir / adapter.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in open(results_path):
            row = json.loads(line)
            if "error" not in row:
                done.add((row["category"], row["video"], row["target"], row["bias"]))
    todo = [k for k in sorted(labels) if any((*k, t, b) not in done for t, b in conditions)]
    print(f"{adapter.name}: {len(labels)} videos, {len(todo)} to run, {len(conditions)} conditions each, "
          f"{adapter.layers} layers steered")

    steerer = SdpaSteerer()
    for n, (category, video) in enumerate(todo, start=1):
        item, label = questions[(category, video)], labels[(category, video)]
        letters = adapter.letters[:len(item["candidates"])]
        answer = item["candidates"].index(item["answer"])
        start = time.time()
        try:
            case = adapter.prepare(item, args.data_dir / "video" / category / category / video, label)
            mask = label_mask(label)
            if case.positions.shape[:3] != mask.shape:
                raise RuntimeError(f"Token grid {case.positions.shape[:3]} differs from the label grid {mask.shape}")
            rng = np.random.default_rng(zlib.crc32(video.encode()))
            keys = {"object": case.positions[mask].ravel(), "random": case.positions[random_like(mask, rng)].ravel(),
                    "none": np.array([], dtype=int)}
            measure = {"object": keys["object"], "video": case.positions.ravel()}
            summary = []
            for target, bias in conditions:
                if (category, video, target, bias) in done:
                    continue
                with torch.inference_mode(), steerer.steer(case.seq_len, case.row, keys[target], bias,
                                                           dict(measure, target=keys[target]) if target != "none" else measure):
                    scores = case.forward()
                if steerer.calls != adapter.layers:
                    raise RuntimeError(f"Steered {steerer.calls} attention calls for {adapter.layers} layers")
                probs = torch.softmax(scores[letters].float(), dim=-1).tolist()
                shares = {name: float(np.mean([s[name] for s in steerer.shares])) for name in steerer.shares[0]}
                pred = int(np.argmax(probs))
                row = {"model": adapter.name, "category": category, "video": video, "target": target, "bias": bias,
                       "answer_letter": LETTERS[answer], "pred_letter": LETTERS[pred], "correct": pred == answer,
                       "p_correct": probs[answer], "letter_probs": probs,
                       **{f"{name}_share": value for name, value in shares.items()}}
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                summary.append(f"{target[0]}{bias:g}:{probs[answer]:.2f}")
            status = " ".join(summary)
        except Exception as e:
            with open(results_path, "a") as f:
                f.write(json.dumps({"model": adapter.name, "category": category, "video": video, "target": "error",
                                    "bias": -1, "error": f"{type(e).__name__}: {e}"}) + "\n")
            traceback.print_exc()
            status = f"{type(e).__name__}: {e}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{n}/{len(todo)}] {category}/{video} (answer {LETTERS[answer]}): p(correct) {status} "
              f"{time.time() - start:.1f}s", flush=True)

    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
