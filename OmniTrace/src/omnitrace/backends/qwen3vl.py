from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from omnitrace.constants import (
    DEFAULT_VIDEO_FPS,
    QWEN3VL_MODEL_PATH,
    QWEN3VL_VIDEO_MAX_FRAMES,
    QWEN3VL_VIDEO_MAX_PIXELS,
)
from omnitrace.core import (
    ModelBundle,
    GenResult,
    GenerationConfig,
    DEFAULT_GENERATION_CONFIG,
    find_source_boundaries,
    GRADIENT_METHODS,
    validate_method,
)

logger = logging.getLogger(__name__)

# Key used to carry video layout from prepare_inputs to generate_with_attn.
VIDEO_INFO_KEY = "_omnitrace_video_info"

QWEN3VL_MIN_FRAMES = 4
QWEN3VL_VIDEO_MIN_PIXELS = 128 * 32 * 32

LAST_QUERY_ATTENTION = "sdpa_last_query"


def sdpa_last_query_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
):
    """
    sdpa attention that also returns the attention weights of the last query token,
    shape [batch, heads, 1, keys].

    Eager attention would build the full [heads, N, N] map for the prompt, which
    does not fit next to an 8B model on a 24 GB GPU; only the last row is needed.
    """
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    attn_output, _ = sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )

    keys = repeat_kv(key, module.num_key_value_groups)
    scores = torch.matmul(query[:, :, -1:, :], keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        mask = attention_mask[:, :, -1:, : keys.shape[-2]]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    return attn_output, weights


def load_model_qwen3vl(model_path: str | None = None) -> ModelBundle:
    from transformers import AttentionInterface, AutoProcessor, Qwen3VLForConditionalGeneration

    path = model_path or QWEN3VL_MODEL_PATH
    logger.info(f"Loading Qwen3-VL from {path} ({LAST_QUERY_ATTENTION} LLM + sdpa vision)...")

    # The LLM returns last-query attention weights; the vision encoder keeps the fast path.
    AttentionInterface.register(LAST_QUERY_ATTENTION, sdpa_last_query_attention)
    # With a GPU present, load onto it or fail; "auto" silently falls back to the CPU
    # when the GPU is busy, which is far too slow for this model.
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype="auto",
        device_map=device_map,
        attn_implementation={"text_config": LAST_QUERY_ATTENTION, "vision_config": "sdpa"},
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(path)
    tokenizer = processor.tokenizer

    image_token_id = int(model.config.image_token_id)
    vision_token_id = int(model.config.video_token_id)

    special_token_ids = {
        tokenizer.convert_tokens_to_ids(tok)
        for tok in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>")
    }
    special_token_ids.update(tokenizer("\n", add_special_tokens=False)["input_ids"])
    special_token_ids.update(range(15))

    logger.info(f"Image token id: {image_token_id}")
    logger.info(f"Vision token id: {vision_token_id}")
    logger.info("Qwen3-VL model loaded successfully")

    return ModelBundle(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        model_type="qwen3vl",
        special_token_ids=special_token_ids,
        audio_token_id=-1,
        vision_token_id=vision_token_id,
        image_token_id=image_token_id,
    )


def read_video_frames(
    video_path: str,
    fps: float,
    max_frames: int,
) -> Tuple[np.ndarray, List[int], float, int]:
    """
    Uniformly sample an even number of RGB frames from a video.

    Returns (frames [T, H, W, 3] uint8, sampled frame indices, native fps, total frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0 or total < 2:
        cap.release()
        raise RuntimeError(f"Cannot read fps/frame count of video: {video_path}")

    num = int(total / native_fps * fps)
    num = min(max(num, QWEN3VL_MIN_FRAMES), max_frames, total)
    num -= num % 2  # frames are merged in pairs
    if num < 2:
        cap.release()
        raise RuntimeError(f"Video has too few frames: {video_path}")

    wanted = np.linspace(0, total - 1, num).round().astype(int).tolist()
    wanted_set = set(wanted)

    frames: List[np.ndarray] = []
    indices: List[int] = []
    for idx in range(wanted[-1] + 1):
        if not cap.grab():
            break
        if idx in wanted_set:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            indices.append(idx)
    cap.release()

    if len(frames) < len(wanted):
        logger.warning(
            f"Video ended early: got {len(frames)} of {len(wanted)} sampled frames ({video_path})"
        )
    if len(frames) % 2 == 1:
        frames, indices = frames[:-1], indices[:-1]
    if len(frames) < 2:
        raise RuntimeError(f"Could not decode enough frames from video: {video_path}")

    return np.stack(frames, axis=0), indices, native_fps, total


def prepare_inputs_qwen3vl(
    bundle: ModelBundle,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Qwen3-VL input preparation from chat messages holding one video plus text.

    Video item options: "fps", "max_frames", "total_pixels".
    """
    processor = bundle.processor

    video_items: List[Dict[str, Any]] = []
    template_messages: List[Dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            template_messages.append(message)
            continue
        new_content = []
        for item in content:
            item_type = item.get("type")
            if item_type == "video":
                video_items.append(item)
                new_content.append({"type": "video"})
            elif item_type == "text":
                new_content.append({"type": "text", "text": item["text"]})
            else:
                raise ValueError(
                    f"Qwen3-VL backend supports video + text inputs only, got: {item_type}"
                )
        template_messages.append({"role": message["role"], "content": new_content})

    if len(video_items) != 1:
        raise ValueError(
            f"Qwen3-VL backend expects exactly one video per sample, got {len(video_items)}"
        )
    video_item = video_items[0]
    video_path = video_item["video"]

    fps = float(video_item.get("fps") or DEFAULT_VIDEO_FPS)
    max_frames = int(video_item.get("max_frames") or QWEN3VL_VIDEO_MAX_FRAMES)
    max_pixels = int(video_item.get("total_pixels") or QWEN3VL_VIDEO_MAX_PIXELS)
    min_pixels = min(QWEN3VL_VIDEO_MIN_PIXELS, max_pixels)

    frames, indices, native_fps, total = read_video_frames(video_path, fps, max_frames)

    text = processor.apply_chat_template(
        template_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = processor(
        text=[text],
        videos=[frames],
        video_metadata=[
            {
                "total_num_frames": total,
                "fps": native_fps,
                "frames_indices": list(indices),
                "duration": total / native_fps,
            }
        ],
        do_sample_frames=False,
        size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
        return_tensors="pt",
    )

    grid_t, grid_h, grid_w = (int(x) for x in batch["video_grid_thw"][0].tolist())
    patch_size = int(processor.video_processor.patch_size)
    merge_size = int(processor.video_processor.merge_size)

    # Each temporal group merges 2 frames; its time is the mean of the pair.
    times = [idx / native_fps for idx in indices]
    frame_times = [(times[i] + times[i + 1]) / 2 for i in range(0, len(times), 2)]

    inputs: Dict[str, Any] = dict(batch.to(bundle.model.device).to(bundle.model.dtype))
    inputs[VIDEO_INFO_KEY] = {
        "grid_thw": (grid_t, grid_h // merge_size, grid_w // merge_size),
        "frame_times": frame_times,
        "frame_indices": list(indices),
        "frame_size": (grid_h * patch_size, grid_w * patch_size),
        "native_fps": native_fps,
        "total_frames": total,
        "duration": total / native_fps,
    }
    return inputs


class _LastQueryAttentionRecorder:
    """
    Forward hooks on every LLM self-attention module. For each forward pass they keep
    the head-averaged attention row of the last query token, i.e. the token whose
    output predicts the next generated token. Full attention maps are never stored.
    """

    def __init__(self, layers) -> None:
        self.rows: List[List[torch.Tensor]] = [[] for _ in layers]
        self.handles = [
            layer.self_attn.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(layers)
        ]

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            weights = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    "No attention weights from the LLM; load the model with load_model_qwen3vl."
                )
            self.rows[layer_idx].append(weights[0, :, -1, :].float().mean(dim=0).cpu())

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def generate_with_attn_qwen3vl(
    bundle: ModelBundle,
    inputs: Dict[str, Any],
    gen_cfg: GenerationConfig = DEFAULT_GENERATION_CONFIG,
) -> GenResult:
    """
    Generate output text and attribution signals for Qwen3-VL.
    """
    method = validate_method(gen_cfg.method)
    if method in GRADIENT_METHODS:
        raise NotImplementedError(
            "Gradient-based attribution is not implemented for Qwen3-VL; use attmean or attraw."
        )

    inputs = dict(inputs)
    video_info = inputs.pop(VIDEO_INFO_KEY, None)

    input_length = int(inputs["input_ids"].shape[1])
    input_ids_1d = inputs["input_ids"][0].detach().cpu()

    (
        source_start,
        source_end,
        image_positions,
        vision_positions,
        audio_positions,
        text_positions,
        text_region_start,
    ) = find_source_boundaries(
        input_ids=input_ids_1d,
        audio_token_id=bundle.audio_token_id,
        tokenizer=bundle.tokenizer,
        special_token_ids=bundle.special_token_ids,
        vision_token_id=bundle.vision_token_id,
        image_token_id=bundle.image_token_id,
    )

    logger.info(f"Input length: {input_length}")
    logger.info(f"Source region: [{source_start}, {source_end})")
    logger.info(f"Vision tokens: {len(vision_positions)}")
    logger.info(f"Text region start: {text_region_start}")

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=gen_cfg.max_new_tokens,
        no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
        return_dict_in_generate=True,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    if gen_cfg.min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = gen_cfg.min_new_tokens

    recorder = _LastQueryAttentionRecorder(bundle.model.model.language_model.layers)
    logger.info(f"Generating with method={method} ...")
    try:
        with torch.inference_mode():
            outputs = bundle.model.generate(**inputs, **gen_kwargs)
    finally:
        recorder.remove()

    generated_ids = outputs.sequences[0].detach().cpu()
    text = bundle.tokenizer.decode(
        generated_ids[input_length:],
        skip_special_tokens=True,
    )

    # One forward pass per generated token: step i attends over the prompt to predict token i.
    num_generated = len(generated_ids) - input_length
    num_steps = min(len(rows) for rows in recorder.rows)
    if num_steps != num_generated:
        logger.warning(
            f"Recorded {num_steps} attention steps for {num_generated} generated tokens"
        )
        num_steps = min(num_steps, num_generated)

    num_layers = len(recorder.rows)
    layer_attentions = np.zeros((num_steps, num_layers, input_length), dtype=np.float32)
    for layer_idx, rows in enumerate(recorder.rows):
        for step in range(num_steps):
            layer_attentions[step, layer_idx] = rows[step][:input_length].numpy()

    if method == "attmean":
        step_scores = layer_attentions.mean(axis=1)
    else:  # attraw: last layer only
        step_scores = layer_attentions[:, -1]
    all_attentions: List[np.ndarray] = [step_scores[i] for i in range(num_steps)]

    vision_grid_thw = None
    if video_info is not None:
        grid_t, grid_h, grid_w = video_info["grid_thw"]
        if grid_t * grid_h * grid_w == len(vision_positions):
            vision_grid_thw = (grid_t, grid_h, grid_w)
        else:
            logger.warning(
                f"Vision token count {len(vision_positions)} does not match grid "
                f"{grid_t}x{grid_h}x{grid_w}"
            )

    logger.info(f"Generated {num_generated} tokens")
    logger.info(f"Collected {len(all_attentions)} attention maps")
    logger.info(f"Response: {text}")

    return GenResult(
        text=text,
        generated_ids=generated_ids,
        input_ids=input_ids_1d,
        input_length=input_length,
        all_attentions=all_attentions,
        source_start=source_start,
        source_end=source_end,
        text_positions=text_positions,
        text_region_start=text_region_start,
        grad_scores_by_step=None,
        image_positions=image_positions,
        audio_positions=audio_positions,
        vision_positions=vision_positions,
        layer_attentions=layer_attentions,
        vision_grid_thw=vision_grid_thw,
        video_info=video_info,
    )
