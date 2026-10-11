# Video-LLM-Temporal-Attention
Studying temporal attention patterns in Video LLMs and developing attention-based methods to improve video-grounded reasoning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python download_data.py
```

`download_data.py` downloads [DyBench](https://huggingface.co/datasets/ddz16/DyBench) (about 8 GB once unzipped) into `DyBench/`.

## Recording attention

`run_attention.py` runs Qwen3-VL-8B on DyBench and saves, for every video, how much each generated token attends to each part of each frame.

```bash
# One video pair first, to check it fits on the GPU
python run_attention.py --category moving_direction --limit 2

# Everything (resumes where it stopped if interrupted)
python run_attention.py
```

Outputs go to `outputs/qwen3vl_8b/`:

- `results.jsonl`: one line per video with the model's answer, whether it was right, and the file holding its attention.
- `<category>/<video>.npz`: `vision_attention` with shape `[steps, layers, T, H, W]`, the attention from each generated token (`steps`) to every video token, per LLM layer, averaged over heads. `T` is time (each step is 2 merged frames, at `frame_times` seconds) and `H x W` is space (each cell is a 32x32 pixel square of the frame resized to `frame_size`).

If the GPU runs out of memory, lower `--max-vision-tokens` (default 3072) or `--max-frames` (default 32).

```python
import numpy as np
d = np.load("outputs/qwen3vl_8b/moving_direction/human_right_23.npz")
att = d["vision_attention"][0]         # first generated token: [layers, T, H, W]
per_frame = att.mean(0).sum((1, 2))    # attention on each time step, averaged over layers
heatmap = att.mean(0)[per_frame.argmax()]  # spatial map of the most attended time step
```

## OmniTrace

`OmniTrace/` is a copy of [UCSB-AI/OmniTrace](https://github.com/UCSB-AI/OmniTrace) (MIT license, commit `2d18e5a`) with these changes:

- New `qwen3vl` backend (`src/omnitrace/backends/qwen3vl.py`) for video + text input. Upstream supports Qwen2.5-Omni and MiniCPM-o only.
- `trace_video(..., return_raw_attention=True)` also returns the per-layer attention over the video token grid. Upstream only reports which time ranges each part of the answer drew on.
- For Qwen3-VL the attention is taken from the token being generated at every step. Upstream averages over all prompt positions for the first generated token.
- `opencv-python-headless` replaces `opencv-python`, so no display libraries are needed on a server.
