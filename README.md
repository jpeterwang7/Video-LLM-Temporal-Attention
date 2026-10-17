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

On a GPU server, check the "CUDA Version" that `nvidia-smi` reports. If it is 12.x, install a matching PyTorch build before `pip install -r requirements.txt`, because the default build needs a newer driver:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
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

## Results

Qwen3-VL-8B-Instruct on DyBench, run on 2026-10-11 with the default settings (2 frames per second, at most 32 frames and 3072 video tokens per video). Pair accuracy counts a pair only when the original and its twin are both answered correctly.

| Category | Accuracy | Paper | Pair accuracy | Paper |
|---|---|---|---|---|
| Event sequence | 635 / 814 = 78.0% | 79.5% | 230 / 407 = 56.5% | 59.7% |
| Moving direction | 439 / 1000 = 43.9% | 40.8% | 105 / 500 = 21.0% | 18.2% |
| Reversible dynamics | 945 / 1200 = 78.8% | 83.3% | 378 / 600 = 63.0% | 70.8% |
| Overall | 2019 / 3014 = 67.0% | 68.2% | 713 / 1507 = 47.3% | 50.4% |

The paper columns are the Qwen3-VL-8B (base) rows of Tables 1 and 5 in [the DyBench paper](https://arxiv.org/abs/2605.21988). The paper evaluates with 32 uniformly sampled frames and the benchmark's own prompt, so the settings are not identical.

`analyze_attention.py` writes these counts to `outputs/qwen3vl_8b/analysis/accuracy.csv`.

## Analysing attention

```bash
python analyze_attention.py
```

This reads `outputs/qwen3vl_8b/` and writes tables and charts to `outputs/qwen3vl_8b/analysis/`, all based on the attention of the first generated token (the answer letter):

- `temporal_profile.png`: where in the video the attention falls, for right and wrong answers.
- `vision_share_by_layer.png`: how much attention the video gets in each layer, as opposed to the text.
- `pair_similarity.png`: for time-reversed pairs, whether the twin's attention sits at the same time steps as the original's (position) or on the same frames (content).
- `accuracy.csv`, `per_video.csv`, `per_pair.csv`, `summary.txt`: the numbers behind the charts.

`dybench_pair_types.json` records how each DyBench `_reversed` twin relates to its original, measured from the pixels. They are not all time reversals:

| Category | Time-reversed | Mirrored left-right | Other |
|---|---|---|---|
| Reversible dynamics | 600 | 0 | 0 |
| Moving direction | 395 | 105 | 0 |
| Event sequence | 6 | 0 | 401 |

The event sequence twins swap the order of the two clips instead of reversing the frames. The content-versus-position comparison only uses the time-reversed pairs.

### Checking for attention sinks

```bash
python check_attention_sinks.py
```

This tests whether the start-of-video attention in `temporal_profile.png` comes from a few tokens that absorb attention whatever the video shows. It writes to `outputs/qwen3vl_8b/analysis/sinks/`:

- `summary.txt`: how much of the video attention the top token and the top 1% of tokens hold, where the top token sits, and whether it stays put or moves when a video is reversed.
- `temporal_profile_sinks.png`: graph 1 with and without each video's top 1% of tokens.
- `top_share_by_layer.png`: the share held by the top 1% of tokens in each layer.
- `per_video.csv`: the same measures for every video.

`python show_sink_tokens.py` draws each video's most-attended token on the frame it belongs to, in `outputs/qwen3vl_8b/analysis/sinks/top_token_examples.png`. It needs only the DyBench videos and the two `per_video.csv` files, so it runs on a laptop.

To redo the main analysis without each video's top 1% of tokens, run `python analyze_attention.py --drop-top 0.01`. Results go to `outputs/qwen3vl_8b/analysis_drop_top/`, leaving `analysis/` unchanged.

## Labelling what matters

```bash
python make_labeling_page.py
```

This builds `outputs/labeling/index.html`, a page for marking which time steps a person needs to see to answer each question, and where in those frames. It shows each video's frames exactly as the model saw them. Open it by double-clicking; no server is needed. By default it picks 8 time-reversed pairs (16 videos), 2 for each combination of moving direction or reversible dynamics with both right or one right.

Labels save in the browser as you go. "Download labels" writes `labels_<name>.json`; keep these in `labels/`. Each label records the needed time steps, the boxes as fractions of the frame, and the token-grid cells the boxes cover. The `outputs/labeling` folder includes the videos, so it can be zipped and sent to a second labeller.

To score the model's attention against the labels (on the machine with the attention files):

```bash
python score_attention.py labels/labels_peter.json
```

It writes `summary.txt`, `per_video.csv`, `auroc_by_layer.png` (real attention against the control) and `lift_by_layer.png` (with and without sink tokens) to `outputs/qwen3vl_8b/analysis/labels/`. AUROC measures how well attention ranks the marked tokens above the rest (0.5 is chance, 1 is perfect). Lift is the share of attention on marked tokens divided by the share of tokens marked (1 is an even spread). Both are reported with all tokens and with each video's sink tokens (top 1%) left out. A control scores each video's labels against the attention of up to 20 other videos with the same token grid: attention that only favours a part of the frame, such as the centre, scores as well there, so only alignment with the video's own content beats the control. Several labels files can be passed at once.

Two scripts draw the results for the labelled videos:

- `python plot_label_scores.py` writes `per_video_scores.png`: one row per labelled video, with its AUROC next to the control, and its lift. It only needs `per_video.csv`.
- `python show_label_attention.py labels/labels_peter.json` writes `attention_vs_labels.png`: the model's attention as a heat map on one frame of each labelled video, with the human boxes on top. Add `--layer 22` to show a single layer. It needs the attention files and the videos.

## OmniTrace

`OmniTrace/` is a copy of [UCSB-AI/OmniTrace](https://github.com/UCSB-AI/OmniTrace) (MIT license, commit `2d18e5a`) with these changes:

- New `qwen3vl` backend (`src/omnitrace/backends/qwen3vl.py`) for video + text input. Upstream supports Qwen2.5-Omni and MiniCPM-o only.
- `trace_video(..., return_raw_attention=True)` also returns the per-layer attention over the video token grid. Upstream only reports which time ranges each part of the answer drew on.
- For Qwen3-VL the attention is taken from the token being generated at every step. Upstream averages over all prompt positions for the first generated token.
- `opencv-python-headless` replaces `opencv-python`, so no display libraries are needed on a server.
