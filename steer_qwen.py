"""
Steer Qwen3-VL's attention onto Dynin-Omni's regions while it answers, and see whether it answers better.

The TabAlign run only measured how much of Qwen3-VL's attention fell where Dynin-Omni looks and told it a score;
its attention itself never changed. This changes it. Qwen3-VL answers each question directly, exactly as
run_attention.py did for its 63.1% (same frames, prompt and greedy decoding), while in every language-model layer the
attention from the text after the video (the question, the options and the answer being written) to a chosen region
of video squares is multiplied by a strength alpha (log alpha added to those attention logits, then softmax as usual).
Nothing is trained. Regions, each a set of Qwen3-VL's video squares:

    none      no steering: Qwen3-VL answering directly, through this script (should match the recorded answers)
    dynin     Dynin-Omni's attention when it reads the same question (ground_dynin.py's "question" query, mean of its
              last 4 layers, one frame per time step): the top 25% of the squares it saw in every time step, its
              attention sinks left out
    random    the same number of squares in every time step, placed at random among the squares Dynin-Omni saw:
              whether any focus helps, or Dynin-Omni's in particular
    human     the human boxes (labelled videos only): the most that steering to the right place can do

Configurations are region:alpha (none takes no alpha). The strength is chosen on the validation videos (--best) and
then used on the test videos. For every answer, the share of the last prompt position's attention over the video
that lands on Dynin-Omni's region and on the steered region (mean of the last 4 layers) is recorded, before and after
steering, so the report shows that the attention really moved.

Writes <out-dir>/steer/<set>/<region>_x<alpha>/results.jsonl (or none/), resuming where it stopped, and with
--report prints accuracy against Qwen3-VL answering directly and against the unsteered run, by category, with exact
McNemar tests, into <out-dir>/steer/<set>/summary.txt.

    python steer_qwen.py --set calib --configs none dynin:3 dynin:10 dynin:30
    python steer_qwen.py --set calib --best dynin                      # prints the best alpha
    python steer_qwen.py --set test --exclude-set smoke_test --configs none dynin:10 random:10 human:10
    python steer_qwen.py --set test --exclude-set smoke_test --report
"""
import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

import tabalign_video as tv

REGIONS = ["none", "dynin", "random", "human"]
SHORT = {"event_sequence": "Event order", "moving_direction": "Moving dir.", "reversible_dynamics": "Reversible"}


# ---------------------------------------------------------------------------------------------------------------------
# Steering: an additive bias on the attention logits from the text after the video to the chosen video squares

class SteerState:
    """What the steered attention adds while a generation runs; None when off."""

    def __init__(self):
        self.active = None      # {"keys": LongTensor of key positions, "start": first query position, "bias": log alpha,
        self.cache = {}         #  "layers": set of 0-based layer indices or None for all}

    def set(self, keys=None, start=None, strength=1.0, layers=None):
        self.cache = {}
        self.active = None if keys is None or strength == 1.0 else \
            {"keys": keys, "start": start, "bias": math.log(strength), "layers": layers}


STEER = SteerState()


def steered_mask(mask, query, key, state):
    """The additive float mask sdpa gets: the causal mask as it was (None means causal), plus log alpha on the chosen
    video squares for every query at or after state["start"]. The same for every layer of one forward pass, so it
    is built once per (queries, keys) and reused."""
    import torch
    q_len, k_len = query.shape[-2], key.shape[-2]
    cache_key = (q_len, k_len, mask is None)
    if cache_key in STEER.cache:
        return STEER.cache[cache_key]
    dtype, device = query.dtype, query.device
    low = torch.finfo(dtype).min
    q_pos = torch.arange(k_len - q_len, k_len, device=device)     # queries are the last q_len positions (KV cache)
    if mask is None:
        k_pos = torch.arange(k_len, device=device)
        base = torch.zeros(q_len, k_len, dtype=dtype, device=device)
        base.masked_fill_(k_pos[None, :] > q_pos[:, None], low)
        base = base[None, None]
    elif mask.dtype == torch.bool:
        mask = mask[..., :k_len]
        base = torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(~mask, low)
    else:
        base = mask[..., :k_len].to(dtype).clone()
    rows = (q_pos >= state["start"]).nonzero().squeeze(-1)
    keys = state["keys"][state["keys"] < k_len]
    if len(rows) and len(keys):
        base[:, :, rows[:, None], keys[None, :]] += state["bias"]
    STEER.cache[cache_key] = base
    return base


def install_steering():
    """Wrap the attention the Qwen3-VL loader registered (sdpa plus the last query's weights) so that, while STEER is
    set, it adds the bias. The weights it returns, which the recorder reads, are the steered ones."""
    from transformers import AttentionInterface
    from omnitrace.backends.qwen3vl import LAST_QUERY_ATTENTION, sdpa_last_query_attention

    def steered_attention(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
        state = STEER.active
        if state is not None and (state["layers"] is None or getattr(module, "layer_idx", None) in state["layers"]):
            attention_mask = steered_mask(attention_mask, query, key, state)
        return sdpa_last_query_attention(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                         **kwargs)

    AttentionInterface.register(LAST_QUERY_ATTENTION, steered_attention)


# ---------------------------------------------------------------------------------------------------------------------
# Regions on Qwen3-VL's grid [T, H, W]

def dynin_region(path, grid):
    """Dynin-Omni's top 25% per time step for the question, and the squares it saw (eligible for the random region)."""
    if not path.exists():
        return None, None
    data = np.load(path)
    queries = [str(q) for q in data["queries"]]
    if "question" not in queries or data["last4"].shape[1] != grid[0]:
        return None, None
    i, crop = queries.index("question"), tuple(data["crop"].astype(np.float64))
    ranking = tv.place_on_grid(data["last4"][i].astype(np.float32), grid, crop)
    sinks = tv.sink_squares(tv.place_on_grid(data["all"][i].astype(np.float32), grid, crop))
    mask = tv.compile_mask(ranking, list(range(grid[0])), tv.TARGET_FRACTION, sinks)
    return mask, np.isfinite(ranking) & ~sinks


def random_region(mask, eligible, rng):
    """The same number of squares as `mask` in every time step, at random among the eligible squares."""
    out = np.zeros_like(mask)
    for t in range(mask.shape[0]):
        k, pool = int(mask[t].sum()), np.flatnonzero(eligible[t])
        if k and len(pool) >= k:
            out[t].flat[rng.choice(pool, size=k, replace=False)] = True
    return out if out.any() else None


def share(attention, mask):
    """Share of the attention over the video squares that falls in the region."""
    if attention is None or mask is None:
        return None
    total = float(attention.sum())
    return float(attention[mask].sum() / total) if total > 0 else None


# ---------------------------------------------------------------------------------------------------------------------

def parse_config(text):
    region, _, alpha = text.partition(":")
    if region not in REGIONS:
        raise argparse.ArgumentTypeError(f"unknown region {region!r} (one of {', '.join(REGIONS)})")
    strength = 1.0 if region == "none" else float(alpha or 10)
    name = "none" if region == "none" else f"{region}_x{strength:g}"
    return {"region": region, "strength": strength, "name": name}


def run(args, configs):
    import torch
    from execute_tabalign import Reasoner, Video
    from run_attention import build_prompt, parse_letter

    sets = tv.load_sets(args.out_dir / "sets.json")
    questions = tv.load_questions(args.data_dir)
    labels = {}
    for path in args.labels:
        for label in json.load(open(path))["labels"]:
            if label["done"] and not label["unsure"] and label.get("approved", True):
                labels[(label["category"], label["video"])] = label
    direct = tv.load_rows(args.direct)
    wanted = tv.select_videos(sets, args.set, args.exclude_set, args.limit)
    root = args.out_dir / "steer" / args.set
    done = {c["name"]: tv.load_rows(root / c["name"] / "results.jsonl") for c in configs}
    todo = [k for k in wanted if any(k not in done[c["name"]] for c in configs)]
    print(f"{len(wanted)} videos; {len(todo)} still to run for {', '.join(c['name'] for c in configs)}", flush=True)
    if not todo:
        return

    reasoner = Reasoner(args.model, args.max_vision_tokens)
    install_steering()
    layers = None if args.layers == "all" else set(range(int(args.layers.split("-")[0]) - 1, int(args.layers.split("-")[1])))
    pad = reasoner.tokenizer.pad_token_id or reasoner.tokenizer.eos_token_id

    for n, key in enumerate(todo, start=1):
        item = questions[key]
        start = time.time()
        try:
            video = Video(tv.video_path(args.data_dir, *key), args.fps, args.max_frames)
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": build_prompt(item)}]}]
            inputs = reasoner._video_inputs(messages, video)
            positions = (inputs["input_ids"][0] == reasoner.bundle.vision_token_id).nonzero().squeeze(-1)
            grid = video.grid
            if len(positions) != int(np.prod(grid)):
                raise RuntimeError(f"{len(positions)} video tokens for a grid of {grid}")
            dynin_path = args.out_dir / "dynin" / args.dynin_mode / key[0] / (Path(key[1]).stem + ".npz")
            dynin, eligible = dynin_region(dynin_path, grid)
            label = labels.get(key)
            regions = {"none": None, "dynin": dynin,
                       "random": random_region(dynin, eligible, np.random.default_rng(tv.video_seed(key[1], args.seed)))
                       if dynin is not None else None,
                       "human": tv.label_cells_mask(label, grid) if label else None}
        except Exception as e:
            print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: could not prepare ({type(e).__name__}: {e})", flush=True)
            traceback.print_exc()
            continue

        notes = []
        for config in configs:
            if key in done[config["name"]]:
                continue
            row = {"category": key[0], "video": key[1], "config": config["name"], "region": config["region"],
                   "strength": config["strength"], "answer_letter": tv.LETTERS[item["candidates"].index(item["answer"])],
                   "direct_correct": direct.get(key, {}).get("correct"),
                   "direct_pred": direct.get(key, {}).get("pred_letter")}
            region = regions[config["region"]]
            if config["region"] != "none" and (region is None or not region.any()):
                row["skipped"] = "no " + config["region"] + " region for this video"
            else:
                t0 = time.time()
                try:
                    keys = None if region is None else positions[torch.from_numpy(region.ravel()).to(positions.device)]
                    STEER.set(keys, int(positions.max()) + 1, config["strength"], layers)
                    recorder = reasoner.recorder_class(reasoner.layers)
                    try:
                        with torch.inference_mode():
                            out = reasoner.model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                          no_repeat_ngram_size=3, do_sample=False, temperature=None,
                                                          top_p=None, top_k=None, pad_token_id=pad)
                    finally:
                        recorder.remove()
                        STEER.set()
                    response = reasoner.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    # the position that writes the answer: the prompt's last position, first forward pass
                    rows = torch.stack([layer_rows[0] for layer_rows in recorder.rows])[-tv.ATTN_LAYERS:].mean(dim=0)
                    attention = rows[positions.cpu()].reshape(grid).numpy()
                    pred = parse_letter(response, item["candidates"])
                    row.update(response=response.strip(), pred_letter=pred, correct=pred == row["answer_letter"],
                               region_squares=None if region is None else int(region.sum()), video_squares=int(np.prod(grid)),
                               share_dynin=share(attention, dynin), share_region=share(attention, region),
                               seconds=round(time.time() - t0, 2))
                    notes.append(f"{config['name']} {pred or '?'} {'right' if row['correct'] else 'wrong'}"
                                 + (f" ({100 * row['share_dynin']:.0f}% on Dynin-Omni's region)" if row["share_dynin"] is not None else ""))
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                    notes.append(f"{config['name']} error")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            path = root / config["name"] / "results.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if "error" not in row:
                done[config["name"]][key] = row
        direct_text = {True: "right", False: "wrong"}.get(direct.get(key, {}).get("correct"), "?")
        print(f"[{n}/{len(todo)}] {key[0]}/{key[1]} (directly {direct_text}): {'; '.join(notes) or 'nothing to run'} "
              f"({time.time() - start:.1f}s)", flush=True)


def load_runs(args):
    root = args.out_dir / "steer" / args.set
    wanted = set(tv.select_videos(tv.load_sets(args.out_dir / "sets.json"), args.set, args.exclude_set, args.limit))
    runs = {}
    for path in sorted(root.glob("*/results.jsonl")):
        rows = {k: r for k, r in tv.load_rows(path).items() if k in wanted and "skipped" not in r}
        if rows:
            runs[path.parent.name] = rows
    return runs


def best_strength(args, region):
    """The alpha with the highest accuracy for a region on this set (the smaller one on a tie)."""
    runs = {name: rows for name, rows in load_runs(args).items() if name.startswith(region + "_x")}
    if not runs:
        raise SystemExit(f"No {region} runs on {args.set} under {args.out_dir / 'steer' / args.set}")
    scored = sorted((-np.mean([r["correct"] for r in rows.values()]), float(name.split("_x")[1])) for name, rows in runs.items())
    return scored[0][1]


def report(args):
    runs = load_runs(args)
    if not runs:
        raise SystemExit("Nothing to report yet")
    none = runs.get("none", {})
    order = ["none"] + sorted((n for n in runs if n != "none"), key=lambda n: (REGIONS.index(n.split("_x")[0]),
                                                                            float(n.split("_x")[1])))
    lines = [f"Steering Qwen3-VL's attention while it answers directly ({args.set}"
             + (f", without {', '.join(args.exclude_set)}" if args.exclude_set else "") + ")",
             "Accuracy in %; vs directly / vs unsteered = right only here / right only there, exact McNemar p, on the "
             "same videos; attention = share of the answer position's attention over the video on Dynin-Omni's "
             "region / on the steered region (mean of the last 4 layers).", ""]
    head = f"{'configuration':<14}{'videos':>7}{'All':>7}" + "".join(f"{SHORT[c]:>13}" for c in tv.CATEGORIES) \
        + f"{'vs directly':>22}{'vs unsteered':>22}{'on Dynin':>10}{'on region':>11}"
    lines.append(head)
    for name in order:
        rows = runs[name]
        keys = sorted(rows)
        acc = 100 * np.mean([rows[k]["correct"] for k in keys])
        cells = []
        for c in tv.CATEGORIES:
            group = [rows[k]["correct"] for k in keys if k[0] == c]
            cells.append(f"{100 * np.mean(group):6.1f} ({len(group):3d})" if group else f"{'—':>12}")
        compared = [k for k in keys if rows[k].get("direct_correct") is not None]
        a, b, p = tv.mcnemar([rows[k]["correct"] for k in compared], [rows[k]["direct_correct"] for k in compared])
        vs_direct = f"+{a}/-{b} p={p:.2g}"
        if name != "none" and none:
            shared = [k for k in keys if k in none]
            a, b, p = tv.mcnemar([rows[k]["correct"] for k in shared], [none[k]["correct"] for k in shared])
            vs_none = f"+{a}/-{b} p={p:.2g}"
        else:
            vs_none = "—"
        on_dynin = [rows[k]["share_dynin"] for k in keys if rows[k].get("share_dynin") is not None]
        on_region = [rows[k]["share_region"] for k in keys if rows[k].get("share_region") is not None]
        lines.append(f"{name:<14}{len(keys):>7}{acc:>7.1f}" + "".join(f"{c:>13}" for c in cells)
                     + f"{vs_direct:>22}{vs_none:>22}"
                     + (f"{100 * np.mean(on_dynin):>9.1f}%" if on_dynin else f"{'—':>10}")
                     + (f"{100 * np.mean(on_region):>10.1f}%" if on_region else f"{'—':>11}"))
    if none:
        agree = [k for k in none if none[k].get("direct_pred") is not None]
        same = sum(none[k]["pred_letter"] == none[k]["direct_pred"] for k in agree)
        lines += ["", f"Check: unsteered answers match the recorded direct answers on {same} of {len(agree)} videos."]
    dynin_runs = [n for n in order if n.startswith("dynin_x")]
    for name in dynin_runs:
        twin = "random_x" + name.split("_x")[1]
        if twin in runs:
            shared = sorted(set(runs[name]) & set(runs[twin]))
            a, b, p = tv.mcnemar([runs[name][k]["correct"] for k in shared], [runs[twin][k]["correct"] for k in shared])
            lines.append(f"{name} vs {twin} on the same {len(shared)} videos: +{a}/-{b}, p = {p:.2g} "
                         "(does Dynin-Omni's region matter, or any focus?)")
    text = "\n".join(lines)
    print(text)
    out = args.out_dir / "steer" / args.set / "summary.txt"
    out.write_text(text + "\n")
    print(f"\nWritten to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, help="A set from sets.json (calib, test, smoke_calib, smoke_test)")
    parser.add_argument("--configs", nargs="+", type=parse_config, help="region:alpha, e.g. none dynin:10 random:10")
    parser.add_argument("--best", choices=REGIONS[1:], help="Print the alpha with the best accuracy for this region")
    parser.add_argument("--report", action="store_true", help="Print and save the accuracy table")
    parser.add_argument("--exclude-set", nargs="*", default=[], help="Leave out the videos of these sets")
    parser.add_argument("--limit", type=int, help="Only the first N videos (one pair per category in turn)")
    parser.add_argument("--layers", default="all", help="Layers to steer: all, or a 1-based range such as 19-36")
    parser.add_argument("--dynin-mode", choices=["crop", "squeeze"], default="crop",
                        help="Which Dynin-Omni maps (crop found the labelled objects best: summary.txt, section F)")
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--labels", type=Path, nargs="*", default=sorted(Path("labels").glob("labels_peter*.json")))
    parser.add_argument("--direct", type=Path, default=Path("outputs/qwen3vl_8b/results.jsonl"))
    parser.add_argument("--model", help="Default: the Qwen3-VL run_attention.py used")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.best:
        print(f"{best_strength(args, args.best):g}")
    elif args.report:
        report(args)
    elif args.configs:
        run(args, args.configs)
    else:
        parser.error("give --configs, --best or --report")


if __name__ == "__main__":
    main()
