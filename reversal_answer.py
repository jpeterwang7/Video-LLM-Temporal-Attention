"""
Reversal-consistent answering: Qwen3-VL answers each video as given and played backward, and the two must agree.

Qwen3-VL leans toward "forward": on the test set it gets 55% of moving-direction videos played forward but 22% of
their reversed twins, and on reversible dynamics it often says nothing changed, which is never right. Every answer
option has a known mirror when the video plays backward:

    moving direction      forward <-> backward and left <-> right (synthetic: up <-> down, left <-> right;
                          "stationary" stays)
    reversible dynamics   the two changes swap ("closing" <-> "opening"); "No change happened" stays
    event order           "First action 1, then action 2" <-> "First action 2, then action 1" (reversing the
                          frames also plays each action backward, so this category fits least well)

Qwen3-VL scores the options twice, by the log-probability of each option's letter at the answer position (the
prompt is run_attention.py's): once on the video and once on the same frames in reverse order (the frames are
reversed here; no labels and no dataset twins are used). An option's combined score is
    log p(option | video) + log p(mirror(option) | reversed video).
A lean toward one reading of time pushes an option up in one pass and its mirror up in the other, so it cancels;
evidence about what actually happens points the same way in both passes and adds up. A video and its time-reversed
twin get mirrored answers, so DyBench's pair accuracy (both twins right) can only gain from consistency.

Writes <out-dir>/reversal/<set>/results.jsonl (resuming), and with --report prints accuracy for Qwen3-VL's recorded
direct answers, the single pass (letter scores on the video alone), the reversed pass mapped back, and the combined
score, by category, with exact McNemar tests and pair accuracy, into <out-dir>/reversal/<set>/summary.txt.

    python reversal_answer.py --set calib
    python reversal_answer.py --set test --exclude-set smoke_test
    python reversal_answer.py --set test --exclude-set smoke_test --report
    python reversal_answer.py --check-mirrors          # checks the mirror rules against DyBench's own twins
"""
import argparse
import copy
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np

import tabalign_video as tv

SHORT = {"event_sequence": "Event order", "moving_direction": "Moving dir.", "reversible_dynamics": "Reversible"}
SWAPS = {"moving_direction": [("forward", "backward"), ("left", "right"), ("up", "down")],
         "event_sequence": [("action 1", "action 2")]}
# Reversible dynamics always has three options, the two changes first and "no change" last, in many wordings
# ("neither of the actions happened", "Staying still", "The cat does not move", ...); --check-mirrors checks it
NO_CHANGE = re.compile(r"\b(no|nothing|neither|still|unchanged|not move|same posture|remains?|staying)\b", re.I)


def swap_words(text, pairs):
    out = text.lower()
    for i, (a, b) in enumerate(pairs):          # through placeholders, so a -> b and b -> a at once
        out = re.sub(rf"\b{re.escape(a)}\b", f"\0{i}a\0", out)
        out = re.sub(rf"\b{re.escape(b)}\b", f"\0{i}b\0", out)
    for i, (a, b) in enumerate(pairs):
        out = out.replace(f"\0{i}a\0", b).replace(f"\0{i}b\0", a)
    return out


def mirror_map(category, candidates):
    """For each option, the index of the option that is right when the video plays backward; None if unclear."""
    if category == "reversible_dynamics":
        if len(candidates) != 3 or not NO_CHANGE.search(candidates[2]):
            return None
        return [1, 0, 2]
    pairs = SWAPS[category]
    if category == "moving_direction" and any("in reverse" in c.lower() for c in candidates):
        pairs = [("forward", "in reverse")] + pairs[1:]
    lowered = [c.lower().strip() for c in candidates]
    out = []
    for c in candidates:
        flipped = swap_words(c, pairs).strip()
        if flipped not in lowered:
            return None
        out.append(lowered.index(flipped))
    return out if sorted(out) == list(range(len(candidates))) else None


def check_mirrors(data_dir):
    """Every question has a mirror map, and it turns each original's answer into its time-reversed twin's answer."""
    types = json.load(open("dybench_pair_types.json"))
    questions = tv.load_questions(data_dir)
    for category in tv.CATEGORIES:
        rows = {v: q for (c, v), q in questions.items() if c == category}
        unmapped = [v for v, q in rows.items() if mirror_map(category, q["candidates"]) is None]
        agree = total = 0
        for video, kind in types[category].items():
            twin = video.replace(".mp4", "_reversed.mp4")
            if video not in rows or twin not in rows or (category != "event_sequence" and kind != "time_reversed"):
                continue
            q, t = rows[video], rows[twin]
            m = mirror_map(category, q["candidates"])
            if m is None or q["candidates"] != t["candidates"]:
                continue
            total += 1
            agree += q["candidates"][m[q["candidates"].index(q["answer"])]] == t["answer"]
        print(f"{category}: {len(rows)} questions, {len(unmapped)} without a mirror map"
              + (f" (e.g. {rows[unmapped[0]]['candidates']})" if unmapped else "")
              + f"; the mirror of the answer is the twin's answer in {agree} of {total} "
              + ("clip-swapped" if category == "event_sequence" else "time-reversed") + " pairs")


# ---------------------------------------------------------------------------------------------------------------------

def letter_log_probs(reasoner, video, prompt, letter_ids):
    """Log-probabilities of the option letters (renormalised over the options) at the answer position."""
    import torch
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    inputs = reasoner._video_inputs(messages, video)
    with torch.inference_mode():
        logits = reasoner.model(**inputs, logits_to_keep=1).logits[0, -1].float()
    return torch.log_softmax(logits[letter_ids], dim=-1).cpu().numpy()


def run(args):
    import torch
    from execute_tabalign import Reasoner, Video
    from run_attention import build_prompt

    sets = tv.load_sets(args.out_dir / "sets.json")
    questions = tv.load_questions(args.data_dir)
    direct = tv.load_rows(args.direct)
    wanted = tv.select_videos(sets, args.set, args.exclude_set, args.limit)
    path = args.out_dir / "reversal" / args.set / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = tv.load_rows(path)
    todo = [k for k in wanted if k not in done]
    print(f"{len(wanted)} videos, {len(wanted) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    reasoner = Reasoner(args.model, args.max_vision_tokens)
    for n, key in enumerate(todo, start=1):
        item = questions[key]
        letters = tv.LETTERS[:len(item["candidates"])]
        row = {"category": key[0], "video": key[1], "answer_letter": letters[item["candidates"].index(item["answer"])],
               "direct_correct": direct.get(key, {}).get("correct"), "direct_pred": direct.get(key, {}).get("pred_letter")}
        start = time.time()
        try:
            mirror = mirror_map(key[0], item["candidates"])     # None: some option's mirror is not offered
            letter_ids = []
            for letter in letters:
                ids = reasoner.tokenizer.encode(letter, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"letter {letter!r} is {len(ids)} tokens")
                letter_ids.append(ids[0])
            video = Video(tv.video_path(args.data_dir, *key), args.fps, args.max_frames)
            prompt = build_prompt(item)
            forward_lp = letter_log_probs(reasoner, video, prompt, letter_ids)
            single = int(np.argmax(forward_lp))
            row.update(mirror=mirror, forward_lp=[round(float(x), 4) for x in forward_lp],
                       single_pred=letters[single], single_correct=letters[single] == row["answer_letter"])
            if mirror is None:                                    # keep the single pass's answer
                row.update(reversed_pred=None, reversed_correct=None, combined_pred=letters[single],
                           combined_correct=row["single_correct"])
            else:
                backward = copy.copy(video)                       # same sampling times, frames in reverse order
                backward.frames = np.ascontiguousarray(video.frames[::-1])
                backward_lp = letter_log_probs(reasoner, backward, prompt, letter_ids)
                reverse_only, best = int(mirror[int(np.argmax(backward_lp))]), int(np.argmax(forward_lp + backward_lp[mirror]))
                row.update(backward_lp=[round(float(x), 4) for x in backward_lp],
                           reversed_pred=letters[reverse_only], reversed_correct=letters[reverse_only] == row["answer_letter"],
                           combined_pred=letters[best], combined_correct=letters[best] == row["answer_letter"])
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - start, 2)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        status = row.get("error") or (f"single {row['single_pred']} ({'right' if row['single_correct'] else 'wrong'}), "
                                      f"combined {row['combined_pred']} ({'right' if row['combined_correct'] else 'wrong'})"
                                      f", answer {row['answer_letter']}")
        print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: {status} ({row['seconds']}s)", flush=True)


def report(args):
    wanted = tv.select_videos(tv.load_sets(args.out_dir / "sets.json"), args.set, args.exclude_set, args.limit)
    rows = tv.load_rows(args.out_dir / "reversal" / args.set / "results.jsonl")
    keys = [k for k in wanted if k in rows]
    if not keys:
        raise SystemExit("Nothing to report yet")
    methods = [("Qwen3-VL directly (recorded)", "direct_correct"), ("single pass (letter scores)", "single_correct"),
               ("reversed pass, mapped back", "reversed_correct"), ("reversal-consistent (combined)", "combined_correct")]
    lines = [f"Reversal-consistent answering ({args.set}" + (f", without {', '.join(args.exclude_set)}" if args.exclude_set else "")
             + f"): {len(keys)} videos",
             "Accuracy in % (videos); pairs = a video and its twin both right; vs single = right only here / right only "
             "with the single pass, exact McNemar p.", "",
             f"{'method':<32}{'All':>7}" + "".join(f"{SHORT[c]:>14}" for c in tv.CATEGORIES) + f"{'pairs':>9}{'vs single':>20}"]
    present = set(keys)
    pairs = [(k, (k[0], k[1].replace(".mp4", "_reversed.mp4"))) for k in keys
             if not k[1].endswith("_reversed.mp4") and (k[0], k[1].replace(".mp4", "_reversed.mp4")) in present]
    for label, field in methods:
        usable = [k for k in keys if rows[k].get(field) is not None]
        cells = []
        for c in tv.CATEGORIES:
            group = [rows[k][field] for k in usable if k[0] == c]
            cells.append(f"{100 * np.mean(group):6.1f} ({len(group):3d})" if group else f"{'—':>13}")
        both = [rows[a][field] and rows[b][field] for a, b in pairs
                if rows[a].get(field) is not None and rows[b].get(field) is not None]
        if field == "single_correct":
            versus = "—"
        else:
            shared = [k for k in usable if rows[k].get("single_correct") is not None]
            a, b, p = tv.mcnemar([rows[k][field] for k in shared], [rows[k]["single_correct"] for k in shared])
            versus = f"+{a}/-{b} p={p:.2g}"
        lines.append(f"{label:<32}{100 * np.mean([rows[k][field] for k in usable]):>7.1f}" + "".join(f"{c:>14}" for c in cells)
                     + (f"{100 * np.mean(both):>8.1f}%" if both else f"{'—':>9}") + f"{versus:>20}")
    same = [k for k in keys if rows[k].get("direct_pred") is not None]
    match = sum(rows[k]["single_pred"] == rows[k]["direct_pred"] for k in same)
    unmapped = [k for k in keys if rows[k].get("mirror") is None]
    lines += ["", f"Check: the single pass picks the recorded direct answer on {match} of {len(same)} videos "
                  "(the recorded answers were generated text; this scores the letters).",
              f"{len(unmapped)} videos keep the single pass's answer because some option's mirror is not offered "
              f"({', '.join(f'{SHORT[c]} {sum(k[0] == c for k in unmapped)}' for c in tv.CATEGORIES)}).",
              f"{len(pairs)} twin pairs in the set. Per category, combined against the single pass:"]
    for c in tv.CATEGORIES:
        group = [k for k in keys if k[0] == c]
        if group:
            a, b, p = tv.mcnemar([rows[k]["combined_correct"] for k in group], [rows[k]["single_correct"] for k in group])
            lines.append(f"   {SHORT[c]:<14} +{a}/-{b}, p = {p:.2g}")
    text = "\n".join(lines)
    print(text)
    out = args.out_dir / "reversal" / args.set / "summary.txt"
    out.write_text(text + "\n")
    print(f"\nWritten to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", help="A set from sets.json (calib, test, smoke_calib, smoke_test)")
    parser.add_argument("--report", action="store_true", help="Print and save the accuracy table")
    parser.add_argument("--check-mirrors", action="store_true", help="Check the mirror rules against DyBench's twins")
    parser.add_argument("--exclude-set", nargs="*", default=[], help="Leave out the videos of these sets")
    parser.add_argument("--limit", type=int, help="Only the first N videos (one pair per category in turn)")
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--direct", type=Path, default=Path("outputs/qwen3vl_8b/results.jsonl"))
    parser.add_argument("--model", help="Default: the Qwen3-VL run_attention.py used")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    args = parser.parse_args()
    if args.check_mirrors:
        check_mirrors(args.data_dir)
    elif not args.set:
        parser.error("give --set (or --check-mirrors)")
    elif args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
