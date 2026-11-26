"""
Qwen3-VL answering with Dynin-Omni's attention as evidence, the same way for any multiple-choice video question.

Steering Qwen3-VL's attention toward Dynin-Omni's regions (steer_qwen.py) changed 22 of 620 answers. Dynin-Omni's
attention says more than where to look: when things happen and where they go. This turns it into evidence for or
against each answer option. Nothing is taken from the dataset's categories or its wording of the options:

 1. Claims (--extract). Qwen3-VL reads only the question and its options (no video, no answer) and writes, as JSON,
    what each option claims that the video could confirm: an order in time (two or more moments: actions one after
    another, something before or after a thing the question names, the states before and after a change) and a
    direction across the picture for the question's main object (left, right, up, down, or still). An option that
    claims neither gets no timing or motion evidence.
 2. Grounding (ground_dynin.py --claims). Dynin-Omni attends to the video for the question, the main object ("Find
    the <object> in the video.") and each moment ("Find this in the video: <moment>."), on whole frames, one per
    Qwen3-VL time step.
 3. Evidence for every option, from those maps (attention sinks left out):
        timing  for each claimed "a, then b": minus the slope over time of log(attention on a / attention on b); a
                moment whose share of the attention grows comes later, and whatever both prompts share cancels
        motion  how far the attention on the object drifts along the claimed direction (time steps weighted by how
                focused the attention is, so steps before the object is in view count little)
        still   minus how far that attention drifts, for an option saying the object does not move
    and for every question a zoom: Qwen3-VL sees the video whole and zoomed in on the box around Dynin-Omni's
    attention to the object (else the moments, else the question), each view at the full token budget.
 4. Answer. Qwen3-VL scores each option by the log-probability of its letter (run_attention.py's prompt), and
        score(option) = a * log p(option) + w_timing * timing + w_motion * motion + w_still * still,
    one set of four weights for every question, fitted on the validation set and written to weights.json, so it can
    be used as it is on the test set or on another dataset.

Controls keep everything except Dynin-Omni's attention: Qwen3-VL's own attention for the same prompts as evidence;
Dynin-Omni's evidence with each video's time steps shuffled (one shuffle for all its maps), which removes timing and
motion and keeps the rest; and a zoom box of the same size at a random place. DyBench's categories are used only to
break the results down.

Writes <out-dir>/dynin_evidence/claims.jsonl, <out-dir>/dynin_evidence/<set>/results.jsonl (letter scores on the
plain video and both zooms, the boxes, Qwen3-VL's own attention per time step), all resuming, and with --report
summary.txt and weights.json in <out-dir>/dynin_evidence/<set>/.

    python answer_with_dynin.py --extract --set calib test                       (Qwen3-VL environment)
    python ground_dynin.py --claims outputs/tabalign/dynin_evidence/claims.jsonl --set calib test --modes squeeze
    python answer_with_dynin.py --set calib
    python answer_with_dynin.py --set test --exclude-set smoke_test
    python answer_with_dynin.py --set test --exclude-set smoke_test --report --fit-set calib
"""
import argparse
import hashlib
import json
import math
import re
import time
import traceback
from pathlib import Path

import numpy as np

import tabalign_video as tv

SHORT = {"event_sequence": "Event order", "moving_direction": "Moving dir.", "reversible_dynamics": "Reversible"}
DYNIN_MODE = "squeeze"          # whole frames, so positions in Dynin-Omni's maps are positions in the picture
ZOOM_SIDES = (0.4, 0.8)         # a zoom box's width and height as fractions of the frame: 4 spreads of the
ZOOM_SPREADS = 4.0              # attention across, kept within these bounds
TWO_VIEWS = ("The first video shows the whole scene. The second video is the same clip, zoomed in on one part of "
             "the picture.\n")
COMPONENTS = ["timing", "motion", "still"]
PENALTY = 1.0                   # L2 penalty on the evidence weights (evidence scaled to unit spread)
DIRECTIONS = ("left", "right", "up", "down", "still")
MAX_MOMENTS = 6

EXTRACT_PROMPT = """Here is a multiple-choice question about a video. Do not answer it. Write down, as JSON, what each option claims that the video could show.

- "object": the main thing the question is about (the one that moves, if the options describe movement), as a short noun phrase, or null.
- "moments": short phrases for the events or states whose order in time the options are about: actions that happen one after another, or something that happens before or after a thing the question names. An option naming an action that changes something also claims an order: the state before the change, then the state after it ("plugging in the cable" goes from "the cable is unplugged" to "the cable is plugged in"; "a flower blooming" from "the flower is closed" to "the flower is open"). Each moment should be something one frame could show. [] if no option is about an order in time or a change.
- "options": for each option letter, {"order": [...], "moves": [...]}.
  "order" lists indices into "moments" in the order this option says they happen (at least two), or [].
  "moves" lists the directions across the picture this option says the object moves ("left", "right", "up", "down"), or ["still"] if it says the object does not move, or []. Forward, backward and in reverse are not directions across the picture: leave them out.

Example
Question: What does the woman do after she opens the door?
Options:
A. She turns on the light.
B. She sits down.
JSON: {"object": "the woman", "moments": ["the woman opens the door", "the woman turns on the light", "the woman sits down"], "options": {"A": {"order": [0, 1], "moves": []}, "B": {"order": [0, 2], "moves": []}}}

Example
Question: Two actions are shown. Action 1. pour milk. Action 2. stir the coffee. In what order?
Options:
A. First action 1, then action 2
B. First action 2, then action 1
JSON: {"object": null, "moments": ["pour milk", "stir the coffee"], "options": {"A": {"order": [0, 1], "moves": []}, "B": {"order": [1, 0], "moves": []}}}

Example
Question: Which way does the red ball move?
Options:
A. Down and to the left.
B. Up.
C. It stays where it is.
JSON: {"object": "the red ball", "moments": [], "options": {"A": {"order": [], "moves": ["left", "down"]}, "B": {"order": [], "moves": ["up"]}, "C": {"order": [], "moves": ["still"]}}}

Example
Question: Which of these actions happened in the video?
Options:
A. opening a drawer
B. closing a drawer
C. neither of the actions happened
JSON: {"object": "the drawer", "moments": ["the drawer is closed", "the drawer is open"], "options": {"A": {"order": [0, 1], "moves": []}, "B": {"order": [1, 0], "moves": []}, "C": {"order": [], "moves": []}}}

Now this one.
Question: <QUESTION>
Options:
<OPTIONS>
JSON:"""


# ---------------------------------------------------------------------------------------------------------------------
# Claims (numpy-free helpers, also imported by ground_dynin.py in the Dynin-Omni environment)

def extract_prompt(item):
    options = "\n".join(f"{tv.LETTERS[i]}. {c}" for i, c in enumerate(item["candidates"]))
    return EXTRACT_PROMPT.replace("<QUESTION>", item["question"].strip()).replace("<OPTIONS>", options)


def digest(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def parse_claim(reply, count):
    """The extractor's JSON, checked: order indices in range (at least two, none repeated), known directions only,
    no contradictions (left and right, up and down, still and a direction); anything unusable is dropped, and so
    are moments no option uses. Returns {"object", "moments", "options": {letter: {"order", "moves"}}}."""
    empty = {"object": None, "moments": [], "options": {}}
    start = (reply or "").find("{")
    if start < 0:
        return empty
    try:
        data, _ = json.JSONDecoder().raw_decode(reply[start:])     # the first JSON object; anything after is ignored
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty
    thing = data.get("object")
    thing = thing.strip().rstrip(".") if isinstance(thing, str) and thing.strip().lower() not in ("", "null", "none") \
        else None
    moments = data.get("moments") if isinstance(data.get("moments"), list) else []
    moments = [m.strip().rstrip(".") for m in moments if isinstance(m, str) and m.strip()][:MAX_MOMENTS]
    stated = data.get("options") if isinstance(data.get("options"), dict) else {}
    options = {}
    for letter in tv.LETTERS[:count]:
        said = stated.get(letter)
        if not isinstance(said, dict):
            continue
        order = []
        for entry in said.get("order") if isinstance(said.get("order"), list) else []:
            if isinstance(entry, str):      # an index written as text, or the moment itself
                text = entry.strip().rstrip(".").lower()
                named = [j for j, m in enumerate(moments) if m.lower() == text]
                order.append(int(text) if text.isdigit() else named[0] if named else -1)
            elif isinstance(entry, (int, float)) and not isinstance(entry, bool) and float(entry).is_integer():
                order.append(int(entry))
            else:
                order.append(-1)
        if len(order) < 2 or len(set(order)) != len(order) or not all(0 <= i < len(moments) for i in order):
            order = []                      # a partly unreadable order could be a wrong one
        moves = said.get("moves") if isinstance(said.get("moves"), list) else []
        moves = sorted({m.strip().lower() for m in moves if isinstance(m, str) and m.strip().lower() in DIRECTIONS})
        if thing is None or ("still" in moves and len(moves) > 1) or {"left", "right"} <= set(moves) \
                or {"up", "down"} <= set(moves):
            moves = []
        if order or moves:
            options[letter] = {"order": order, "moves": moves}
    used = sorted({i for said in options.values() for i in said["order"]})
    renumber = {old: new for new, old in enumerate(used)}
    for said in options.values():
        said["order"] = [renumber[i] for i in said["order"]]
    return {"object": thing, "moments": [moments[i] for i in used], "options": options}


def claim_queries(claim):
    """(name, prompt) for everything attended to for one question: the question itself ("question"; Dynin-Omni
    answers it with the multiple-choice prompt), the main object, and each moment."""
    queries = [("question", None)]
    if claim.get("object"):
        thing = claim["object"] if re.match(r"(the|a|an)\b", claim["object"], re.I) else "the " + claim["object"]
        queries.append((f"object: {claim['object']}", f"Find {thing} in the video."))
    queries += [(f"moment: {m}", f"Find this in the video: {m}.") for m in claim.get("moments", [])]
    return queries


def describe(claim):
    """One line for the log."""
    parts = [f"object {claim['object']!r}" if claim.get("object") else "no object"]
    for letter, said in sorted(claim.get("options", {}).items()):
        bits = []
        if said["order"]:
            bits.append(" then ".join(repr(claim["moments"][i]) for i in said["order"]))
        if said["moves"]:
            bits.append("moves " + "+".join(said["moves"]))
        parts.append(f"{letter}: {'; '.join(bits)}")
    return ", ".join(parts) if claim.get("options") else parts[0] + ", no timing or motion claims"


# ---------------------------------------------------------------------------------------------------------------------
# Attention maps -> evidence

def map_profile(last4, every):
    """Per time step of one attention map ([T, H, W], mean of the last 4 layers and of all layers): the attention
    mass, its centre (x, y as fractions of the frame) and focus (1 - entropy / log(squares): 0 spread evenly, 1 on
    one square), attention sinks (top 1% of squares over all layers) left out."""
    clean = np.nan_to_num(np.where(tv.sink_squares(every), 0.0, np.asarray(last4, dtype=np.float64))).clip(0)
    steps, height, width = clean.shape
    mass = clean.sum(axis=(1, 2))
    safe = np.where(mass > 0, mass, 1.0)
    x = (clean.sum(axis=1) * ((np.arange(width) + 0.5) / width)).sum(axis=1) / safe
    y = (clean.sum(axis=2) * ((np.arange(height) + 0.5) / height)).sum(axis=1) / safe
    share = clean.reshape(steps, -1) / safe[:, None]
    entropy = -(share * np.log(np.where(share > 0, share, 1.0))).sum(axis=1)
    focus = np.where(mass > 0, 1.0 - entropy / math.log(height * width), 0.0)
    return {"mass": [float(v) for v in mass], "x": [float(v) for v in x], "y": [float(v) for v in y],
            "focus": [float(v) for v in focus]}


def slope(values, weights=None):
    """Least-squares slope of a series against time running from 0 (first time step) to 1 (last), each time step
    weighted if weights are given."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return 0.0
    t = np.linspace(0.0, 1.0, len(v))
    w = np.ones(len(v)) if weights is None else np.clip(np.asarray(weights, dtype=np.float64), 0, None)
    if w.sum() <= 0:
        w = np.ones(len(v))
    t_mean, v_mean = (w * t).sum() / w.sum(), (w * v).sum() / w.sum()
    denominator = (w * (t - t_mean) ** 2).sum()
    return 0.0 if denominator <= 0 else float((w * (t - t_mean) * (v - v_mean)).sum() / denominator)


def timing(first, second):
    """Evidence that moment `first` comes before `second` (map_profiles): minus the slope of log(attention on the
    first / on the second) over time. None if either is missing."""
    if first is None or second is None:
        return None
    a, b = np.asarray(first["mass"], dtype=np.float64), np.asarray(second["mass"], dtype=np.float64)
    if len(a) != len(b) or len(a) < 2 or a.sum() <= 0 or b.sum() <= 0:
        return None
    a, b = np.maximum(a, 1e-6 * a.mean()), np.maximum(b, 1e-6 * b.mean())     # a time step with no attention left
    return -slope(np.log(a) - np.log(b))


def drift(profile):
    """How far the attention's centre moves over the video, (right, down) as fractions of the frame, time steps
    weighted by focus. None if there is no profile."""
    if profile is None or len(profile["x"]) < 2:
        return None
    return slope(profile["x"], profile.get("focus")), slope(profile["y"], profile.get("focus"))


def option_evidence(claim, profiles, count):
    """[options, COMPONENTS]: how far the attention (profiles: {query name: map_profile}) agrees with what each
    option claims. Zero where an option claims nothing of that kind or the attention is missing."""
    out = np.zeros((count, len(COMPONENTS)))
    if not claim or not profiles:
        return out
    moments = claim.get("moments", [])
    moving = drift(profiles.get(f"object: {claim['object']}")) if claim.get("object") else None
    for i, letter in enumerate(tv.LETTERS[:count]):
        said = claim.get("options", {}).get(letter)
        if not said:
            continue
        steps = [timing(profiles.get(f"moment: {moments[a]}"), profiles.get(f"moment: {moments[b]}"))
                 for a, b in zip(said["order"], said["order"][1:])]
        steps = [s for s in steps if s is not None]
        if steps:
            out[i, 0] = float(np.mean(steps))
        if moving is not None and said["moves"]:
            if said["moves"] == ["still"]:
                out[i, 2] = -math.hypot(*moving)
            else:
                x = ("right" in said["moves"]) - ("left" in said["moves"])
                y = ("down" in said["moves"]) - ("up" in said["moves"])
                if x or y:
                    out[i, 1] = (x * moving[0] + y * moving[1]) / math.hypot(x, y)
    return out


def dynin_maps(path):
    """Dynin-Omni's saved maps for one video: {query name: (last4 [T, H, W], all layers [T, H, W])}; {} if none."""
    if not Path(path).exists():
        return {}
    data = np.load(path)
    return {str(q): (data["last4"][i].astype(np.float32), data["all"][i].astype(np.float32))
            for i, q in enumerate(data["queries"])}


def claim_maps(args, key, claim):
    """Dynin-Omni's maps for what this question's claims name (maps left from earlier claims are ignored)."""
    names = {name for name, _ in claim_queries(claim)}
    maps = dynin_maps(args.out_dir / "dynin_claims" / DYNIN_MODE / key[0] / (Path(key[1]).stem + ".npz"))
    return {q: m for q, m in maps.items() if q in names}


def claims_digest(row):
    """Identifies the claims a result was computed from, so results are redone when the claims change."""
    return digest(json.dumps(row["claim"], sort_keys=True)) if row else None


def zoom_box(maps):
    """The box around Dynin-Omni's attention to the main object (else the moments, else the question), averaged
    over time steps, sinks left out: centred on the attention, 4 spreads across, each side kept between 40% and 80%
    of the frame. [x0, y0, x1, y1] as fractions of the frame, or None."""
    names = [n for n in maps if n.startswith("object: ")] or [n for n in maps if n.startswith("moment: ")] \
        or [n for n in maps if n == "question"]
    total = None
    for name in names:
        last4, every = maps[name]
        clean = np.nan_to_num(np.where(tv.sink_squares(every), 0.0, last4.astype(np.float64))).clip(0).sum(axis=0)
        if clean.sum() > 0:
            total = clean / clean.sum() if total is None else total + clean / clean.sum()
    if total is None:
        return None
    total = total / total.sum()
    height, width = total.shape
    box = []
    for weights, centres in ((total.sum(axis=0), (np.arange(width) + 0.5) / width),
                             (total.sum(axis=1), (np.arange(height) + 0.5) / height)):
        centre = float((weights * centres).sum())
        spread = math.sqrt(float((weights * (centres - centre) ** 2).sum()))
        side = min(ZOOM_SIDES[1], max(ZOOM_SIDES[0], ZOOM_SPREADS * spread))
        low = min(max(centre - side / 2, 0.0), 1.0 - side)
        box.append((low, low + side))
    (x0, x1), (y0, y1) = box
    return [round(v, 4) for v in (x0, y0, x1, y1)]


def random_box(box, rng):
    """A box of the same size at a random place in the frame."""
    width, height = box[2] - box[0], box[3] - box[1]
    x0, y0 = rng.uniform(0, 1 - width), rng.uniform(0, 1 - height)
    return [round(v, 4) for v in (x0, y0, x0 + width, y0 + height)]


def zoom_frames(frames, box):
    """The frames cut to the box and enlarged back to the frame's area, so the box gets the whole token budget."""
    from PIL import Image
    count, height, width = frames.shape[:3]
    left, top = int(round(box[0] * width)), int(round(box[1] * height))
    right, bottom = max(int(round(box[2] * width)), left + 2), max(int(round(box[3] * height)), top + 2)
    scale = math.sqrt(height * width / ((right - left) * (bottom - top)))
    size = (max(32, round((right - left) * scale)), max(32, round((bottom - top) * scale)))
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f[top:bottom, left:right])).resize(size, Image.BICUBIC))
                     for f in frames])


# ---------------------------------------------------------------------------------------------------------------------
# Running Qwen3-VL

def wanted_videos(args, names):
    sets = tv.load_sets(args.out_dir / "sets.json")
    wanted = []
    for name in names:
        wanted += [k for k in tv.select_videos(sets, name, args.exclude_set, args.limit) if k not in wanted]
    return wanted


def extract(args):
    """Step 1: the claims of every question's options, from the text alone. Questions worded the same share a reply."""
    import torch
    from execute_tabalign import Reasoner

    questions = tv.load_questions(args.data_dir)
    wanted = wanted_videos(args, args.set)
    path = args.out_dir / "dynin_evidence" / "claims.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = tv.load_rows(path)
    replies = {row["digest"]: row["reply"] for row in rows.values() if row.get("digest")}
    prompts = {k: extract_prompt(questions[k]) for k in wanted}
    current = [k for k in wanted if k in rows and rows[k].get("digest") == digest(prompts[k])]   # read with this prompt
    reparse = [k for k in current if rows[k].get("claim") != parse_claim(rows[k]["reply"], len(questions[k]["candidates"]))]
    todo = [k for k in wanted if k not in current]
    print(f"{len(wanted)} questions: {len(current) - len(reparse)} already read, {len(reparse)} to parse again, "
          f"{len(todo)} to read", flush=True)

    def write(keys):
        with open(path, "a") as f:
            for k in keys:
                reply = replies[digest(prompts[k])]
                claim = parse_claim(reply, len(questions[k]["candidates"]))
                f.write(json.dumps({"category": k[0], "video": k[1], "digest": digest(prompts[k]), "reply": reply,
                                    "claim": claim}) + "\n")
                print(f"   {k[0]}/{k[1]}: {describe(claim)}"
                      + ("" if claim["options"] else f" (reply: {' '.join(reply.split())[:200]!r})"), flush=True)

    write(reparse)                      # same reply, read by the current parser
    waiting = {}
    for k in todo:
        waiting.setdefault(digest(prompts[k]), []).append(k)
    write([k for d, keys in waiting.items() if d in replies for k in keys])
    if all(d in replies for d in waiting):
        return

    reasoner = Reasoner(args.model, args.max_vision_tokens)
    tokenizer = reasoner.processor.tokenizer
    tokenizer.padding_side = "left"                      # generation continues each prompt at the right edge
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pending = [d for d in waiting if d not in replies]
    text_of = {digest(p): p for p in prompts.values()}
    for start in range(0, len(pending), args.batch):
        batch = pending[start:start + args.batch]
        texts = [reasoner.processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": text_of[d]}]}],
                                                        add_generation_prompt=True, tokenize=False) for d in batch]
        inputs = reasoner.processor(text=texts, padding=True, return_tensors="pt").to(reasoner.model.device)
        with torch.inference_mode():
            out = reasoner.model.generate(**inputs, max_new_tokens=args.extract_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None, pad_token_id=pad)
        for d, sequence in zip(batch, out):
            replies[d] = tokenizer.decode(sequence[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[{min(start + args.batch, len(pending))}/{len(pending)} differently worded questions]", flush=True)
        write([k for d in batch for k in waiting[d]])


def run(args):
    """Step 3: Qwen3-VL's letter scores on the plain video and both zooms, and its own attention for the controls."""
    import torch
    from execute_tabalign import Reasoner, Video
    from run_attention import build_prompt

    questions = tv.load_questions(args.data_dir)
    claims = tv.load_rows(args.out_dir / "dynin_evidence" / "claims.jsonl")
    direct = tv.load_rows(args.direct)
    wanted = wanted_videos(args, [args.set])
    path = args.out_dir / "dynin_evidence" / args.set / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {k: r for k, r in tv.load_rows(path).items() if r.get("claims_digest") == claims_digest(claims.get(k))}
    todo = [k for k in wanted if k not in done]          # new videos, and videos whose claims have changed
    print(f"{len(wanted)} videos, {len(wanted) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    reasoner = Reasoner(args.model, args.max_vision_tokens)
    size = {"shortest_edge": reasoner.min_pixels, "longest_edge": reasoner.max_pixels}
    letter_ids = []
    for letter in tv.LETTERS:
        ids = reasoner.tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"letter {letter!r} is {len(ids)} tokens")
        letter_ids.append(ids[0])

    def letter_scores(inputs, count):
        with torch.inference_mode():
            logits = reasoner.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        return [round(float(v), 4) for v in torch.log_softmax(logits[letter_ids[:count]], dim=-1).cpu()]

    def two_view_inputs(video, zoomed, text):
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "video"}, {"type": "text", "text": text}]}]
        template = reasoner.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        meta = {"total_num_frames": video.total, "fps": video.native_fps, "frames_indices": list(video.indices),
                "duration": video.total / video.native_fps}
        batch = reasoner.processor(text=[template], videos=[video.frames, zoomed], video_metadata=[meta, dict(meta)],
                                   do_sample_frames=False, size=size, return_tensors="pt")
        return dict(batch.to(reasoner.model.device).to(reasoner.model.dtype))

    def own_attention(video, text):
        """Qwen3-VL's attention from the prompt's last position over the video, per time step (map_profile)."""
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
        inputs = reasoner._video_inputs(messages, video)
        recorder = reasoner.recorder_class(reasoner.layers)
        try:
            with torch.inference_mode():
                reasoner.model(**inputs, logits_to_keep=1)
        finally:
            recorder.remove()
        maps = reasoner._video_attention(recorder, inputs, video)
        if maps is None:
            return None
        maps = np.asarray(maps, dtype=np.float32)
        return map_profile(maps[-tv.ATTN_LAYERS:].mean(axis=0), maps.mean(axis=0))

    for n, key in enumerate(todo, start=1):
        item = questions[key]
        count = len(item["candidates"])
        row = {"category": key[0], "video": key[1],
               "answer_letter": tv.LETTERS[item["candidates"].index(item["answer"])],
               "direct_correct": direct.get(key, {}).get("correct"), "direct_pred": direct.get(key, {}).get("pred_letter"),
               "claims_digest": claims_digest(claims.get(key))}
        start = time.time()
        try:
            if key not in claims:
                raise RuntimeError("no claims for this question (run --extract first)")
            claim = claims[key]["claim"]
            video = Video(tv.video_path(args.data_dir, *key), args.fps, args.max_frames)
            prompt = build_prompt(item)
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
            row["lp_plain"] = letter_scores(reasoner._video_inputs(messages, video), count)
            row["single_pred"] = tv.LETTERS[int(np.argmax(row["lp_plain"]))]

            maps = claim_maps(args, key, claim)
            box = zoom_box(maps)
            if box is not None:
                other = random_box(box, np.random.default_rng(tv.video_seed(key[1], args.seed)))
                row.update(zoom_box=box, random_box=other,
                           lp_zoom_dynin=letter_scores(two_view_inputs(video, zoom_frames(video.frames, box),
                                                                       TWO_VIEWS + prompt), count),
                           lp_zoom_random=letter_scores(two_view_inputs(video, zoom_frames(video.frames, other),
                                                                        TWO_VIEWS + prompt), count))
            row["qwen_profiles"] = {name: own_attention(video, text) for name, text in claim_queries(claim)
                                    if name != "question"}
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - start, 2)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if "error" in row:
            status = row["error"]
        else:
            scored = [("plain", row["lp_plain"])] + [(name, row[f"lp_{name}"]) for name in ("zoom_dynin", "zoom_random")
                                                     if f"lp_{name}" in row]
            status = ", ".join(f"{name} {tv.LETTERS[int(np.argmax(lp))]}" for name, lp in scored) + \
                f"; answer {row['answer_letter']}"
        print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: {status} ({row['seconds']}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Report

def fit_weights(samples, penalty=PENALTY, iterations=100):
    """score = a * log p(option) + w . evidence / spread, fitted by Newton's method on the log-likelihood of the right
    option (L2 penalty on w, a held very lightly near 1; components with no spread keep weight 0).
    samples: [(lp [n], evidence [n, K], answer)]. Returns (a, w, spread)."""
    k = len(COMPONENTS)
    stacked = np.concatenate([e for _, e, _ in samples]) if samples else np.zeros((0, k))
    spread = stacked.std(axis=0) if len(stacked) else np.zeros(k)
    use = spread > 1e-12
    reg = np.r_[1e-3, np.where(use, penalty, 1e6)]
    centre = np.r_[1.0, np.zeros(k)]
    data = []
    for lp, ev, y in samples:
        z = np.zeros_like(ev)
        z[:, use] = ev[:, use] / spread[use]
        data.append((np.column_stack([np.asarray(lp, dtype=np.float64), z]), y))

    def objective(theta):
        total = -0.5 * float((reg * (theta - centre) ** 2).sum())
        for x, y in data:
            s = x @ theta
            s = s - s.max()
            total += float(s[y] - math.log(np.exp(s).sum()))
        return total

    theta = centre.copy()
    current = objective(theta)
    for _ in range(iterations):
        grad, hess = -reg * (theta - centre), -np.diag(reg)
        for x, y in data:
            s = x @ theta
            p = np.exp(s - s.max())
            p /= p.sum()
            mean = p @ x
            grad += x[y] - mean
            hess -= (x * p[:, None]).T @ x - np.outer(mean, mean)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        t, improved = 1.0, False
        while t > 1e-6:
            candidate = theta - t * step
            value = objective(candidate)
            if value >= current:
                improved = True
                break
            t /= 2
        if not improved:
            break
        finished = value - current < 1e-9
        theta, current = candidate, value
        if finished:
            break
    return float(theta[0]), theta[1:], np.where(use, spread, 0.0)


def predict(lp, ev, fitted):
    a, w, spread = fitted
    if a <= 0 and not w.any():
        return int(np.argmax(lp))
    z = np.divide(ev, spread, out=np.zeros_like(ev), where=spread > 0)
    return int(np.argmax(a * np.asarray(lp, dtype=np.float64) + z @ w))


def time_shuffled(profiles, rng):
    """The same profiles with their time steps in a random order (one order for all of a video's maps)."""
    if not profiles:
        return profiles
    steps = len(next(iter(profiles.values()))["mass"])
    order = rng.permutation(steps)
    return {name: {k: [p[k][j] for j in order] if len(p[k]) == steps else p[k] for k in p}
            for name, p in profiles.items()}


def load_set(args, name, exclude, claims):
    questions = tv.load_questions(args.data_dir)
    sets = tv.load_sets(args.out_dir / "sets.json")
    rows = tv.load_rows(args.out_dir / "dynin_evidence" / name / "results.jsonl")
    videos = []
    for key in tv.select_videos(sets, name, exclude):
        if key not in rows or key not in claims:
            continue
        row, item, claim = rows[key], questions[key], claims[key]["claim"]
        if row.get("claims_digest") != claims_digest(claims[key]):
            continue                    # computed from older claims: run this set again first
        count = len(item["candidates"])
        maps = claim_maps(args, key, claim)
        dynin = {q: map_profile(*maps[q]) for q in maps if q != "question"}
        qwen = {q: p for q, p in (row.get("qwen_profiles") or {}).items() if p is not None}
        videos.append({"key": key, "item": item, "row": row, "claim": claim, "count": count,
                       "answer": tv.LETTERS.index(row["answer_letter"]),
                       "lp_plain": np.asarray(row["lp_plain"], dtype=np.float64),
                       "lp_zoom": np.asarray(row.get("lp_zoom_dynin", row["lp_plain"]), dtype=np.float64),
                       "lp_random": np.asarray(row.get("lp_zoom_random", row["lp_plain"]), dtype=np.float64),
                       "zoomed": "lp_zoom_dynin" in row, "dynin": dynin, "qwen": qwen,
                       "ev_dynin": option_evidence(claim, dynin, count), "ev_qwen": option_evidence(claim, qwen, count)})
    return videos


def evaluate(fit_videos, test_videos, lp_name, ev_fit, ev_test):
    """One set of weights for every question, fitted on fit_videos; predictions for test_videos."""
    fitted = fit_weights([(v[lp_name], ev_fit(i), v["answer"]) for i, v in enumerate(fit_videos)])
    return {i: predict(v[lp_name], ev_test(i), fitted) for i, v in enumerate(test_videos)}, fitted


def evidence_alone(v, ev, columns):
    """Whether this kind of evidence on its own ranks the right option first among the options it speaks to
    (None when the right option or fewer than two options have it)."""
    score = ev[:, columns].sum(axis=1)
    speaks = [i for i in range(v["count"]) if (ev[i, columns] != 0).any()]
    if v["answer"] not in speaks or len(speaks) < 2:
        return None
    return max(speaks, key=lambda i: score[i]) == v["answer"]


def report(args):
    claims = tv.load_rows(args.out_dir / "dynin_evidence" / "claims.jsonl")
    fit_videos = load_set(args, args.fit_set, [] if args.fit_set != args.set else args.exclude_set, claims)
    test_videos = load_set(args, args.set, args.exclude_set, claims)
    if not fit_videos or not test_videos:
        raise SystemExit("Nothing to report yet")
    n = len(test_videos)
    groups = [("All", list(range(n)))] + [(SHORT.get(c, c), [i for i, v in enumerate(test_videos) if v["key"][0] == c])
                                          for c in sorted({v["key"][0] for v in test_videos},
                                                          key=lambda c: tv.CATEGORIES.index(c) if c in tv.CATEGORIES else 99)]

    def right(preds):
        return [preds[i] == v["answer"] for i, v in enumerate(test_videos)]

    plain = {i: int(np.argmax(v["lp_plain"])) for i, v in enumerate(test_videos)}
    dynin, dynin_fit = evaluate(fit_videos, test_videos, "lp_plain", lambda i: fit_videos[i]["ev_dynin"],
                                lambda i: test_videos[i]["ev_dynin"])
    qwen, qwen_fit = evaluate(fit_videos, test_videos, "lp_plain", lambda i: fit_videos[i]["ev_qwen"],
                              lambda i: test_videos[i]["ev_qwen"])
    zoom = {i: int(np.argmax(v["lp_zoom"])) for i, v in enumerate(test_videos)}
    random_zoom = {i: int(np.argmax(v["lp_random"])) for i, v in enumerate(test_videos)}
    both, both_fit = evaluate(fit_videos, test_videos, "lp_zoom", lambda i: fit_videos[i]["ev_dynin"],
                              lambda i: test_videos[i]["ev_dynin"])
    methods = [("Qwen3-VL alone (letter scores)", plain), ("+ Dynin-Omni timing and motion", dynin),
               ("+ Qwen3-VL's own attention (control)", qwen), ("zoom on Dynin-Omni's box", zoom),
               ("zoom on a random box (control)", random_zoom), ("zoom on Dynin-Omni's box + timing, motion", both)]

    rng = np.random.default_rng(args.seed)            # null: Dynin-Omni's maps with their time steps shuffled
    shuffled_accuracy = {label: [] for label, _ in groups}
    for _ in range(args.shuffles):
        fit_ev = [option_evidence(v["claim"], time_shuffled(v["dynin"], rng), v["count"]) for v in fit_videos]
        test_ev = [option_evidence(v["claim"], time_shuffled(v["dynin"], rng), v["count"]) for v in test_videos]
        preds, _ = evaluate(fit_videos, test_videos, "lp_plain", fit_ev.__getitem__, test_ev.__getitem__)
        ok = right(preds)
        for label, members in groups:
            if members:
                shuffled_accuracy[label].append(np.mean([ok[i] for i in members]))

    base = right(plain)
    lines = [f"Qwen3-VL with Dynin-Omni's attention as evidence ({args.set}"
             + (f", without {', '.join(args.exclude_set)}" if args.exclude_set else "")
             + f"): {n} videos; one set of weights for every question, fitted on {args.fit_set} ({len(fit_videos)} videos)",
             "Accuracy in % (videos); vs alone = right only with this method / right only with Qwen3-VL alone, exact "
             "McNemar p. Categories are DyBench's, used only for this breakdown.", "",
             f"{'method':<44}" + "".join(f"{label:>14}" for label, _ in groups) + f"{'vs alone':>22}"]
    for label, preds in methods:
        ok = right(preds)
        cells = "".join(f"{100 * np.mean([ok[i] for i in m]):6.1f} ({len(m):3d})".rjust(14) if m else "-".rjust(14)
                        for _, m in groups)
        if preds is plain:
            versus = "-"
        else:
            a, b, p = tv.mcnemar(ok, base)
            versus = f"+{a}/-{b} p={p:.2g}"
        lines.append(f"{label:<44}{cells}{versus:>22}")
    cells = "".join((f"{100 * np.mean(shuffled_accuracy[label]):6.1f}{'':6}" if shuffled_accuracy[label] else "-").rjust(14)
                    for label, _ in groups)
    lines.append(f"{'+ Dynin-Omni, time steps shuffled (control)':<44}{cells}{f'mean of {args.shuffles}':>22}")

    lines.append("")
    for (label_a, preds_a), (label_b, preds_b) in [(methods[1], methods[2]), (methods[3], methods[4])]:
        a, b, p = tv.mcnemar(right(preds_a), right(preds_b))
        lines.append(f"{label_a.strip('+ ')} against {label_b.strip('+ ')}: +{a}/-{b}, p = {p:.2g}")
    for label, members in groups[1:]:
        a, b, p = tv.mcnemar([right(dynin)[i] for i in members], [base[i] for i in members])
        lines.append(f"   {label:<14} timing and motion against alone: +{a}/-{b}, p = {p:.2g}")
    lines.append(f"{sum(v['zoomed'] for v in test_videos)} of {n} videos had a zoom box (the rest keep the plain "
                 "scores in the zoom rows).")

    lines += ["", "What the options claim (test): share of questions with an order-in-time claim / a direction claim"]
    for label, members in groups:
        if members:
            order = np.mean([any(s["order"] for s in test_videos[i]["claim"]["options"].values()) for i in members])
            moves = np.mean([any(s["moves"] for s in test_videos[i]["claim"]["options"].values()) for i in members])
            lines.append(f"   {label:<14} {100 * order:5.1f}% / {100 * moves:5.1f}%")

    lines += ["", "Evidence on its own (test): how often it ranks the right option first among the options it speaks to"]
    for kind, columns in [("timing", [0]), ("motion and still", [1, 2])]:
        for source in ("dynin", "qwen"):
            cells = []
            for label, members in groups:
                says = [evidence_alone(test_videos[i], test_videos[i][f"ev_{source}"], columns) for i in members]
                says = [s for s in says if s is not None]
                cells.append(f"{label} {100 * np.mean(says):.1f}% of {len(says)}" if says else f"{label} -")
            lines.append(f"   {kind:<17}{'Dynin-Omni' if source == 'dynin' else 'Qwen3-VL':<12}" + "; ".join(cells))

    weights = {}
    lines += ["", "Weights (a on Qwen3-VL's log-probabilities; evidence weights per unit spread):"]
    for label, fitted in [("timing and motion", dynin_fit), ("Qwen3-VL's own attention", qwen_fit),
                          ("zoom + timing and motion", both_fit)]:
        a, w, spread = fitted
        weights[label] = {"a": a, **{c: float(w[j]) for j, c in enumerate(COMPONENTS)},
                          "spread": {c: float(spread[j]) for j, c in enumerate(COMPONENTS)}}
        lines.append(f"   {label}: a {a:.3f}, " + ", ".join(f"{c} {w[j]:+.3f}" + ("" if spread[j] > 0 else " (unused)")
                                                           for j, c in enumerate(COMPONENTS)))

    same = [v for v in test_videos if v["row"].get("direct_pred")]
    match = sum(v["row"]["single_pred"] == v["row"]["direct_pred"] for v in same)
    lines += ["", f"Check: Qwen3-VL alone picks the recorded direct answer on {match} of {len(same)} videos."]
    text = "\n".join(lines)
    print(text)
    out = args.out_dir / "dynin_evidence" / args.set
    (out / "summary.txt").write_text(text + "\n")
    (out / "weights.json").write_text(json.dumps({"fit_set": args.fit_set, "videos": len(fit_videos), **weights}, indent=1))
    print(f"\nWritten to {out / 'summary.txt'} and {out / 'weights.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="+", required=True, help="Sets from sets.json (one, except with --extract)")
    parser.add_argument("--extract", action="store_true", help="Step 1: read what each question's options claim")
    parser.add_argument("--report", action="store_true", help="Print and save the accuracy table")
    parser.add_argument("--exclude-set", nargs="*", default=[], help="Leave out the videos of these sets")
    parser.add_argument("--limit", type=int, help="Only the first N videos (one pair per category in turn)")
    parser.add_argument("--fit-set", default="calib", help="Set whose videos fit the weights (--report)")
    parser.add_argument("--shuffles", type=int, default=20, help="Shuffles for the time-shuffled control (--report)")
    parser.add_argument("--batch", type=int, default=8, help="Questions read at once (--extract)")
    parser.add_argument("--extract-tokens", type=int, default=320, help="Longest reply when reading claims")
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--direct", type=Path, default=Path("outputs/qwen3vl_8b/results.jsonl"))
    parser.add_argument("--model", help="Default: the Qwen3-VL run_attention.py used")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.extract:
        extract(args)
        return
    if len(args.set) != 1:
        parser.error("give one --set (several only with --extract)")
    args.set = args.set[0]
    if args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
