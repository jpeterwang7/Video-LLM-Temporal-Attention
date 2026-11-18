"""
Shared pieces of TabAlign for video: the evaluation sets, the planner's prompt and parser, the reasoner's tools and
prompts, the grounding masks, TabAttn's attention-overlap score and its calibration, and the statistics.

TabAlign (Kwok et al., NeurIPS 2026, arXiv 2605.14465) works on tables. Its parts and their counterparts here:

    table cell                       one of Qwen3-VL's video tokens: a 32x32 pixel square of one time step
    planner: LLaDA-8B-Instruct,      the same model and decoding, on the question, the options and compact metadata
      compact metadata                 (the video's length and number of time steps; it never sees the frames)
    six operations (filter, sort,    six video operations: select, lookup, filter, sort, compare, aggregate
      aggregate, lookup, compare,
      select)
    [target: col] -> cell mask m_t   [target: object; steps: ...] -> mask of squares. Matching a column name to its
                                     cells needs no vision; finding an object in a video does, so Dynin-Omni (a
                                     diffusion model that sees the video) grounds the target, on the same frames
                                     Qwen3-VL sees
    reasoner: Qwen3-VL-8B, at most   the same, calling tools that read the frames at full size (the image modality)
      6 tool steps, temperature 0.7
    TabAttn: R = attention on m_t /  the same formula over the video tokens, from the position that emits each tool
      attention on the table,          call, last 4 layers, leaving out Qwen3-VL's attention sinks (its top 1% of
      logistic calibration             squares, which otherwise decide R on their own; text cells have no such sinks);
                                       a two-parameter logistic fit on terminal correctness of the validation split,
                                       80/20 with early stopping
    in-loop feedback, stagnation     the same: score and rationale go into the next turn; the reasoner stops when the
      halt (0.02, 2 steps)             score has not improved by 0.02 for 2 steps and the state has not changed

Nothing here imports torch, so the planner, Dynin-Omni and Qwen3-VL scripts (each in its own environment) and the
summary all use it.
"""
import json
import math
import random
import re
import zlib
from pathlib import Path

import numpy as np

CATEGORIES = ["event_sequence", "moving_direction", "reversible_dynamics"]
LETTERS = "ABCDEFGH"
OPS = ["select", "lookup", "filter", "sort", "compare", "aggregate"]
TOOLS = ["f_lookup", "f_compare", "f_filter", "f_sort", "f_aggregate", "f_select", "f_final_answer"]

MAX_TOOL_STEPS = 6          # TabAlign's T_max
STAGNATION_DELTA = 0.02     # TabAlign's stagnation halt: no improvement above this ...
STAGNATION_STEPS = 2        # ... for this many consecutive steps, with the state unchanged
ATTN_LAYERS = 4             # TabAttn reads the last 4 layers
TARGET_FRACTION = 0.25      # share of a time step's squares in a target mask: the median share of a labelled time
                            # step the human boxes cover (243 labels)
SINK_FRACTION = 0.01        # top 1% of squares by attention are sinks (check_attention_sinks.py)

OUT_DIR = Path("outputs/tabalign")


# ---------------------------------------------------------------------------------------------------------------------
# Videos and evaluation sets

def video_path(data_dir, category, video):
    for path in (Path(data_dir) / "video" / category / category / video, Path(data_dir) / "video" / category / video):
        if path.exists():
            return path
    raise FileNotFoundError(f"Video not found: {category}/{video}")


def qwen_time_steps(path, fps=2.0, max_frames=32):
    """Qwen3-VL's time steps for a video without decoding it: (count, duration in seconds, time of each step).
    The same sampling as run_attention.py: an even number of frames, 2 per second, at most 32, merged in pairs."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    native, total = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    num = min(max(int(total / native * fps), 4), max_frames, total)
    num -= num % 2
    wanted = np.linspace(0, total - 1, num).round().astype(int)
    times = [float((wanted[i] + wanted[i + 1]) / 2 / native) for i in range(0, num, 2)]
    return num // 2, total / native, times


def pair_name(video):
    return video.replace("_reversed.mp4", ".mp4")


def twin(video):
    return video.replace("_reversed.mp4", ".mp4") if video.endswith("_reversed.mp4") else video.replace(".mp4", "_reversed.mp4")


def build_sets(split, es_pairs=100, seed=0, exclude=()):
    """The evaluation sets, as TabAlign keeps its verifier's data apart from its test questions.

    calib: the validation split (TabAttn's calibration, like TabAlign's attention standards from train/dev splits).
    test:  the test split (every pair with a labelled video: moving direction and reversible dynamics) and, since no
           event-order video is labelled, es_pairs event-order pairs from the training split, which nothing here trains
           on. Videos in `exclude` (those the earlier step 3 planned and carried out) are skipped.
    smoke: one pair per category from each, to check the whole pipeline in a few minutes."""
    exclude = {tuple(k) for k in exclude}
    test = [tuple(k) for k in split["test"]]
    pairs = sorted({pair_name(v) for c, v in split["train"] if c == "event_sequence"
                    and ("event_sequence", v) not in exclude and ("event_sequence", twin(v)) not in exclude})
    random.Random(seed).shuffle(pairs)
    chosen = set(pairs[:es_pairs])
    test += [tuple(k) for k in split["train"] if k[0] == "event_sequence" and pair_name(k[1]) in chosen]
    calib = [tuple(k) for k in split["val"]]

    def one_pair_each(videos):
        out = []
        for category in CATEGORIES:
            names = sorted(v for c, v in videos if c == category)
            first = next((v for v in names if twin(v) in names), None)
            if first:
                out += [(category, pair_name(first)), (category, twin(pair_name(first)))]
        return out

    sets = {"calib": sorted(calib), "test": sorted(test)}
    sets["smoke_calib"], sets["smoke_test"] = one_pair_each(sets["calib"]), one_pair_each(sets["test"])
    return sets


def load_sets(path=OUT_DIR / "sets.json"):
    return {name: [tuple(k) for k in videos] for name, videos in json.load(open(path)).items()}


def load_questions(data_dir):
    questions = {}
    for category in CATEGORIES:
        for q in json.load(open(Path(data_dir) / "json" / f"{category}.json")):
            questions[(category, q["video"])] = q
    return questions


def load_rows(path):
    """Rows of a .jsonl file keyed by (category, video), skipping rows with an error and a last line still being
    written (so a summary can be read while a run is going)."""
    rows = {}
    if Path(path).exists():
        for line in open(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in row:
                rows[(row["category"], row["video"])] = row
    return rows


def select_videos(sets, name, exclude=(), limit=None):
    """A set's videos in the executor's order (interleave_pairs), leaving out the videos of the sets in `exclude`
    (the smoke-test videos, looked at while debugging), then the first `limit`."""
    excluded = {k for other in exclude for k in sets.get(other, [])}
    return [k for k in interleave_pairs(sets[name]) if k not in excluded][:limit]


def interleave_pairs(keys):
    """Videos with each reversed twin next to its original, taking one pair from each category in turn, so a run read
    or stopped part way through covers every category."""
    from itertools import zip_longest
    groups = {}
    for category, video in keys:
        groups.setdefault((category, pair_name(video)), []).append((category, video))
    by_category = [[sorted(members) for (c, _), members in sorted(groups.items()) if c == category] for category in CATEGORIES]
    return [key for round_ in zip_longest(*by_category) for members in round_ if members for key in members]


def video_seed(video, seed=0):
    return (zlib.crc32(video.encode()) + seed) % 2**31


# ---------------------------------------------------------------------------------------------------------------------
# Planner (TabAlign Section 4.1, B.2)

PLAN_INSTRUCTIONS = """The video is {duration:.1f} seconds long and is shown as {steps} time steps, numbered 1 to {steps}.
Do not answer the question yet. Write a short plan for answering it: 2 to 4 numbered steps, one per line.
Each step uses one of these operations:
select: choose the time steps to look at
lookup: find an object in a time step: where it is, what state it is in, which way it faces
filter: find the time steps in which an event or action can be seen
sort: put events in the order in which they happen
compare: compare an object's position or state between time steps
aggregate: sum up how an object moves or changes over the whole video
End every step with its target: the object or event it is about, and its time steps (first, last, all, or numbers), in this form:
<step number>. <operation>: <what to do> [target: <object or event>; steps: <time steps>]

Example, for a different video and the question "In which direction is the dog moving?":
1. lookup: find where the dog is at the start [target: dog; steps: first]
2. lookup: find where the dog is at the end [target: dog; steps: last]
3. compare: compare the two positions to get the direction [target: dog; steps: first, last]"""
EXAMPLE_TARGETS = ["dog", "dog", "dog"]

STEP_LINE = re.compile(r"^\s*(?:step\s*)?(\d+)\s*[.):]\s*(.*)$", re.I)
TARGET_TAG = re.compile(r"\[\s*target\s*:\s*([^\]]*)\]", re.I)
OP_PREFIX = re.compile(r"^\s*\**\s*(select|lookup|look up|filter|sort|compare|aggregate)\b\s*\**\s*([:\-–]?)\s*", re.I)
TARGET_STEPS = re.compile(r"^(.*?)[;,]?\s*\b(?:time\s*)?steps?\s*:\s*(.*)$", re.I | re.S)


def question_text(item):
    lines = [item["question"], "Options:"] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(item["candidates"])]
    return "\n".join(lines)


def plan_prompt(item, steps, duration):
    return question_text(item) + "\n\n" + PLAN_INSTRUCTIONS.format(steps=steps, duration=duration)


def parse_steps(text, count):
    """'first, last', 'all', '2-4' or '1 and 5' -> sorted 0-based time steps, or None if nothing can be read."""
    if not text:
        return None
    text = text.lower()
    chosen = set()
    if re.search(r"\b(all|every|whole|entire|each)\b", text):
        chosen.update(range(count))
    if re.search(r"\b(first|start|beginning|begin|initial|earliest)\b", text):
        chosen.add(0)
    if re.search(r"\b(last|end|final|latest)\b", text):
        chosen.add(count - 1)
    if re.search(r"\b(middle|halfway)\b", text):
        chosen.add(count // 2)
    for start, end in re.findall(r"(\d+)(?:\s*(?:-|–|to)\s*(\d+))?", text):
        lo, hi = int(start), int(end or start)
        for s in range(min(lo, hi), max(lo, hi) + 1):
            if 1 <= s <= count:
                chosen.add(s - 1)
    return sorted(chosen) or None


GENERIC_TARGETS = {"video", "the video", "scene", "the scene", "frame", "frames", "image", "object", "objects",
                   "time step", "time steps", "step", "steps", "nothing", "none", "n/a", "all", "everything", "target",
                   "action", "actions", "event", "events", "answer", "options", "option"}


def clean_target(target, item=None):
    """A plan target as a grounding query: markup and stray symbols removed, "Action 1" replaced by the question's
    action (TabAlign resolves a [target: col] tag against the table's columns), and None for a target that names
    nothing in particular ("video", "scene"), which TabAlign treats as an unparseable tag."""
    if not target:
        return None
    text = re.sub(r"[<>&*`{}\[\]|]", " ", target)
    text = re.sub(r"\btarget\b\s*:?|\b(?:object|steps?)\s*:", " ", text, flags=re.I)  # tag words that leaked in
    text = re.sub(r"^[\s;,.:]+|[\s;,.:]+$", "", re.sub(r"\s*[;,]\s*(?=[;,])", "", text))
    if item is not None:
        actions = question_actions(item)
        text = re.sub(r"\b(?:action|event)\s*_?\s*([1-9])\b", lambda m: actions.get(m.group(1), m.group(0)), text,
                      flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ;,.:'\"")
    if text.lower() in GENERIC_TARGETS or not re.search(r"[a-z]{3}", text.lower()):
        return None
    return text


def parse_plan(text, count, item=None):
    """Numbered steps as {"op", "text", "target", "steps_text", "steps"}. A step without a readable target gets
    target None, and without readable time steps steps None: TabAlign then uses a uniform mask and treats the step's
    score as uninformative. With the question (item), targets are cleaned by clean_target."""
    steps = []
    for line in text.splitlines():
        match = STEP_LINE.match(line)
        if not match:
            continue
        body = match.group(2).strip()
        op = OP_PREFIX.match(body)
        tag = TARGET_TAG.search(body)
        # "lookup: find the dog" loses its label; "Compare the two positions" keeps its verb
        what = body[op.end():] if op and op.group(2) else body
        what = TARGET_TAG.sub("", what).replace("**", "").strip().rstrip(".").strip()
        target, steps_text = None, None
        if tag:
            inner = tag.group(1).strip()
            split = TARGET_STEPS.match(inner)
            if split:
                target, steps_text = split.group(1), split.group(2)
            else:
                # "[target: dog, first, last]": trailing pieces made only of step words are the time steps
                pieces = [p.strip() for p in re.split(r"[;,]", inner)]
                tail = []
                while len(pieces) > 1 and re.fullmatch(r"(first|last|all|middle|start|end|\d+(\s*-\s*\d+)?|\s|and)+",
                                                        pieces[-1].lower() or "x"):
                    tail.insert(0, pieces.pop())
                target, steps_text = (", ".join(pieces), ", ".join(tail)) if tail else (inner, None)
            target = clean_target(target.strip(" ;,.'\""), item)
            steps_text = steps_text.strip(" ;,.") if steps_text else None
        if not what and not target:
            continue
        name = op.group(1).lower().replace("look up", "lookup") if op else None
        steps.append({"op": name, "text": what, "target": target,
                      "steps_text": steps_text, "steps": parse_steps(steps_text, count)})
    return steps


def describe_step(step):
    """A parsed step written back in the plan's own format."""
    text = (f"{step['op']}: " if step.get("op") else "") + step["text"]
    if step.get("target") or step.get("steps_text"):
        text += f" [target: {step.get('target') or '?'}; steps: {step.get('steps_text') or '?'}]"
    return text


CONTENT_STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "its", "it", "which", "what", "with",
                "from", "at", "by", "for", "this", "that", "video", "frame", "frames", "step", "steps", "object",
                "objects", "position", "positions", "state", "action", "actions", "event", "events", "something"}


def content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in CONTENT_STOP and len(w) > 2}


def target_hallucinated(target, item):
    """TabAlign's plan-quality measure is column hallucination: a target naming no real column. Here a target is
    hallucinated if none of its words appear in the question or the options."""
    words = content_words(target or "")
    if not words:
        return None
    stems = {w[:5] for w in content_words(question_text(item))}
    return not any(w[:5] in stems for w in words)


# ---------------------------------------------------------------------------------------------------------------------
# Reasoner prompts and tool calls (TabAlign Section 4.2)

TOOLS_DOC = """Tools:
f_lookup(object, step): find the object in one time step: where it is (x and y from 0 to 1, measured from the left and from the top of the frame), how much of the frame it covers, what it is doing and which way it faces. Two objects can be looked up together, joined by "and".
f_compare(object, step_a, step_b): how the object moved and changed between two time steps.
f_filter(event): the time steps in which an event or action can be seen.
f_sort(event_a, event_b): which of two events or actions happens first.
f_aggregate(object): how the object moves over the whole video.
f_select(steps): one sentence on what each chosen time step shows, for example f_select(1, 4).
f_final_answer(letter): your answer, the letter of the correct option.
A step is a time step number, or first or last."""


def step_times_text(times):
    shown = [f"step {k + 1} at {t:.1f}s" for k, t in enumerate(times)]
    if len(shown) > 6:
        shown = shown[:3] + ["..."] + shown[-2:]
    return ", ".join(shown)


def reasoner_intro(item, times, plan_steps):
    """The first turn: the question, the tools, the plan if there is one, and the first plan step."""
    count = len(times)
    lines = [question_text(item), "",
             f"Answer the question by calling tools, one call per turn{', following the plan below' if plan_steps else ''}. "
             f"The video has {count} time steps, numbered 1 to {count} ({step_times_text(times)}).",
             TOOLS_DOC,
             "Reply with exactly one tool call, for example f_lookup(dog, 1). Only before f_final_answer, first say in "
             "one or two sentences what the observations show about the question."]
    if plan_steps:
        lines += ["", "Plan:"] + [f"{n}. {describe_step(s)}" for n, s in enumerate(plan_steps, start=1)]
        lines += ["", plan_step_prompt(1, plan_steps[0])]
    else:
        lines += ["", f"Call the tools you need, at most {MAX_TOOL_STEPS}, then f_final_answer."]
    return "\n".join(lines)


def plan_step_prompt(number, step):
    return f"Plan step {number}: {describe_step(step)}"


ANSWER_NOW = ("In one or two sentences, say what the observations show about the question, then call f_final_answer "
              "with the letter of the correct option.")
PLAN_DONE = "The plan is done. " + ANSWER_NOW
NO_PLAN_NEXT = "Next call."
BUDGET_DONE = "No tool calls are left. " + ANSWER_NOW
FINAL_FALLBACK = "Answer with the option's letter from the given choices directly."

CALL_START = re.compile(r"\b(f_(?:lookup|compare|filter|sort|aggregate|select|final_answer))\s*\(", re.I)


def parse_call(reply):
    """The tool call in a reply: (name, argument text), or None. The last f_final_answer if there is one (the
    sentences before it may name other tools), else the first call."""
    matches = list(CALL_START.finditer(reply))
    if not matches:
        return None
    finals = [m for m in matches if m.group(1).lower() == "f_final_answer"]
    match = finals[-1] if finals else matches[0]
    rest, depth = reply[match.end():], 1
    for i, ch in enumerate(rest):
        depth += (ch == "(") - (ch == ")")
        if depth == 0:
            return match.group(1).lower(), rest[:i].strip()
    return match.group(1).lower(), rest.split("\n")[0].strip()


def split_args(text):
    """Arguments of a call, in order: quoted strings whole, everything else split at commas; 'name=' prefixes go."""
    if text.count("'") % 2:
        text = text.replace("'", "’")         # an apostrophe ("person's"), not a quote
    parts = []
    for double, single, bare in re.findall(r"\"([^\"]*)\"|'([^']*)'|([^,\"']+)", text):
        part = strip_arg_name(double or single or bare)
        if part:
            parts.append(part)
    return parts


def strip_arg_name(text):
    return re.sub(r"^\s*[a-z_]+\s*=\s*", "", text, flags=re.I).strip(" \"'")


STEP_TOKEN = re.compile(r"(?:the\s+)?(?:time\s*)?(?:step|frame)?\s*(\d+|first|last|start|end|beginning|final|middle)"
                        r"(?:\s*(?:time\s*)?(?:step|frame))?")


def parse_step_token(text, count):
    """'3', 'step 3', 'time step 3', 'first', 'the last' -> a 0-based time step, or None (anything longer, such as
    'the end of the video', is not a step)."""
    match = STEP_TOKEN.fullmatch(strip_arg_name(str(text)).lower().strip())
    if not match:
        return None
    word = match.group(1)
    if word.isdigit():
        return min(max(int(word), 1), count) - 1
    return {"first": 0, "start": 0, "beginning": 0, "middle": count // 2}.get(word, count - 1)


def question_actions(item):
    """'Action 1. crack egg' lines of an event-order question -> {"1": "crack egg", ...}."""
    return dict(re.findall(r"Action\s*(\d)\s*[.:]\s*(.+)", item["question"]))


def resolve_event(text, item):
    text = strip_arg_name(text)
    ref = re.fullmatch(r"(?:action|event)\s*_?\s*(\d)", text.strip().lower())
    return question_actions(item).get(ref.group(1), text) if ref else text


def read_call(name, args, count, item):
    """Interpret a parsed call's arguments: a dict for the tool, or an error message. Time steps may come anywhere
    and several at once (f_lookup(person, 1, 5) looks in both); a dict with "tool" redirects the call
    (f_compare of two of the question's actions is the order question f_sort answers)."""
    parts = split_args(args)
    if name == "f_final_answer":
        letter = parse_letter(args, item["candidates"])
        return {"letter": letter} if letter else "f_final_answer needs one of the option letters"
    if name == "f_select":
        steps = [parse_step_token(p, count) for p in re.split(r"[,\s]+and\s+|,|\s+", args)]
        steps = sorted({s for s in steps if s is not None})
        return {"steps": steps[:4]} if steps else "f_select needs time steps"
    steps = list(dict.fromkeys(s for s in (parse_step_token(p, count) for p in parts) if s is not None))
    words = [p for p in parts if parse_step_token(p, count) is None]
    if name == "f_lookup":
        if not words or not steps:
            return "f_lookup needs an object and a time step"
        return {"object": ", ".join(words), "steps": steps[:4]}
    if name == "f_compare":
        actions = set(question_actions(item).values())
        events = [resolve_event(w, item) for w in words]
        if len(set(events) & actions) >= 2:
            return {"events": [e for e in events if e in actions][:2], "tool": "f_sort"}
        if not words or len(steps) < 2:
            return "f_compare needs an object and two time steps"
        return {"object": ", ".join(words), "steps": [steps[0], steps[-1]]}
    if name == "f_filter":
        return {"event": resolve_event(", ".join(words), item)} if words else "f_filter needs an event"
    if name == "f_sort":
        if len(words) == 1:
            words = re.split(r"\s+(?:and|vs\.?|versus|before|or)\s+", words[0], maxsplit=1)
        if len(words) < 2:
            return "f_sort needs two events"
        half = len(words) // 2
        return {"events": [resolve_event(", ".join(words[:half]), item), resolve_event(", ".join(words[half:]), item)]}
    if name == "f_aggregate":
        return {"object": ", ".join(words)} if words else "f_aggregate needs an object"
    return f"unknown tool {name}"


def call_target(name, parsed):
    """The object and time steps a call is about, for the mask when there is no plan (the executor-derived mask)."""
    if not isinstance(parsed, dict):
        return None, None
    if "object" in parsed:
        return parsed["object"], parsed.get("steps")
    if "event" in parsed:
        return parsed["event"], None
    if "events" in parsed:
        return " and ".join(parsed["events"]), None
    return None, parsed.get("steps")


def parse_letter(response, candidates):
    allowed = LETTERS[: len(candidates)]
    match = re.search(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", response)
    if match:
        return match.group(1)
    for i, candidate in enumerate(candidates):
        if candidate.lower().rstrip(".") in response.lower():
            return LETTERS[i]
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Tool outputs that are computed, not perceived

BOX = re.compile(r"\"?bbox(?:_2d)?\"?\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")


def parse_boxes(text, image_size=None):
    """Qwen3-VL's grounding reply -> [{"box": [x0, y0, x1, y1] as fractions of the frame, "label", "state"}].
    Qwen3-VL gives coordinates from 0 to 1000; if any is above 1000 they are taken as pixels of the image it was
    given (image_size = (width, height))."""
    found = []
    for match in BOX.finditer(text):
        box = [float(v) for v in match.groups()]
        scale = (1000.0, 1000.0)
        if max(box) > 1000 and image_size:
            scale = (float(image_size[0]), float(image_size[1]))
        x0, y0, x1, y1 = box[0] / scale[0], box[1] / scale[1], box[2] / scale[0], box[3] / scale[1]
        x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
        y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
            continue
        # the fields of the same entry: after the box up to the next entry, or before it back to the entry's start
        tail = text[match.end():match.end() + 400].split("}")[0]
        head = text[max(0, match.start() - 300):match.start()].split("{")[-1]

        def field(name):
            found_field = re.search(rf"\"{name}\"\s*:\s*\"([^\"]*)\"", tail) or re.search(rf"\"{name}\"\s*:\s*\"([^\"]*)\"", head)
            return found_field.group(1).strip() if found_field else None

        found.append({"box": [round(v, 4) for v in (x0, y0, x1, y1)], "label": field("label"), "state": field("state"),
                      "facing": field("facing")})
    return found


def box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def describe_box(name, found):
    x, y = box_center(found["box"])
    text = f"{name} at x {x:.2f}, y {y:.2f}, covering {100 * box_area(found['box']):.0f}% of the frame"
    notes = [n for n in (found.get("state"), facing_text(found)) if n]
    if notes:
        text += f" ({'; '.join(notes)})"
    return text


def facing_side(found):
    facing = ((found or {}).get("facing") or "").lower()
    if "left" in facing and "right" not in facing:
        return "left"
    if "right" in facing and "left" not in facing:
        return "right"
    return None


def facing_relation(found_a, found_b, still=0.03):
    """Whether an object moved the way it faces, from its left/right facing in the two lookups and its left/right
    movement between them (computed, like a table tool deriving a column); None if either is unclear."""
    dx = box_center(found_b["box"])[0] - box_center(found_a["box"])[0]
    sides = [s for s in (facing_side(found_a), facing_side(found_b)) if s]
    if abs(dx) < still or not sides:
        return None
    moved = "right" if dx > 0 else "left"
    if len(set(sides)) > 1:
        return f"it moved {moved}, but it faces {sides[0]} in the first and {sides[1]} in the second"
    if sides[0] == moved:
        return f"it moved {moved}, the way it faces"
    return f"it moved {moved}, opposite to the way it faces ({sides[0]})"


def facing_text(found):
    facing = (found or {}).get("facing")
    if not facing or facing.lower() in ("unclear", "unknown", "none", "n/a", ""):
        return None
    return facing if facing.lower().startswith("facing") else f"facing {facing}"


def movement_words(box_a, box_b, still=0.03):
    """How a box moved between two time steps, in words (y grows downwards)."""
    (xa, ya), (xb, yb) = box_center(box_a), box_center(box_b)
    dx, dy = xb - xa, yb - ya
    parts = []
    parts.append(f"moved {'right' if dx > 0 else 'left'} by {abs(dx):.2f} of the frame width (x {xa:.2f} to {xb:.2f})"
                 if abs(dx) >= still else f"did not move left or right (x {xa:.2f} to {xb:.2f})")
    parts.append(f"moved {'down' if dy > 0 else 'up'} by {abs(dy):.2f} of the frame height (y {ya:.2f} to {yb:.2f})"
                 if abs(dy) >= still else f"did not move up or down (y {ya:.2f} to {yb:.2f})")
    ratio = box_area(box_b) / max(box_area(box_a), 1e-6)
    if ratio > 4 or ratio < 0.25:
        parts.append("its box changed size too much to compare (the two boxes may not cover the same thing)")
    elif ratio > 1.2:
        parts.append(f"grew to {ratio:.1f} times its size (closer to the camera or larger)")
    elif ratio < 1 / 1.2:
        parts.append(f"shrank to {ratio:.1f} times its size (farther from the camera or smaller)")
    else:
        parts.append("kept about the same size")
    return "; ".join(parts)


def distance_words(name_a, name_b, boxes_a, boxes_b):
    """Change in distance between two objects' centres from one time step to another."""
    def dist(p, q):
        (x1, y1), (x2, y2) = box_center(p), box_center(q)
        return math.hypot(x1 - x2, y1 - y2)
    before, after = dist(boxes_a[0], boxes_a[1]), dist(boxes_b[0], boxes_b[1])
    if abs(after - before) < 0.03:
        return f"the {name_a} and the {name_b} stayed about as far apart ({before:.2f} to {after:.2f})"
    return f"the {name_a} and the {name_b} got {'closer' if after < before else 'farther apart'} ({before:.2f} to {after:.2f})"


def split_objects(text):
    """'pen and glass' or 'pen, glass' -> ['pen', 'glass']; one object -> [object]."""
    parts = [p.strip(" .'\"") for p in re.split(r"\s+and\s+|,|;", text) if p.strip(" .'\"")]
    return parts[:2] or [text.strip()]


def steps_text(steps):
    if not steps:
        return "none"
    runs, start = [], steps[0]
    for prev, cur in zip(steps, steps[1:] + [None]):
        if cur != prev + 1:
            runs.append(f"{start + 1}" if start == prev else f"{start + 1}-{prev + 1}")
            start = cur
    return ", ".join(runs)


# ---------------------------------------------------------------------------------------------------------------------
# Masks and TabAttn (TabAlign Eq. 1, B.4, E.5)

def resize_map(frames_map, height, width):
    """[T, h, w] -> [T, height, width] by area or bilinear resizing, frame by frame."""
    import cv2
    out = []
    for frame in np.asarray(frames_map, dtype=np.float32):
        shrink = height <= frame.shape[0] and width <= frame.shape[1]
        out.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR))
    return np.stack(out)


def place_on_grid(frames_map, grid, crop=(0.0, 0.0, 1.0, 1.0), frame_steps=None, scale=8):
    """Dynin-Omni maps [F, h, w] -> Qwen3-VL's grid [T, H, W]. crop is the part of the frame Dynin-Omni saw (fractions
    x0, y0, x1, y1; the whole frame when squeezed); frame_steps the time step of each map (default: map f is time step
    f; maps landing on the same time step are averaged). Squares Dynin-Omni did not see are NaN."""
    import cv2
    steps, height, width = grid
    x0, y0, x1, y1 = crop
    top, bottom = int(round(y0 * height * scale)), int(round(y1 * height * scale))
    left, right = int(round(x0 * width * scale)), int(round(x1 * width * scale))
    inside = np.zeros((height * scale, width * scale), dtype=np.float32)
    inside[top:bottom, left:right] = 1
    covered = cv2.resize(inside, (width, height), interpolation=cv2.INTER_AREA) > 0.5
    total, counts = np.zeros(grid, dtype=np.float64), np.zeros(steps)
    frame_steps = list(range(len(frames_map))) if frame_steps is None else list(frame_steps)
    for frame, t in zip(np.asarray(frames_map, dtype=np.float32), frame_steps):
        if not 0 <= t < steps:
            continue
        fine = np.zeros((height * scale, width * scale), dtype=np.float32)
        fine[top:bottom, left:right] = cv2.resize(frame, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
        total[t] += cv2.resize(fine, (width, height), interpolation=cv2.INTER_AREA)
        counts[t] += 1
    out = np.full(grid, np.nan)
    for t in range(steps):
        if counts[t]:
            out[t] = np.where(covered, total[t] / counts[t], np.nan)
    return out


def sink_squares(all_layer_map, fraction=SINK_FRACTION):
    """The top `fraction` of squares by attention averaged over all layers (squares with no value are never sinks)."""
    flat = np.nan_to_num(np.asarray(all_layer_map, dtype=np.float64).ravel(), nan=-np.inf)
    sinks = np.zeros(flat.size, dtype=bool)
    count = math.ceil(fraction * np.isfinite(flat).sum())
    if count:
        sinks[np.argsort(flat)[-count:]] = True
    return sinks.reshape(np.shape(all_layer_map))


def shuffled_r(attention, mask, rng, draws=20, eligible=None):
    """TabAlign's density-preserving null (D.4): R with the mask's squares moved at random within each of its time
    steps (among the eligible squares), the count per time step kept, averaged over draws."""
    if mask is None or attention is None:
        return None
    attention = np.asarray(attention, dtype=np.float64)
    eligible = np.ones(mask.shape, dtype=bool) if eligible is None else eligible
    values = []
    for _ in range(draws):
        shuffled = np.zeros_like(mask)
        for t in range(mask.shape[0]):
            k = int(mask[t].sum())
            pool = np.flatnonzero(eligible[t])
            if k and len(pool) >= k:
                shuffled[t].flat[rng.choice(pool, size=k, replace=False)] = True
        values.append(r_attn(attention, shuffled))
    return float(np.mean(values))


def sink_free_r(attention, all_layer_attention, mask, rng=None):
    """R (Eq. 1) without the reasoner's attention sinks. TabAlign scores text cells; Qwen3-VL's attention over video
    tokens piles onto a few sink tokens (check_attention_sinks.py), so whether one sink square falls inside a mask
    swings R from a few percent to over 80%. The top 1% of squares by all-layer attention are left out of both sums.
    Returns (R without sinks, TabAlign's literal R, the shuffled-mask null of the first)."""
    if mask is None or attention is None:
        return None, None, None
    sinks = sink_squares(all_layer_attention)
    clean = np.where(sinks, 0.0, np.asarray(attention, dtype=np.float64))
    kept = mask & ~sinks
    null = shuffled_r(clean, kept, rng, eligible=~sinks) if rng is not None else None
    return r_attn(clean, kept), r_attn(attention, mask), null


def compile_mask(score_map, steps, fraction=TARGET_FRACTION, sinks=None):
    """A grounding map [T, H, W] and the step's time steps -> binary mask m_t: the top `fraction` of each chosen time
    step's squares (sinks left out). None if the step has no time steps (TabAlign's uniform mask)."""
    if steps is None:
        return None
    score_map = np.asarray(score_map, dtype=np.float64)
    mask = np.zeros(score_map.shape, dtype=bool)
    for t in steps:
        if not 0 <= t < score_map.shape[0]:
            continue
        frame = score_map[t].copy()
        if sinks is not None:
            frame[sinks[t]] = -np.inf
        eligible = np.isfinite(frame)                   # squares the grounding model did not see are NaN
        if not eligible.any():
            continue
        k = max(1, int(round(fraction * eligible.sum())))
        order = np.argsort(np.where(eligible, frame, -np.inf).ravel())[::-1][:k]
        chosen = np.zeros(frame.size, dtype=bool)
        chosen[order] = True
        mask[t] = chosen.reshape(frame.shape) & eligible
    return mask if mask.any() else None


def label_cells_mask(label, grid):
    """A labels-file label (on Qwen3-VL's grid) -> bool [T, H, W]."""
    mask = np.zeros(grid, dtype=bool)
    if list(label["grid"]) != list(grid):
        return None
    for t, cells in label.get("cells", {}).items():
        for row, col in cells:
            mask[int(t), row, col] = True
    return mask


def oracle_mask(label_mask, steps, noise=0.0, rng=None):
    """TabAlign's Oracle row: the human mask, restricted to the step's time steps when it marks any of them; with
    noise > 0 every square flips with that probability (TabAlign's 20% noise row)."""
    if label_mask is None or not label_mask.any():
        return None
    mask = label_mask.copy()
    inside = np.ones(mask.shape[0], dtype=bool)
    if steps:
        chosen = np.zeros(mask.shape[0], dtype=bool)
        chosen[[t for t in steps if 0 <= t < mask.shape[0]]] = True
        if (mask & chosen[:, None, None]).any():
            mask &= chosen[:, None, None]
            inside = chosen
    if noise > 0:                       # flips only inside the time steps the mask covers
        flips = (rng or np.random.default_rng(0)).random(mask.shape) < noise
        mask ^= flips & inside[:, None, None]
    return mask if mask.any() else None


def r_attn(attention, mask):
    """TabAlign's Eq. 1: the share of the attention on the video tokens that falls inside the mask."""
    if mask is None or attention is None:
        return None
    attention = np.asarray(attention, dtype=np.float64)
    if attention.shape != mask.shape:
        return None
    return float((attention * mask).sum() / max(attention.sum(), 1e-6))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


def fit_calibration(scores, correct, groups, seed=0, lr=0.05, max_iter=20000, patience=500):
    """TabAttn's calibration: p(correct) = sigmoid(a * R + b), fit by binary cross-entropy on terminal correctness,
    80/20 split (by group, so a video and its twin stay together) with the inner 20% used only for early stopping."""
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    groups = np.asarray(groups)
    names = sorted(set(groups.tolist()))
    random.Random(seed).shuffle(names)
    held = set(names[: max(1, len(names) // 5)])
    inner = np.array([g in held for g in groups])
    mu, sd = x[~inner].mean(), x[~inner].std() or 1.0
    z = (x - mu) / sd

    def bce(a, b, rows):
        p = 1 / (1 + np.exp(-(a * z[rows] + b)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y[rows] * np.log(p) + (1 - y[rows]) * np.log(1 - p)).mean())

    prior = min(max(y[~inner].mean(), 1e-3), 1 - 1e-3)
    a, b = 0.0, math.log(prior / (1 - prior))
    best = (bce(a, b, inner), a, b, 0)
    since = 0
    for step in range(1, max_iter + 1):
        p = 1 / (1 + np.exp(-(a * z[~inner] + b)))
        a -= lr * float(((p - y[~inner]) * z[~inner]).mean())
        b -= lr * float((p - y[~inner]).mean())
        loss = bce(a, b, inner)
        if loss < best[0] - 1e-9:
            best, since = (loss, a, b, step), 0
        else:
            since += 1
            if since >= patience:
                break
    _, a, b, stopped = best
    return {"a": a / sd, "b": b - a * mu / sd, "n": int(len(x)), "n_inner": int(inner.sum()),
            "inner_bce": best[0], "inner_bce_prior_only": bce(0.0, math.log(prior / (1 - prior)), inner),
            "stopped_at": stopped, "prior": float(y.mean())}


def calibrated(score, calibration):
    if score is None or not calibration:
        return None
    return sigmoid(calibration["a"] * score + calibration["b"])


def feedback_text(score, share, target, steps, prior):
    """TabAttn's in-loop feedback: the calibrated score and a short rationale."""
    if score is None:
        return "Verifier: no score for this call (its target could not be placed in the video)."
    where = f" in time steps {steps_text(steps)}" if steps else ""
    verdict = ("looks grounded" if score >= prior else
               f"weakly grounded: look at the {target} again before relying on this step")
    return (f"Verifier: grounding score {score:.2f} ({100 * share:.0f}% of your attention on the video was on the "
            f"{target}{where}); {verdict}.")


# ---------------------------------------------------------------------------------------------------------------------
# Statistics

def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney), ties given their average rank."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binomial_sd(p, n):
    """TabAlign's Table 9: sqrt(p (1 - p) / n)."""
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


def exact_two_sided(k, n):
    """Two-sided exact binomial test of k successes in n at p = 0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar(a_correct, b_correct):
    """Paired comparison of two systems on the same videos: (only A right, only B right, exact p)."""
    only_a = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    only_b = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    return only_a, only_b, exact_two_sided(only_a, only_a + only_b)


def sign_test(diffs):
    diffs = [d for d in diffs if d == d and d != 0]
    up = sum(1 for d in diffs if d > 0)
    return up, len(diffs), exact_two_sided(up, len(diffs))


def pair_accuracy(correct_by_video):
    """DyBench's pair accuracy: a video and its _reversed twin must both be right."""
    pairs = [(v, twin(v)) for v in correct_by_video if not v.endswith("_reversed.mp4") and twin(v) in correct_by_video]
    return sum(correct_by_video[a] and correct_by_video[b] for a, b in pairs), len(pairs)


def crop_window(width, height, resolution=224):
    """The part of the frame (fractions x0, y0, x1, y1) that Resize(resolution) + CenterCrop(resolution) keeps, as in
    Dynin-Omni's image_transform (same as run_attention_dynin.crop_window, without its torch imports)."""
    if width <= height:
        new_w, new_h = resolution, int(resolution * height / width)
    else:
        new_w, new_h = int(resolution * width / height), resolution
    left, top = int(round((new_w - resolution) / 2.0)), int(round((new_h - resolution) / 2.0))
    return left / new_w, top / new_h, (left + resolution) / new_w, (top + resolution) / new_h


def main():
    """Write <out-dir>/sets.json once, from the split (make_split.py); later runs reuse it."""
    import argparse
    parser = argparse.ArgumentParser(description="Build the evaluation sets for TabAlign for video")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split", type=Path, default=Path("outputs/split.json"))
    parser.add_argument("--exclude", type=Path, default=Path("outputs/traces_train/traces.jsonl"),
                        help="Videos the earlier step 3 used, kept out of the event-order test sample")
    parser.add_argument("--es-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = args.out_dir / "sets.json"
    if not path.exists():
        exclude = list(load_rows(args.exclude)) if args.exclude.exists() else []
        sets = build_sets(json.load(open(args.split)), args.es_pairs, args.seed, exclude)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        json.dump({name: [list(k) for k in videos] for name, videos in sets.items()}, open(path, "w"), indent=1)
    for name, videos in load_sets(path).items():
        counts = {c: sum(k[0] == c for k in videos) for c in CATEGORIES}
        print(f"{name}: {len(videos)} videos, {counts}")
    print(f"Sets: {path}")


if __name__ == "__main__":
    main()
