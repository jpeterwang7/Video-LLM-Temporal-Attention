"""
TabAlign's grounded execution for video: Qwen3-VL carries out the plan by calling tools, and TabAttn checks each step.

Qwen3-VL watches the video exactly as run_attention.py gives it (2 frames per second, at most 32, merged in pairs into
time steps), reads the question, the options, the tool list and the plan, and then issues one tool call per turn,
one plan step at a time, at most 6 tool steps, temperature 0.7 (TabAlign Section 4.2, E.5). The tools are the six
plan operations; they look at single frames at full size (TabAlign's image modality) and compute what can be
computed (positions, movement, order) in code:

    f_lookup(object, step)        Qwen3-VL's grounding on that time step's frame: box, state, which way it faces
    f_compare(object, a, b)       both lookups, the movement between them in code, and Qwen3-VL on the two frames
    f_filter(event)               yes/no on every time step's frame
    f_sort(event_a, event_b)      which of the two each time step's frame shows; the order in code
    f_aggregate(object)           lookups across the video (up to 6 time steps), the path in code
    f_select(steps)               one sentence on each chosen frame
    f_final_answer(letter)

TabAttn. At the position that emits each tool call, Qwen3-VL's attention over the video tokens (mean of the last 4
layers and of the heads) is compared with the plan step's mask m_t: R = attention inside m_t / attention on the video
(Eq. 1). The mask is the top 25% of each of the step's time steps by a grounding map of the step's target:
    dynin         Dynin-Omni's attention for "Find the <target> in the video." (ground_dynin.py), the diffusion
                  model's understanding, as TabAlign's masks come from its diffusion planner
    qwen          Qwen3-VL's own attention for the same query (the autoregressive counterpart)
    oracle        the human label (labelled test videos only); oracle_noise flips 20% of its squares
A step with no readable time steps gets no mask and no score (TabAlign's uniform mask). The score
sigmoid(a R + b) (fit_tabattn.py) and a short rationale go into the next turn, and a stagnation halt ends the run when
the score has not improved by more than 0.02 for 2 steps and the last tool call added nothing new. R is recorded for
every mask source on every step, whichever one gives the feedback.

Configurations: --plan llada|qwen|none (none: Qwen3-VL picks its own tool calls, TabAlign's "LLM reasoning") and
--reward none|dynin|qwen|oracle|oracle_noise. Writes <out-dir>/runs/<set>/<plan>_<reward>/traces.jsonl (every turn:
the reply, the call, the observation, R per mask source, the score and the feedback) and attn/<category>/<video>.npz
(the attention behind each step), and prints accuracy next to Qwen3-VL's direct answers on the same videos.

Usage (main environment):
    python execute_tabalign.py --set test --plan llada --reward dynin
"""
import argparse
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import tabalign_video as tv

LOOKUP_PROMPT = """Locate {what} in this image. Reply with JSON only, a list with one entry per object:
[{{"bbox_2d": [x1, y1, x2, y2], "label": "<name>", "state": "<a few words: what it is doing or what state it is in, and which way it faces or moves>"}}]
Reply [] if it is not in the image."""
COMPARE_PROMPT = ("These are two frames of the same video: the first image is time step {a} ({ta:.1f}s) and the second "
                  "is time step {b} ({tb:.1f}s). In one sentence, how did {what} change between them (position, state "
                  "or shape)?")
FILTER_PROMPT = "Does this image show {event}? Answer yes or no."
SORT_PROMPT = "Which of these can be seen in this image?\n1. {a}\n2. {b}\n0. neither\nAnswer with the number only."
SELECT_PROMPT = "Describe what this image shows in one short sentence."
HALT_TEXT = ("The verifier stopped the run: the last steps did not improve the grounding and added nothing new. "
             "Call f_final_answer with the letter of the correct option.")


class Video:
    """The frames Qwen3-VL sees, read once per video."""

    def __init__(self, path, fps, max_frames):
        from omnitrace.backends.qwen3vl import read_video_frames
        self.path = str(path)
        self.frames, self.indices, self.native_fps, self.total = read_video_frames(self.path, fps, max_frames)
        self.count = len(self.frames) // 2
        self.times = [(self.indices[2 * t] + self.indices[2 * t + 1]) / 2 / self.native_fps for t in range(self.count)]
        self.grid = None

    def image(self, step, max_side):
        """The frame for a time step (the first of its pair; for the last time step the video's last sampled frame)."""
        from PIL import Image
        frame = self.frames[min(2 * step + (1 if step == self.count - 1 else 0), len(self.frames) - 1)]
        image = Image.fromarray(frame)
        scale = max_side / max(image.size)
        if scale < 1:
            image = image.resize((max(32, round(image.width * scale)), max(32, round(image.height * scale))), Image.BICUBIC)
        return image


class Reasoner:
    """Qwen3-VL-8B with last-query attention recording (OmniTrace's loader)."""

    def __init__(self, model_path, max_vision_tokens):
        from omnitrace.backends.qwen3vl import QWEN3VL_VIDEO_MIN_PIXELS, _LastQueryAttentionRecorder, load_model_qwen3vl
        from omnitrace.constants import QWEN3VL_PIXELS_PER_TOKEN
        self.bundle = load_model_qwen3vl(model_path)
        self.model, self.processor = self.bundle.model, self.bundle.processor
        self.tokenizer = self.processor.tokenizer
        self.layers = self.model.model.language_model.layers
        self.recorder_class = _LastQueryAttentionRecorder
        self.max_pixels = max_vision_tokens * QWEN3VL_PIXELS_PER_TOKEN
        self.min_pixels = min(QWEN3VL_VIDEO_MIN_PIXELS, self.max_pixels)
        self.merge = int(self.processor.video_processor.merge_size)
        self.patch = int(self.processor.image_processor.patch_size)

    def _video_inputs(self, messages, video):
        template = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                template.append(message)
            else:
                template.append({"role": message["role"], "content": [
                    {"type": "video"} if c["type"] == "video" else {"type": "text", "text": c["text"]} for c in content]})
        text = self.processor.apply_chat_template(template, add_generation_prompt=True, tokenize=False)
        batch = self.processor(text=[text], videos=[video.frames],
                               video_metadata=[{"total_num_frames": video.total, "fps": video.native_fps,
                                                "frames_indices": list(video.indices),
                                                "duration": video.total / video.native_fps}],
                               do_sample_frames=False, size={"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels},
                               return_tensors="pt")
        t, h, w = (int(x) for x in batch["video_grid_thw"][0].tolist())
        video.grid = (t, h // self.merge, w // self.merge)
        return dict(batch.to(self.model.device).to(self.model.dtype))

    def _video_attention(self, recorder, inputs, video):
        """Each layer's attention row of the last prompt position, over the video tokens: [layers, T, H, W]."""
        rows = torch.stack([layer_rows[0] for layer_rows in recorder.rows])
        positions = (inputs["input_ids"][0] == self.bundle.vision_token_id).nonzero().squeeze(-1).cpu()
        if len(positions) != int(np.prod(video.grid)):
            return None
        return rows[:, positions].reshape(len(self.layers), *video.grid).numpy()

    def chat(self, messages, video, sample, max_new_tokens, record=False):
        inputs = self._video_inputs(messages, video)
        recorder = self.recorder_class(self.layers) if record else None
        options = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if sample else \
            dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        options["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **options)
        finally:
            if recorder:
                recorder.remove()
        reply = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return reply, (self._video_attention(recorder, inputs, video) if record else None)

    def look(self, video, query):
        """Qwen3-VL's own grounding map for a target: attention of the last position of 'Find the <target> in the
        video.' over the video tokens, [layers, T, H, W]."""
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": f"Find the {query} in the video."}]}]
        inputs = self._video_inputs(messages, video)
        recorder = self.recorder_class(self.layers)
        try:
            with torch.inference_mode():
                self.model(**inputs)
        finally:
            recorder.remove()
        return self._video_attention(recorder, inputs, video)

    def ask_images(self, images, prompt, max_new_tokens):
        """One greedy reply about one or more frames; also the first image's size as the model saw it."""
        messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        batch = self.processor(text=[text], images=images, return_tensors="pt")
        _, h, w = (int(x) for x in batch["image_grid_thw"][0].tolist())
        inputs = dict(batch.to(self.model.device).to(self.model.dtype))
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        reply = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return reply, (w * self.patch, h * self.patch)


class Tools:
    """The six operations on one video. Perception goes to Qwen3-VL on single frames (greedy, so a repeated call
    returns the same observation); positions, movement and order are computed here."""

    def __init__(self, reasoner, video, item, image_side):
        self.r, self.v, self.item, self.side = reasoner, video, item, image_side
        self.lookups, self.answers, self.log = {}, {}, []

    def _ask(self, steps, prompt, tokens):
        key = (tuple(steps), prompt)
        if key not in self.answers:
            reply, size = self.r.ask_images([self.v.image(s, self.side) for s in steps], prompt, tokens)
            self.answers[key] = (reply, size)
            self.log.append({"steps": [s + 1 for s in steps], "prompt": prompt[:200], "reply": reply[:300]})
        return self.answers[key]

    def at(self, step):
        return f"time step {step + 1} ({self.v.times[step]:.1f}s)"

    def lookup(self, obj, step):
        names = tv.split_objects(obj)
        key = (tuple(n.lower() for n in names), step)
        if key not in self.lookups:
            what = " and ".join(f"the {n}" for n in names)
            reply, size = self._ask([step], LOOKUP_PROMPT.format(what=what), 200)
            found = tv.parse_boxes(reply, size)
            matched, left = {}, list(found)
            for name in names:          # by label first, then in order
                pick = next((f for f in left if f["label"] and (name.lower() in f["label"].lower()
                                                               or f["label"].lower() in name.lower())), None)
                if pick:
                    left.remove(pick)
                matched[name] = pick
            for name in names:
                if matched[name] is None and left:
                    matched[name] = left.pop(0)
            self.lookups[key] = matched
        return self.lookups[key]

    def run(self, name, args):
        if name == "f_lookup":
            step = args["steps"][0]
            found = self.lookup(args["object"], step)
            return f"In {self.at(step)}: " + "; ".join(
                tv.describe_box(f"the {n}", f) if f else f"the {n} was not found" for n, f in found.items()) + "."
        if name == "f_compare":
            a, b = args["steps"]
            first, second = self.lookup(args["object"], a), self.lookup(args["object"], b)
            parts = []
            for n in first:
                if first[n] and second[n]:
                    parts.append(f"the {n} " + tv.movement_words(first[n]["box"], second[n]["box"]))
                    states = [f.get("state") for f in (first[n], second[n])]
                    if any(states):
                        parts.append(f"the {n} at the first: {states[0] or '?'}; at the second: {states[1] or '?'}")
                else:
                    parts.append(f"the {n} was not found in both time steps")
            names = list(first)
            if len(names) == 2 and all(first[n] and second[n] for n in names):
                parts.append(tv.distance_words(names[0], names[1], [first[n]["box"] for n in names],
                                               [second[n]["box"] for n in names]))
            what = " and ".join(f"the {n}" for n in names)
            change, _ = self._ask([a, b], COMPARE_PROMPT.format(a=a + 1, b=b + 1, ta=self.v.times[a], tb=self.v.times[b],
                                                                what=what), 60)
            return (f"From {self.at(a)} to {self.at(b)}: " + "; ".join(parts) + f". Seen in the two frames: {change}")
        if name == "f_filter":
            seen = [s for s in range(self.v.count)
                    if self._ask([s], FILTER_PROMPT.format(event=args["event"]), 3)[0].strip().lower().startswith("yes")]
            if not seen:
                return f"{args['event']}: not seen in any time step."
            return (f"{args['event']}: seen in time steps {tv.steps_text(seen)} "
                    f"({self.v.times[seen[0]]:.1f}s to {self.v.times[seen[-1]]:.1f}s).")
        if name == "f_sort":
            a, b = args["events"]
            labels = {}
            for s in range(self.v.count):
                digit = re.search(r"[012]", self._ask([s], SORT_PROMPT.format(a=a, b=b), 3)[0])
                labels[s] = int(digit.group()) if digit else 0
            seen_a = [s for s, c in labels.items() if c == 1]
            seen_b = [s for s, c in labels.items() if c == 2]
            text = f"{a}: seen in time steps {tv.steps_text(seen_a)}; {b}: seen in time steps {tv.steps_text(seen_b)}."
            if not seen_a or not seen_b:
                return text + " One of them was not seen, so the order is unclear."
            mid_a, mid_b = np.mean(seen_a) + 1, np.mean(seen_b) + 1
            if abs(mid_a - mid_b) < 0.5:
                return text + f" They are seen at about the same time (middle at step {mid_a:.1f} and {mid_b:.1f})."
            first = a if mid_a < mid_b else b
            return text + f" {first} happens first (middle at step {min(mid_a, mid_b):.1f} against {max(mid_a, mid_b):.1f})."
        if name == "f_aggregate":
            obj = tv.split_objects(args["object"])[0]
            steps = sorted(set(np.linspace(0, self.v.count - 1, min(self.v.count, 6)).round().astype(int).tolist()))
            path = [(s, self.lookup(obj, s)[obj]) for s in steps]
            path = [(s, f) for s, f in path if f]
            if len(path) < 2:
                return f"The {obj} was found in fewer than two time steps, so its path is unknown."
            points = "; ".join(f"step {s + 1} x {tv.box_center(f['box'])[0]:.2f} y {tv.box_center(f['box'])[1]:.2f}"
                               for s, f in path)
            return (f"The {obj} through the video: {points}. From step {path[0][0] + 1} to step {path[-1][0] + 1} it "
                    + tv.movement_words(path[0][1]["box"], path[-1][1]["box"]) + ".")
        if name == "f_select":
            return " ".join(f"{self.at(s).capitalize()}: {self._ask([s], SELECT_PROMPT, 40)[0]}" for s in args["steps"])
        return f"Error: unknown tool {name}."


def default_call(step, count):
    """The call a plan step implies, used when the reply has no readable call (TabAlign's recoverable parse error)."""
    target = step.get("target") or "main object"
    steps = step.get("steps") or [0, count - 1]
    op = step.get("op")
    if op == "lookup":
        return "f_lookup", {"object": target, "steps": [steps[0]]}
    if op == "compare":
        return "f_compare", {"object": target, "steps": [steps[0], steps[-1]] if len(steps) > 1 else [0, count - 1]}
    if op == "filter":
        return "f_filter", {"event": target}
    if op == "sort":
        parts = tv.split_objects(target)
        return ("f_sort", {"events": parts}) if len(parts) == 2 else ("f_filter", {"event": target})
    if op == "aggregate":
        return "f_aggregate", {"object": target}
    return "f_select", {"steps": sorted(set(steps))[:4]}


class Masks:
    """Grounding maps on Qwen3-VL's grid for one video, and the step masks built from them."""

    def __init__(self, reasoner, video, dynin_files, label, rng, record_qwen):
        self.r, self.v, self.label, self.rng = reasoner, video, label, rng
        self.dynin, self.qwen, self.record_qwen = {}, {}, record_qwen
        for mode, path in dynin_files.items():
            if path and path.exists():
                data = np.load(path)
                self.dynin[mode] = {"queries": [str(q) for q in data["queries"]], "last4": data["last4"].astype(np.float32),
                                    "all": data["all"].astype(np.float32), "crop": data["crop"].astype(np.float64)}

    def _dynin_map(self, mode, target):
        data = self.dynin.get(mode)
        if not data or target not in data["queries"] or self.v.grid is None:
            return None
        i = data["queries"].index(target)
        if data["last4"].shape[1] != self.v.grid[0]:
            return None
        # with the centre crop, squares outside it were not seen and cannot be in the mask (NaN)
        ranking = tv.place_on_grid(data["last4"][i], self.v.grid, tuple(data["crop"]))
        return ranking, tv.sink_squares(tv.place_on_grid(data["all"][i], self.v.grid, tuple(data["crop"])))

    def _qwen_map(self, target):
        if target not in self.qwen:
            attention = self.r.look(self.v, target)
            self.qwen[target] = None if attention is None else (
                attention[-tv.ATTN_LAYERS:].mean(axis=0), tv.sink_squares(attention.mean(axis=0)))
        return self.qwen[target]

    def mask(self, source, target, steps):
        if steps is None or self.v.grid is None:
            return None
        if source in ("oracle", "oracle_noise"):
            if not self.label:
                return None
            human = tv.label_cells_mask(self.label, self.v.grid)
            return tv.oracle_mask(human, steps, noise=0.2 if source == "oracle_noise" else 0.0, rng=self.rng)
        if not target:
            return None
        found = self._qwen_map(target) if source == "qwen" else self._dynin_map(source.replace("dynin_", ""), target)
        if found is None:
            return None
        ranking, sinks = found
        return tv.compile_mask(ranking, steps, tv.TARGET_FRACTION, sinks)

    def sources(self, reward):
        names = [f"dynin_{m}" for m in ("squeeze", "crop") if m in self.dynin]
        if self.record_qwen or reward == "qwen":
            names.append("qwen")
        if self.label:
            names += ["oracle", "oracle_noise"]
        return names


def run_video(reasoner, item, key, plan_steps, args, calibration, masks_args, dynin_choice):
    video = Video(tv.video_path(args.data_dir, *key), args.fps, args.max_frames)
    count = video.count
    torch.manual_seed(tv.video_seed(key[1], args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tv.video_seed(key[1], args.seed))
    tools = Tools(reasoner, video, item, args.image_side)
    masks = Masks(reasoner, video, *masks_args)
    plan = [dict(s, steps=[t for t in (s.get("steps") or []) if t < count] or None) for s in plan_steps][:tv.MAX_TOOL_STEPS] \
        if plan_steps else None
    feedback_source = {"dynin": dynin_choice}.get(args.reward, args.reward)
    fit = calibration["sources"].get(dynin_choice if args.reward.startswith(("dynin", "oracle")) else args.reward) \
        if args.reward != "none" else None
    prior = calibration.get("prior", 0.5) if calibration else 0.5

    messages = [{"role": "user", "content": [{"type": "video", "video": video.path},
                                             {"type": "text", "text": tv.reasoner_intro(item, video.times, plan)}]}]
    turns, observations, attention_maps = [], [], []
    final, bypass, halted, best, stagnant = None, False, False, None, 0
    for i in range(tv.MAX_TOOL_STEPS):
        reply, attention = reasoner.chat(messages, video, sample=True, max_new_tokens=args.call_tokens, record=True)
        step = plan[i] if plan and i < len(plan) else None
        turn = {"plan_step": i + 1 if step else None, "reply": reply}
        call = tv.parse_call(reply)
        if call is not None:
            name, parsed = call[0], tv.read_call(call[0], call[1], count, item)
        elif step is not None:
            (name, parsed), turn["parse_error"] = default_call(step, count), True
        else:
            name, parsed, turn["parse_error"] = None, "no tool call found; reply with exactly one tool call", True
        if name == "f_final_answer" and isinstance(parsed, dict):
            final, bypass = parsed["letter"], bool(plan) and i < len(plan)
            turn["call"] = f"f_final_answer({final})"
            turns.append(turn)
            break

        # TabAttn: the step's target (the plan's; without a plan, the call's own)
        if step is not None:
            target, steps = step.get("target"), step.get("steps")
        else:
            target, steps = tv.call_target(name, parsed)
            steps = steps if steps is not None else (list(range(count)) if target else None)
        att4 = att_all = None
        if attention is not None:
            att4, att_all = attention[-tv.ATTN_LAYERS:].mean(axis=0), attention.mean(axis=0)
            attention_maps.append((att4, att_all))
        sinks = tv.sink_squares(att_all) if att_all is not None else None
        r_values, r_nosink, r_null = {}, {}, {}
        for source in masks.sources(args.reward):
            mask = masks.mask(source, target, steps)
            r_values[source] = tv.r_attn(att4, mask)
            r_nosink[source] = tv.r_attn(np.where(sinks, 0, att4), mask & ~sinks) if mask is not None and att4 is not None else None
            r_null[source] = tv.shuffled_r(att4, mask, masks.rng)

        observation = tools.run(name, parsed) if isinstance(parsed, dict) else f"Error: {parsed}."
        turn.update(call=name, args=parsed if isinstance(parsed, dict) else None, observation=observation,
                    tool_error=not isinstance(parsed, dict), target=target, steps=steps, R=r_values, R_nosink=r_nosink,
                    R_null=r_null)
        unchanged = observation in observations
        observations.append(observation)

        feedback = None
        if fit:
            share = r_values.get(feedback_source)
            score = tv.calibrated(share, fit)
            feedback = tv.feedback_text(score, share or 0.0, target, steps, prior)
            turn.update(score=score, feedback=feedback)
            if score is not None:
                stagnant = stagnant + 1 if best is not None and score - best <= tv.STAGNATION_DELTA else 0
                best = score if best is None else max(best, score)
                halted = stagnant >= tv.STAGNATION_STEPS and unchanged
        text = f"Observation: {observation}" + (f"\n{feedback}" if feedback else "")
        if halted:
            text += "\n" + HALT_TEXT
        elif i == tv.MAX_TOOL_STEPS - 1:
            text += "\n" + tv.BUDGET_DONE
        elif plan and i + 1 < len(plan):
            text += "\n\n" + tv.plan_step_prompt(i + 2, plan[i + 1])
        elif plan:
            text += "\n" + tv.PLAN_DONE
        else:
            text += "\n" + tv.NO_PLAN_NEXT
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": text}]
        turns.append(turn)
        if halted:
            break

    fallback = False
    if final is None:
        reply, _ = reasoner.chat(messages, video, sample=True, max_new_tokens=16)
        call = tv.parse_call(reply)
        parsed = tv.read_call(call[0], call[1], count, item) if call else None
        if call and call[0] == "f_final_answer" and isinstance(parsed, dict):
            final = parsed["letter"]
        elif call is None:
            final = tv.parse_letter(reply, item["candidates"])
        turns.append({"reply": reply, "call": "final"})
        if final is None:
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": tv.FINAL_FALLBACK}]
            reply, _ = reasoner.chat(messages, video, sample=False, max_new_tokens=8)
            final, fallback = tv.parse_letter(reply, item["candidates"]), True
            turns.append({"reply": reply, "call": "fallback"})

    extra = {}
    if attention_maps:
        extra["att4"] = np.stack([a for a, _ in attention_maps]).astype(np.float16)
        extra["att_all"] = np.stack([b for _, b in attention_maps]).astype(np.float16)
    if masks.qwen:
        keys = [k for k, v in masks.qwen.items() if v is not None]
        if keys:
            extra["qwen_queries"] = np.array(keys)
            extra["qwen_last4"] = np.stack([masks.qwen[k][0] for k in keys]).astype(np.float16)
    calls = [t for t in turns if t.get("call") not in (None, "final", "fallback") and not str(t.get("call")).startswith("f_final")]
    return {
        "time_steps": count, "grid": list(video.grid) if video.grid else None, "frame_times": video.times,
        "plan": plan, "turns": turns, "tool_log": tools.log, "final_letter": final, "bypass": bypass, "halted": halted,
        "fallback_final": fallback, "n_calls": len(calls), "parse_errors": sum(bool(t.get("parse_error")) for t in turns),
        "tool_errors": sum(bool(t.get("tool_error")) for t in turns),
    }, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, help="A set from sets.json (calib, test, smoke_calib, smoke_test)")
    parser.add_argument("--plan", choices=["llada", "qwen", "none"], default="llada")
    parser.add_argument("--reward", choices=["none", "dynin", "qwen", "oracle", "oracle_noise"], default="dynin")
    parser.add_argument("--record-qwen", action="store_true", help="Also score every step with Qwen3-VL's own masks")
    parser.add_argument("--only-labelled", action="store_true", help="Only videos with a usable human label")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=tv.OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("DyBench"))
    parser.add_argument("--labels", type=Path, nargs="*", default=sorted(Path("labels").glob("labels_peter*.json")))
    parser.add_argument("--direct", type=Path, default=Path("outputs/qwen3vl_8b/results.jsonl"))
    parser.add_argument("--model", help="Default: the Qwen3-VL run_attention.py used")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-vision-tokens", type=int, default=3072)
    parser.add_argument("--image-side", type=int, default=640, help="Longest side of the frames the tools read")
    parser.add_argument("--call-tokens", type=int, default=64, help="Most tokens for each tool call")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sets = tv.load_sets(args.out_dir / "sets.json")
    questions = tv.load_questions(args.data_dir)
    labels = {}
    for path in args.labels:
        for label in json.load(open(path))["labels"]:
            if label["done"] and not label["unsure"] and label.get("approved", True):
                labels[(label["category"], label["video"])] = label
    wanted = sets[args.set][:args.limit]
    if args.only_labelled:
        wanted = [k for k in wanted if k in labels]
    plans = tv.load_rows(args.out_dir / f"plans_{args.plan}.jsonl") if args.plan != "none" else {}
    calibration = {}
    if args.reward != "none":
        calibration = json.load(open(args.out_dir / "calibration.json"))
    dynin_choice = calibration.get("dynin_choice", "dynin_squeeze")
    direct = {k: r["correct"] for k, r in tv.load_rows(args.direct).items()}

    run_dir = args.out_dir / "runs" / args.set / f"{args.plan}_{args.reward}"
    run_dir.mkdir(parents=True, exist_ok=True)
    traces_path = run_dir / "traces.jsonl"
    done = tv.load_rows(traces_path)
    todo = [k for k in wanted if k not in done]
    missing = [k for k in todo if args.plan != "none" and not plans.get(k, {}).get("steps")]
    print(f"{len(wanted)} videos, {len(wanted) - len(todo)} already done, {len(todo)} to run"
          + (f" ({len(missing)} without a parsed plan run without one)" if missing else ""), flush=True)
    if args.reward != "none":
        print(f"TabAttn feedback from {args.reward} masks" + (f" ({dynin_choice})" if args.reward == "dynin" else ""), flush=True)

    reasoner = Reasoner(args.model, args.max_vision_tokens) if todo else None
    for n, key in enumerate(todo, start=1):
        item = questions[key]
        row = {"category": key[0], "video": key[1], "question": item["question"], "candidates": item["candidates"],
               "answer": item["answer"], "answer_letter": tv.LETTERS[item["candidates"].index(item["answer"])],
               "plan_source": args.plan, "reward": args.reward, "direct_correct": direct.get(key)}
        start = time.time()
        try:
            dynin_files = {m: args.out_dir / "dynin" / m / key[0] / (Path(key[1]).stem + ".npz") for m in ("squeeze", "crop")}
            rng = np.random.default_rng(tv.video_seed(key[1], args.seed))
            result, extra = run_video(reasoner, item, key, plans.get(key, {}).get("steps"), args, calibration,
                                      (dynin_files, labels.get(key), rng, args.record_qwen), dynin_choice)
            row.update(result)
            row["correct"] = row["final_letter"] == row["answer_letter"]
            if extra:
                path = run_dir / "attn" / key[0] / (Path(key[1]).stem + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **extra)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - start, 2)
        with open(traces_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if "error" not in row:
            done[key] = row
        status = row.get("error") or (f"{row['n_calls']} calls, final {row['final_letter']} "
                                      f"({'right' if row['correct'] else 'wrong'}, direct "
                                      f"{ {True: 'right', False: 'wrong'}.get(row['direct_correct'], '?') })"
                                      + (", halted" if row["halted"] else "") + (", bypass" if row["bypass"] else ""))
        print(f"[{n}/{len(todo)}] {key[0]}/{key[1]}: {status} ({row['seconds']}s)", flush=True)

    rows = [done[k] for k in wanted if k in done]
    print(f"\n{len(rows)} of {len(wanted)} videos finished ({args.plan} plan, {args.reward} reward).")
    for category in [None] + tv.CATEGORIES:
        group = [r for r in rows if category in (None, r["category"])]
        if not group:
            continue
        compared = [r for r in group if r.get("direct_correct") is not None]
        only_new, only_direct, p = tv.mcnemar([r["correct"] for r in compared], [r["direct_correct"] for r in compared])
        print(f"{category or 'All':<22} {sum(r['correct'] for r in group)} / {len(group)} = "
              f"{100 * np.mean([r['correct'] for r in group]):.1f}%; Qwen3-VL directly "
              f"{sum(r['direct_correct'] for r in compared)} / {len(compared)} on the same videos "
              f"({only_new} right only here, {only_direct} only directly, McNemar p = {p:.3g})")
    print(f"Traces: {traces_path}")


if __name__ == "__main__":
    main()
