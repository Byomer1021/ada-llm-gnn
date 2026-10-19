"""
LLM data generation (teacher -> verification -> student), stage 1: two-room buildings.

Each loop iteration:
  1. pick a random scenario (target verdict + failure cause + door placement)
  2. send ONE fresh request: system prompt (always the same) + user prompt (changes)
  3. split the answer into reasoning trace and building JSON
  4. verify with the rule engine (src/ada/rules.py) + structure / plausibility checks
  5. save every attempt to <role>_raw.jsonl, the correct ones to <role>_verified.jsonl
     together with their labels (verdict + targets of the three GNN agents),
     and (optional) write each verified building as an IFC file

The same script runs the TEACHER (big model via API) and the STUDENT (fine-tuned
Qwen 7B served locally with an OpenAI-compatible server such as vLLM), so all
student outputs go through the same rule-engine check.

Usage
  # teacher, Nemotron (build.nvidia.com key)
  export NVIDIA_API_KEY=nvapi-...
  python -m src.llm.generate --role teacher --provider nemotron --n 20
  # several batches: give each run a new --seed (it fixes the scenario order; every record
  # stores its seed and the script warns if the seed was already used for this role)
  python -m src.llm.generate --role teacher --provider nemotron --n 20 --seed 2
  # several requests at the same time (answers are still saved in request order)
  python -m src.llm.generate --role teacher --provider nemotron --n 60 --seed 3 --workers 4 --write-ifc
  # top up a scenario that came out rarely (scenarios are drawn uniformly by default)
  python -m src.llm.generate --role teacher --provider nemotron --n 16 --seed 4 --scenario r1_small_turn --write-ifc
  # teacher, GPT-4.1 (no reasoning trace is returned by this API, only the answer)
  export OPENAI_API_KEY=sk-...
  python -m src.llm.generate --role teacher --provider openai --model gpt-4.1 --n 20
  # student, local server (e.g. vllm serve <finetuned-qwen> --port 8000)
  python -m src.llm.generate --role student --provider local --model qwen-ada --n 2000 --write-ifc
  # no API at all: fake answers, to test the script. They go to their own files
  # (teacher_dryrun_*.jsonl, data/synthetic/teacher_dryrun/), never into the real data
  python -m src.llm.generate --dry-run --n 10
  # verified teacher records -> training file for the student
  python -m src.llm.generate --export-sft
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ada.rules import ACCESSIBLE, EXTERIOR, NOT_ACCESSIBLE, door_side, evaluate, load_rules, make_labels  # noqa: E402

OUT_DIR = ROOT / "data" / "llm"
SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
USER_TEMPLATE = (ROOT / "prompts" / "user_template.txt").read_text(encoding="utf-8")
PROMPT_VERSION = 5

PROVIDERS = {
    "nemotron": {"base_url": "https://integrate.api.nvidia.com/v1", "key": "NVIDIA_API_KEY",
                 "model": "nvidia/nemotron-3-ultra-550b-a55b",
                 "extra": {"chat_template_kwargs": {"enable_thinking": True}}},
    "openai": {"base_url": None, "key": "OPENAI_API_KEY", "model": "gpt-4.1", "extra": {}},
    "local": {"base_url": "http://localhost:8000/v1", "key": None, "model": "qwen-ada", "extra": {}},
}

A, NA = ACCESSIBLE, NOT_ACCESSIBLE
ADJ = "D1 and D2 are on ADJACENT walls of R1 (passing through R1 needs a turn)"
OPP = "D1 and D2 are on OPPOSITE walls of R1 (R1 can be crossed in a straight line)"
ANY = "free choice"


def cause(text, element=None, field=None, rule=None, value=None):
    """One possible failure cause. `rule` is set only if it must appear in failing_rules."""
    return {"text": text, "element": element, "field": field, "rule": rule, "value": value}


# two-room chain: EXTERIOR -D1- R1 -D2- R2.  label = target building verdict.
SCENARIOS = [
    {"id": "all_ok", "label": A, "placement": ANY, "causes": [cause("none, every value complies")]},
    {"id": "d2_width", "label": NA, "placement": ANY,
     "causes": [cause("door D2 clear width is too narrow", "D2", "clear_width_mm", "ADA 404.2.3")]},
    {"id": "d2_threshold", "label": NA, "placement": ANY,
     "causes": [cause("door D2 threshold is too high", "D2", "threshold_height_mm", "ADA 404.2.5")]},
    {"id": "d2_hardware", "label": NA, "placement": ANY,
     "causes": [cause("door D2 hardware is mounted too high", "D2", "hardware_height_mm", "ADA 404.2.7", "high"),
                cause("door D2 hardware is mounted too low", "D2", "hardware_height_mm", "ADA 404.2.7", "low")]},
    {"id": "d2_maneuvering", "label": NA, "placement": ANY,
     "causes": [cause("door D2 has too little maneuvering clearance on the pull side", "D2", "maneuvering_pull_mm",
                      "ADA 404.2.4"),
                cause("door D2 has too little maneuvering clearance on the push side", "D2", "maneuvering_push_mm",
                      "ADA 404.2.4")]},
    {"id": "d1_fails", "label": NA, "placement": ANY,
     "causes": [cause("entrance door D1 clear width is too narrow", "D1", "clear_width_mm", "ADA 404.2.3"),
                cause("entrance door D1 threshold is too high", "D1", "threshold_height_mm", "ADA 404.2.5"),
                cause("entrance door D1 has too little maneuvering clearance on the pull side", "D1",
                      "maneuvering_pull_mm", "ADA 404.2.4"),
                cause("entrance door D1 has too little maneuvering clearance on the push side", "D1",
                      "maneuvering_push_mm", "ADA 404.2.4"),
                cause("entrance door D1 hardware is mounted too high", "D1", "hardware_height_mm", "ADA 404.2.7", "high"),
                cause("entrance door D1 hardware is mounted too low", "D1", "hardware_height_mm", "ADA 404.2.7", "low")]},
    {"id": "r1_small_turn", "label": NA, "placement": ADJ,
     "causes": [cause("room R1 turning space is too small and a turn is needed inside R1, so R2 cannot be reached",
                      "R1", "turning_space_diameter_mm", "ADA 304.3.1")]},
    {"id": "r1_small_straight", "label": A, "placement": OPP,
     "causes": [cause("room R1 turning space is too small, but R1 can be crossed without turning "
                      "(only a warning)", "R1", "turning_space_diameter_mm")]},
    {"id": "r2_small", "label": A, "placement": ANY,
     "causes": [cause("room R2 turning space is too small (R2 is the last room and can still be entered; "
                      "only a warning)", "R2", "turning_space_diameter_mm")]},
]


# Variety comes from the code, not from the model: every request is a fresh conversation, so the
# model cannot know what it chose before and falls back to the same defaults (prompt v4: 27 of 36
# doors had 1000 mm hardware, R2 was always east of R1). Each prompt therefore fixes the layout,
# approximate room sizes, the hardware type and the value that fails (half of them just past the
# limit), and often one compliant value close to its limit, so the agents see both sides of every
# threshold and every door orientation.
WALL_NAMES = {"S": "south", "N": "north", "W": "west", "E": "east"}
OPPOSITE_WALL = {"S": "N", "N": "S", "W": "E", "E": "W"}
HARDWARE_TYPES = ["Lever", "Pull handle", "Push bar"]
DOOR_FIELDS = ["clear_width_mm", "threshold_height_mm", "hardware_height_mm", "maneuvering_pull_mm",
               "maneuvering_push_mm"]
TURNING = "turning_space_diameter_mm"
ROOM_SIDE_MM = (2600, 6000)  # approximate room sizes given in the prompt (validation allows 2500-6000)
NEAR_MM = {"threshold_height_mm": 3}  # "close to the limit"; every other value: 30 mm
FAR_MM = {"clear_width_mm": 165, "threshold_height_mm": 30, "hardware_height_mm": 150,  # failing values stay
          "maneuvering_pull_mm": 400, "maneuvering_push_mm": 400, TURNING: 400}         # plausible (door >= 650)
NEAR_SHARE = 0.5  # share of failing values just past the limit, and of prompts with a near-limit compliant value


def value_limit(field: str, end: str | None, R: dict) -> tuple[float, str]:
    """(limit, sense) from config/ada_rules.yaml; sense "min" = value must be >= limit, "max" = <= limit.
    end: "high" / "low" for the hardware height range."""
    d = R["door"]
    if field == "hardware_height_mm":
        lo, hi = d["hardware_height_mm"]
        return (hi, "max") if end == "high" else (lo, "min")
    return {"clear_width_mm": (d["clear_width_min_mm"], "min"), "threshold_height_mm": (d["threshold_max_mm"], "max"),
            "maneuvering_pull_mm": (d["maneuvering_pull_min_mm"], "min"),
            "maneuvering_push_mm": (d["maneuvering_push_min_mm"], "min"),
            TURNING: (R["room"]["turning_diameter_min_mm"], "min")}[field]


def required_value(element: str, field: str, end: str | None, fails: bool, rng: random.Random, R: dict) -> dict:
    """A value the prompt asks for: failing (just past the limit or further) or compliant close to the limit."""
    limit, sense = value_limit(field, end, R)
    near = NEAR_MM.get(field, 30)
    if fails:
        m = rng.randint(1, near) if rng.random() < NEAR_SHARE else rng.randint(near + 1, FAR_MM[field])
        value = limit - m if sense == "min" else limit + m
    else:
        m = rng.randint(0, near)
        value = limit + m if sense == "min" else limit - m
    return {"element": element, "field": field, "value": value, "fails": fails,
            "limit": f"{'>=' if sense == 'min' else '<='} {limit}"}


def make_layout(placement: str, rng: random.Random) -> dict:
    """Wall of R1 for each door. D2 is on the wall R1 shares with R2 (R2 lies on that side of R1)."""
    d2 = rng.choice("SNWE")
    if placement == OPP:
        d1 = OPPOSITE_WALL[d2]
    elif placement == ADJ:
        d1 = rng.choice([w for w in "SNWE" if w not in (d2, OPPOSITE_WALL[d2])])
    else:
        d1 = rng.choice([w for w in "SNWE" if w != d2])
    return {"D1": d1, "D2": d2}


def make_user_prompt(rng: random.Random, R: dict | None = None, scenarios: list[str] | None = None):
    """scenarios: ids to choose from (default: all, uniformly), e.g. to top up a rare scenario."""
    R = R or load_rules()
    sc = rng.choice([s for s in SCENARIOS if not scenarios or s["id"] in scenarios])
    c = rng.choice(sc["causes"])
    lay = make_layout(sc["placement"], rng)
    sizes = {r: [rng.randrange(ROOM_SIDE_MM[0], ROOM_SIDE_MM[1] + 1, 50) for _ in range(2)] for r in ("R1", "R2")}
    hardware = {d: rng.choice(HARDWARE_TYPES) for d in ("D1", "D2")}
    values = [required_value(c["element"], c["field"], c["value"], True, rng, R)] if c["field"] else []
    if rng.random() < NEAR_SHARE:
        cands = [(e, f) for e in ("D1", "D2") for f in DOOR_FIELDS] + [("R1", TURNING), ("R2", TURNING)]
        e, f = rng.choice([x for x in cands if x != (c["element"], c["field"])])
        values.append(required_value(e, f, rng.choice(["high", "low"]), False, rng, R))
    placement = OPP if lay["D1"] == OPPOSITE_WALL[lay["D2"]] else ADJ
    prompt = USER_TEMPLATE.format(
        n_rooms=2, rooms="R1, R2", connections="EXTERIOR -D1- R1, R1 -D2- R2",
        layout=f"R2 lies to the {WALL_NAMES[lay['D2']]} of R1; they share R1's {WALL_NAMES[lay['D2']]} wall, "
               f"where D2 is. The entrance door D1 is on R1's {WALL_NAMES[lay['D1']]} wall.",
        placement=placement,
        sizes=", ".join(f"{r} about {w} x {l} mm" for r, (w, l) in sizes.items()),
        hardware=", ".join(f"{d} {h}" for d, h in hardware.items()),
        label=sc["label"], cause=c["text"],
        values="; ".join(f"{v['element']} {v['field']} = {v['value']} "
                         f"({'fails the limit' if v['fails'] else 'complies, close to the limit'} {v['limit']})"
                         for v in values) or "none")
    target = {"label": sc["label"], "failing_rules": [{"element": c["element"], "rule": c["rule"]}] if c["rule"] else [],
              "layout": lay, "sizes": sizes, "hardware": hardware, "values": values}
    return sc, c, target, prompt


def _pairs(rules: list) -> set:
    return {(f.get("element"), f.get("rule")) for f in rules or []}


MANEUVER_MIN_MM = 700     # plausibility: a smaller clearance is not a real design value
EXTERIOR_DEPTH_MM = 6000  # free depth assumed outside the entrance door


def side_depth(door: dict, side: str, rooms: dict, R: dict) -> float | None:
    """Free floor depth on one side of a door: the depth of that room perpendicular to the door's wall
    (None if the door is not on a wall of the room)."""
    if side == EXTERIOR:
        return EXTERIOR_DEPTH_MM
    room = rooms[side]
    wall = door_side(door, room, R)
    if wall is None:
        return None
    return room["length_mm"] if wall in ("S", "N") else room["width_mm"]


# ---------------------------------------------------------------- verification
def validate(b: dict, target: dict, rules: dict) -> list[str]:
    """Structure + plausibility + rule-engine checks. Empty list = keep."""
    p = []
    try:
        rooms = {r["id"]: r for r in b["rooms"]}
        for r in b["rooms"]:
            for k in ("x_mm", "y_mm", "width_mm", "length_mm", "turning_space_diameter_mm"):
                float(r[k])
        for d in b["doors"]:
            for k in ("x_mm", "y_mm", "clear_width_mm", "threshold_height_mm", "hardware_height_mm",
                      "maneuvering_pull_mm", "maneuvering_push_mm"):
                float(d[k])
            if len(d["connects"]) != 2 or any(x not in set(rooms) | {EXTERIOR} for x in d["connects"]):
                return [f"door {d['id']} connects unknown elements"]
    except (KeyError, TypeError, ValueError) as e:
        return [f"bad structure: {e!r}"]
    if sum(EXTERIOR in d["connects"] for d in b["doors"]) != 1:
        p.append("not exactly one entrance door")
    if set(rooms) - {x for d in b["doors"] for x in d["connects"]}:
        p.append("a room has no door")
    for r in b["rooms"]:
        if not (2500 <= r["width_mm"] <= 6000 and 2500 <= r["length_mm"] <= 6000):
            p.append(f"room {r['id']} size out of range")
        if r["turning_space_diameter_mm"] > min(r["width_mm"], r["length_mm"]):
            p.append(f"turning space of {r['id']} larger than the room")
    for d in b["doors"]:
        if not 600 <= d["clear_width_mm"] <= 1200 or not 0 <= d["threshold_height_mm"] <= 60:
            p.append(f"implausible value on {d['id']}")
        pull, push = d["maneuvering_pull_mm"], d["maneuvering_push_mm"]
        if min(pull, push) < MANEUVER_MIN_MM:
            p.append(f"implausible maneuvering clearance on {d['id']}")
        depths = [side_depth(d, rid, rooms, rules) for rid in d["connects"]]
        for rid, depth in zip(d["connects"], depths):
            if depth is None:
                p.append(f"door {d['id']} is not on a wall of {rid}")
        # the record does not say which side is the pull side, so either assignment may fit
        if None not in depths and not (pull <= depths[0] and push <= depths[1] or
                                       pull <= depths[1] and push <= depths[0]):
            p.append(f"maneuvering clearance of {d['id']} is deeper than the room on that side")
    lay = target.get("layout")  # prompt v5+: the walls of R1 asked for each door
    if lay:
        doors = {d["id"]: d for d in b["doors"]}
        for did, wall in lay.items():
            if did not in doors or "R1" not in rooms or door_side(doors[did], rooms["R1"], rules) != wall:
                p.append(f"door {did} is not on the {WALL_NAMES[wall]} wall of R1 as asked")
    if p:
        return p
    v = evaluate(b, rules)["verdict"]
    got = {"label": v["label"], "failing": _pairs(v["failing_rules"])}
    if got["label"] != target["label"] or got["failing"] != _pairs(target["failing_rules"]):
        p.append(f"rule engine {got['label']} {sorted(got['failing'])} != target {target['label']} "
                 f"{sorted(_pairs(target['failing_rules']))}")
    iv = b.get("intended_verdict") or {}
    if iv.get("label") != got["label"] or _pairs(iv.get("failing_rules")) != got["failing"]:
        p.append("model's intended_verdict disagrees with the rule engine")
    return p


def split_answer(msg):
    """Reasoning may come in a separate field or inside <think>...</think>."""
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    m = re.search(r"<think>(.*?)</think>", content, re.S)
    if m:
        reasoning = reasoning or m.group(1)
        content = content[m.end():]
    elif "</think>" in content:
        reasoning, content = content.split("</think>", 1)
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    i, j = content.find("{"), content.rfind("}")
    return reasoning.strip(), (content[i:j + 1] if i != -1 and j != -1 else content)


def call_model(client, user_prompt, args, extra, retries=5):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},   # same every time
                          {"role": "user", "content": user_prompt}],      # changes every time
                temperature=args.temperature, max_tokens=args.max_tokens, extra_body=extra or None)
        except Exception as e:  # rate limit, timeout, server error
            wait = 2 ** attempt * 5
            print(f"  API error ({e.__class__.__name__}: {e}); retry in {wait}s")
            time.sleep(wait)
    return None


class FakeMsg:
    def __init__(self, content, reasoning):
        self.content, self.reasoning_content = content, reasoning


def fake_answer(sc, target, rng):
    """Building that follows the prompt (layout, sizes, hardware, required values), with a deliberately
    wrong intended_verdict now and then - for --dry-run and the tests."""
    lay = target["layout"]
    (w1, l1), (w2, l2) = target["sizes"]["R1"], target["sizes"]["R2"]
    w2, l2 = (w2, l1) if lay["D2"] in "EW" else (w1, l2)  # R2 shares one whole wall of R1
    x2, y2 = {"E": (w1, 0), "W": (-w2, 0), "N": (0, l1), "S": (0, -l2)}[lay["D2"]]
    dx, dy = max(0, -x2), max(0, -y2)  # shift so that every coordinate is >= 0
    mid = {"S": (w1 / 2, 0), "N": (w1 / 2, l1), "W": (0, l1 / 2), "E": (w1, l1 / 2)}  # wall centres of R1

    def door(did, a, b_):
        x, y = mid[lay[did]]
        return {"id": did, "connects": [a, b_], "x_mm": x + dx, "y_mm": y + dy,
                "clear_width_mm": rng.randint(840, 1000), "threshold_height_mm": rng.choice([0, 5, 10]),
                "hardware_height_mm": rng.randint(900, 1150), "hardware_type": target["hardware"][did],
                "maneuvering_pull_mm": rng.randint(1530, 1800), "maneuvering_push_mm": rng.randint(1230, 1500)}

    rooms = [{"id": "R1", "x_mm": dx, "y_mm": dy, "width_mm": w1, "length_mm": l1,
              "turning_space_diameter_mm": 1525 + rng.randint(0, 300)},
             {"id": "R2", "x_mm": x2 + dx, "y_mm": y2 + dy, "width_mm": w2, "length_mm": l2,
              "turning_space_diameter_mm": 1525 + rng.randint(0, 300)}]
    doors = [door("D1", EXTERIOR, "R1"), door("D2", "R1", "R2")]
    elements = {e["id"]: e for e in rooms + doors}
    for v in target["values"]:
        elements[v["element"]][v["field"]] = v["value"]
    b = {"rooms": rooms, "doors": doors}
    v = evaluate(b)["verdict"]
    iv = {"label": v["label"], "failing_rules": [{k: f[k] for k in ("element", "rule", "value")} for f in v["failing_rules"]]}
    if rng.random() < 0.15:  # simulate a model mistake
        iv["label"] = A if iv["label"] == NA else NA
    b["intended_verdict"] = iv
    return FakeMsg("```json\n" + json.dumps(b) + "\n```", f"(fake reasoning for scenario {sc['id']})")


def used_seeds(raw_path: Path) -> set:
    """Seeds of earlier runs, read from <role>_raw.jsonl (records without a seed are ignored)."""
    if not raw_path.exists():
        return set()
    with open(raw_path, encoding="utf-8") as f:
        return {json.loads(line).get("seed") for line in f if line.strip()} - {None}


def generate(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    tag = f"{args.role}_dryrun" if args.dry_run else args.role  # fake records never mix with the real ones
    used = used_seeds(OUT_DIR / f"{tag}_raw.jsonl")
    if args.seed in used:
        print(f"WARNING: seed {args.seed} was already used for {tag}: same scenario order as before. "
              f"Next free seed: {max(used) + 1}")
    rng = random.Random(args.seed)
    prov = PROVIDERS[args.provider]
    args.model = "dry-run" if args.dry_run else args.model or prov["model"]
    client = None
    if not args.dry_run:
        from openai import OpenAI
        key = os.environ.get(prov["key"]) if prov["key"] else "EMPTY"
        if not key:
            raise SystemExit(f"Set {prov['key']} first.")
        client = OpenAI(base_url=args.base_url or prov["base_url"], api_key=key)
        if args.provider == "openai":
            print("Note: the OpenAI API does not return the model's reasoning; only answers are saved.")

    raw_f = open(OUT_DIR / f"{tag}_raw.jsonl", "a", encoding="utf-8")
    ok_f = open(OUT_DIR / f"{tag}_verified.jsonl", "a", encoding="utf-8")
    ifc_dir = ROOT / "data" / "synthetic" / tag
    jobs = [make_user_prompt(rng, rules, args.scenario) for _ in range(args.n)]  # all prompts first: same seed, same prompts

    def answers():
        """(message, usage, seconds) per job, in job order; API calls run --workers at a time."""
        if args.dry_run:
            for sc, _, target, _ in jobs:
                yield fake_answer(sc, target, rng), None, 0.0
            return

        def ask(job):
            t0 = time.time()
            return call_model(client, job[3], args, prov["extra"]), time.time() - t0

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for resp, secs in ex.map(ask, jobs):
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
                yield (resp.choices[0].message if resp else None), usage, secs

    kept = 0
    for i, ((sc, c, target, user_prompt), (msg, usage, secs)) in enumerate(zip(jobs, answers())):
        if msg is None:
            print(f"[{i + 1}/{args.n}] gave up after retries")
            continue
        reasoning, json_text = split_answer(msg)
        try:
            building = json.loads(json_text)
            problems = validate(building, target, rules)
        except json.JSONDecodeError as e:
            building, problems = None, [f"invalid JSON: {e}"]
        bid = f"{tag}_{int(time.time() * 1000)}_{i}"
        rec = {"id": bid, "role": args.role, "model": args.model, "seed": args.seed, "scenario": sc["id"],
               "prompt_version": PROMPT_VERSION,
               "user_prompt": user_prompt, "target": target, "reasoning": reasoning, "answer_text": json_text,
               "building": building, "problems": problems, "kept": not problems,
               "seconds": round(secs, 1), "usage": usage}
        if not problems:  # labels are written HERE, in the verify step, and stored with the record
            rec["labels"] = make_labels(building, rules)
        raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not problems:
            ok_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if args.write_ifc:
                from src.ada.ifc_writer import write_ifc
                ifc_dir.mkdir(parents=True, exist_ok=True)
                write_ifc({**building, "building_id": bid}, str(ifc_dir / f"{bid}.ifc"))
        raw_f.flush()
        ok_f.flush()
        print(f"[{i + 1}/{args.n}] {sc['id']:<18} " + ("kept" if not problems else "rejected: " + "; ".join(problems)))
    print(f"\nKept {kept}/{args.n} ({100 * kept / max(args.n, 1):.0f}%). Files in {OUT_DIR}/")


def export_sft(dry_run: bool = False):
    """Verified teacher records -> chat-format training file for the student (Qwen).
    dry_run: use the fake teacher_dryrun_* records and write student_dryrun_sft.jsonl."""
    t, s = ("teacher_dryrun", "student_dryrun") if dry_run else ("teacher", "student")
    src, dst = OUT_DIR / f"{t}_verified.jsonl", OUT_DIR / f"{s}_sft.jsonl"
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            answer = json.dumps(r["building"], ensure_ascii=False)
            if r.get("reasoning"):
                answer = f"<think>\n{r['reasoning']}\n</think>\n" + answer
            g.write(json.dumps({"id": r["id"], "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["user_prompt"]},
                {"role": "assistant", "content": answer}]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} training examples to {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="nemotron")
    ap.add_argument("--model", default=None, help="override the provider's default model name")
    ap.add_argument("--base-url", default=None, help="override the provider's API address")
    ap.add_argument("--n", type=int, default=20, help="number of buildings to request")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="API requests at the same time (default 1)")
    ap.add_argument("--scenario", nargs="+", choices=[s["id"] for s in SCENARIOS], default=None,
                    help="only these scenarios (default: all, uniformly), e.g. to top up a rare one")
    ap.add_argument("--write-ifc", action="store_true", help="write every verified building as an IFC file")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls, fake answers, written to <role>_dryrun_* files (apart from the real data)")
    ap.add_argument("--export-sft", action="store_true", help="build the student training file and exit")
    a = ap.parse_args()
    export_sft(a.dry_run) if a.export_sft else generate(a)
