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
  6. (teacher only) the architecture council reviews the verified buildings:
     scripts/council.py packet / collect -> teacher_approved.jsonl

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ada.rules import ACCESSIBLE, EXTERIOR, NOT_ACCESSIBLE, door_side, evaluate, load_rules, make_labels  # noqa: E402

OUT_DIR = ROOT / "data" / "llm"
SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
USER_TEMPLATE = (ROOT / "prompts" / "user_template.txt").read_text(encoding="utf-8")
PROMPT_VERSION = 4

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
                      "maneuvering_pull_mm", "ADA 404.2.4")]},
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


def make_user_prompt(rng: random.Random):
    sc = rng.choice(SCENARIOS)
    c = rng.choice(sc["causes"])
    prompt = USER_TEMPLATE.format(
        n_rooms=2, rooms="R1, R2", connections="EXTERIOR -D1- R1, R1 -D2- R2", placement=sc["placement"],
        label=sc["label"], cause=c["text"])
    target = {"label": sc["label"], "failing_rules": [{"element": c["element"], "rule": c["rule"]}] if c["rule"] else []}
    return sc, c, target, prompt


def _pairs(rules: list) -> set:
    return {(f.get("element"), f.get("rule")) for f in rules or []}


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
        if not 700 <= d["maneuvering_pull_mm"] <= 3000 or not 700 <= d["maneuvering_push_mm"] <= 3000:
            p.append(f"implausible maneuvering clearance on {d['id']}")
        for rid in d["connects"]:
            if rid != EXTERIOR and door_side(d, rooms[rid], rules) is None:
                p.append(f"door {d['id']} is not on a wall of {rid}")
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


def fake_answer(sc, c, rng):
    """Building that matches the scenario (sometimes deliberately wrong) - for --dry-run."""
    w1, w2, L = rng.randint(4000, 5600), rng.randint(3000, 4200), rng.randint(3800, 5200)
    door = lambda did, a, b: {"id": did, "connects": [a, b], "clear_width_mm": rng.randint(840, 1000),
                              "threshold_height_mm": rng.choice([0, 5, 10]), "hardware_height_mm": rng.randint(900, 1150),
                              "hardware_type": "Lever", "maneuvering_pull_mm": rng.randint(1530, 1800),
                              "maneuvering_push_mm": rng.randint(1230, 1500)}
    d1, d2 = door("D1", EXTERIOR, "R1"), door("D2", "R1", "R2")
    # D2 on the east wall of R1; D1 on the west wall (opposite) or the south wall (adjacent)
    d1.update(x_mm=0, y_mm=L / 2) if sc["placement"] == OPP or (sc["placement"] == ANY and rng.random() < .5) \
        else d1.update(x_mm=w1 / 2, y_mm=0)
    d2.update(x_mm=w1, y_mm=L / 2)
    rooms = [{"id": "R1", "x_mm": 0, "y_mm": 0, "width_mm": w1, "length_mm": L,
              "turning_space_diameter_mm": 1525 + rng.randint(0, 300)},
             {"id": "R2", "x_mm": w1, "y_mm": 0, "width_mm": w2, "length_mm": L,
              "turning_space_diameter_mm": 1525 + rng.randint(0, 300)}]
    if c["field"]:
        el = {"D1": d1, "D2": d2, "R1": rooms[0], "R2": rooms[1]}[c["element"]]
        el[c["field"]] = {"clear_width_mm": rng.randint(690, 790), "threshold_height_mm": rng.randint(20, 35),
                          "hardware_height_mm": rng.randint(1250, 1350) if c["value"] == "high" else rng.randint(700, 840),
                          "maneuvering_pull_mm": rng.randint(1100, 1450), "maneuvering_push_mm": rng.randint(900, 1150),
                          "turning_space_diameter_mm": rng.randint(1100, 1450)}[c["field"]]
    b = {"rooms": rooms, "doors": [d1, d2]}
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
    kept = 0
    for i in range(args.n):
        sc, c, target, user_prompt = make_user_prompt(rng)
        t0 = time.time()
        if args.dry_run:
            msg, usage = fake_answer(sc, c, rng), None
        else:
            resp = call_model(client, user_prompt, args, prov["extra"])
            if resp is None:
                print(f"[{i + 1}/{args.n}] gave up after retries")
                continue
            msg = resp.choices[0].message
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
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
               "seconds": round(time.time() - t0, 1), "usage": usage}
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
    """Council-approved teacher records -> chat-format training file for the student (Qwen).
    Falls back to all verified teacher records if the council step has not been run yet.
    dry_run: use the fake teacher_dryrun_* records and write student_dryrun_sft.jsonl."""
    t, s = ("teacher_dryrun", "student_dryrun") if dry_run else ("teacher", "student")
    src, dst = OUT_DIR / f"{t}_approved.jsonl", OUT_DIR / f"{s}_sft.jsonl"
    if not src.exists():
        src = OUT_DIR / f"{t}_verified.jsonl"
        print(f"WARNING: {t}_approved.jsonl not found (run scripts/council.py collect); using every verified record")
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            answer = json.dumps(r["building"], ensure_ascii=False)
            if r.get("reasoning"):
                answer = f"<think>\n{r['reasoning']}\n</think>\n" + answer
            g.write(json.dumps({"messages": [
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
    ap.add_argument("--write-ifc", action="store_true", help="write every verified building as an IFC file")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls, fake answers, written to <role>_dryrun_* files (apart from the real data)")
    ap.add_argument("--export-sft", action="store_true", help="build the student training file and exit")
    a = ap.parse_args()
    export_sft(a.dry_run) if a.export_sft else generate(a)
