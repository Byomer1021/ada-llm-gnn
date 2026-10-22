"""
Training data for a FINE-TUNED main agent (Qwen 7B + LoRA). The untuned model was not enough
(E3, 22.10.2026, 336 test buildings: zero-shot building macro F1 0.57, few-shot 0.57): it saw the
failing values but did not trace whether they block a room.

Input  = exactly what the main agent sees at run time: the system prompt + the three GNN agents'
         findings on the building read back from its IFC file (src.agents.main_agent.messages).
Target = the building's labels (written by the rule engine at verification, step 2) as a main-agent
         answer. The explanation comes FIRST and traces the path (doors, rooms that cannot turn, door
         walls, where an agent is wrong), so the model writes its reasoning before the verdict.
Only TRAIN and VALIDATION buildings of data/datasets/split.json; the test split is never used.

    python -m src.agents.main_agent_sft                  # -> data/llm/main_agent_sft.jsonl
    python -m src.llm.train_student --config config/main_agent_sft.yaml        (GPU, Colab)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ada.rules import DOOR_RULES, EXTERIOR, OPPOSITE, ROOM_RULE, load_rules  # noqa: E402
from src.agents.main_agent import PASSAGE_RULE, messages  # noqa: E402

OUT = ROOT / "data" / "llm" / "main_agent_sft.jsonl"


def _mm(x) -> str:
    return "not in the file" if x is None else f"{x:.0f} mm"


def door_why(d: dict, rule: str, R: dict) -> str:
    """Why a door fails a rule, from its measured values."""
    v, t = d["values"], R["door"]
    if rule == "ADA 404.2.3":
        return f"clear width {_mm(v.get('clear_width_mm'))} < {t['clear_width_min_mm']} mm"
    if rule == "ADA 404.2.5":
        return f"threshold {_mm(v.get('threshold_height_mm'))} > {t['threshold_max_mm']} mm"
    if rule == "ADA 404.2.7":
        lo, hi = t["hardware_height_mm"]
        return f"hardware height {_mm(v.get('hardware_height_mm'))} outside {lo}-{hi} mm"
    pull, push = v.get("maneuvering_pull_mm"), v.get("maneuvering_push_mm")
    why = []
    if pull is not None and pull < t["maneuvering_pull_min_mm"]:
        why.append(f"pull side {_mm(pull)} < {t['maneuvering_pull_min_mm']} mm")
    if push is not None and push < t["maneuvering_push_min_mm"]:
        why.append(f"push side {_mm(push)} < {t['maneuvering_push_min_mm']} mm")
    return ", ".join(why) or "maneuvering clearance too small"


def explain(findings: dict, labels: dict, R: dict) -> str:
    """Path-trace explanation of the labels, in the words the system prompt uses."""
    doors, rooms = findings["doors"], {r["id"]: r for r in findings["rooms"]}
    fails = {d["id"]: [r for r in DOOR_RULES if labels["door_rule_fails"].get(d["id"], {}).get(r)] for d in doors}
    reach = labels["room_reachable"]
    out = []
    ok = [d["id"] for d in doors if not fails[d["id"]]]
    if ok:
        out.append(f"{' and '.join(ok)} pass{'es' if len(ok) == 1 else ''} every door rule.")
    for d in doors:
        if fails[d["id"]]:
            a, b = d["connects"]
            out.append(f"{d['id']} ({a} to {b}) fails " + "; ".join(f"{r} ({door_why(d, r, R)})" for r in fails[d["id"]])
                       + ".")
    for rid, room in rooms.items():
        if not labels["room_turning_fails"].get(rid):
            continue
        s = f"{rid} cannot turn around ({_mm(room['values'].get('turning_space_diameter_mm'))} < " \
            f"{R['room']['turning_diameter_min_mm']} mm, {ROOM_RULE})"
        own = [d for d in doors if rid in d["connects"]]
        if not reach.get(rid):
            out.append(s + ", but it cannot be reached anyway.")
        elif len(own) < 2:
            out.append(s + "; it has a single door and blocks no other room: a warning.")
        elif len(own) == 2:
            (d1, w1), (d2, w2) = [(d["id"], d.get("walls", {}).get(rid, "?")) for d in own]
            behind = [x for d in own for x in d["connects"] if x not in (rid, EXTERIOR)]
            where = f"{d1} on its {w1} wall, {d2} on its {w2} wall"
            minimum = R["passage"]["straight_min_width_mm"]
            if (w1, w2) in OPPOSITE:
                across = room["values"].get("width_mm" if w1 in ("S", "N") else "length_mm")
                if across is not None and across >= minimum:
                    out.append(s + f". {where}: opposite walls, so it is crossed in a straight line "
                               f"({_mm(across)} across >= {minimum} mm, {PASSAGE_RULE}).")
                else:
                    out.append(s + f". {where}: opposite walls, but it is {_mm(across)} across "
                               f"(< {minimum} mm, {PASSAGE_RULE}), so {', '.join(behind)} cannot be reached that way.")
            elif "?" in (w1, w2):
                out.append(s + f". {where}: a wall is unknown, so a turn is assumed and "
                           f"{', '.join(behind)} cannot be reached that way.")
            else:
                out.append(s + f". {where}: adjacent walls, so going on needs a turn and "
                           f"{', '.join(behind)} cannot be reached that way.")
        else:
            out.append(s + ".")
    for d in doors:  # where the agents are wrong, as the system prompt asks
        for r in DOOR_RULES:
            p = d["fail_prob"].get(r)
            if p is not None and (p >= 0.5) != bool(r in fails[d["id"]]):
                out.append(f"The door agent's {p:.2f} for {d['id']} {r} is wrong: the measured value "
                           f"{'fails' if r in fails[d['id']] else 'meets'} the rule.")
    for rid, room in rooms.items():
        p, f = room["turning_fail_prob"], bool(labels["room_turning_fails"].get(rid))
        if p is not None and (p >= 0.5) != f:
            out.append(f"The room agent's {p:.2f} for {rid} is wrong: the turning space "
                       f"{'fails' if f else 'meets'} {ROOM_RULE}.")
        if (room["reachable_prob"] >= 0.5) != bool(reach.get(rid)):
            out.append(f"The route agent disagrees on {rid} ({room['reachable_prob']:.2f}); the traced path is trusted.")
    gone = [rid for rid in rooms if not reach.get(rid)]
    out.append(f"Not reachable from the entrance: {', '.join(gone)}." if gone
               else "Every room can be reached from the entrance.")
    return " ".join(out)


def answer(findings: dict, labels: dict, R: dict) -> dict:
    v = labels["verdict"]
    return {"explanation": explain(findings, labels, R), "label": v["label"],
            "failing_rules": [{"element": f["element"], "rule": f["rule"]} for f in v["failing_rules"]],
            "warnings": list(v.get("warnings", []))}


def main(a):
    from src.agents.dataset import load_building, records
    from src.agents.gnn_agents import run_agents
    from src.agents.main_agent import load_config
    from src.pipeline import get_agents

    R = load_rules()
    split = json.loads((ROOT / "data" / "datasets" / "split.json").read_text(encoding="utf-8"))
    part = {i: s for s in ("train", "val") for i in split[s]}
    assert not set(part) & set(split["test"])
    recs = [(role, r) for role, r in records(a.sources) if r["id"] in part]
    if a.limit:
        recs = recs[:a.limit]
    agents = get_agents(ROOT / load_config()["agents"]["models_dir"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = Counter()
    with open(out, "w", encoding="utf-8") as f:
        for k, (role, r) in enumerate(recs, 1):
            b, _ = load_building(role, r, from_json=False)  # read back from the IFC, as at run time
            findings = run_agents(b, agents)
            ans = answer(findings, r["labels"], R)
            msgs = messages(findings) + [{"role": "assistant", "content": json.dumps(ans, ensure_ascii=False, indent=1)}]
            f.write(json.dumps({"id": r["id"], "split": part[r["id"]], "messages": msgs}, ensure_ascii=False) + "\n")
            n[(part[r["id"]], ans["label"])] += 1
            n["agent_wrong"] += "is wrong" in ans["explanation"] or "disagrees" in ans["explanation"]
            if k % 200 == 0:
                print(f"{k}/{len(recs)}")
    print(f"wrote {sum(v for k, v in n.items() if k != 'agent_wrong')} examples to {out}: {dict(n)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["teacher_verified", "student_verified"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
