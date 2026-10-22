"""
Run the three GNN agents on one building and collect their findings for the main agent.

FINDINGS FORMAT (input of the main agent, also shown in the report)
{
  "source": "gnn" | "oracle",
  "doors": [{"id": "D2", "connects": ["R1", "R2"], "values": {...mm...},
             "fail_prob": {"ADA 404.2.3": 0.97, "ADA 404.2.5": 0.01, "ADA 404.2.7": None, "ADA 404.2.4": 0.03},
             "walls": {"R1": "E", "R2": "W"},   # wall of each room the door is on (S/N/W/E, "?" unknown)
             "not_checked": ["ADA 404.2.7"], "estimated": {"clear_width_mm": "OverallWidth ..."}}],
  "rooms": [{"id": "R2", "values": {...mm...}, "turning_fail_prob": 0.02, "reachable_prob": 0.04,
             "not_checked": [], "estimated": {}}],
  "data_issues": ["..."]          # from the IFC reader
}
- door agent  -> doors[].fail_prob
- room agent  -> rooms[].turning_fail_prob
- route agent -> rooms[].reachable_prob
Probabilities are rescaled so that each agent's own threshold (chosen on the validation split,
saved in the checkpoint) sits at 0.5: ">= 0.5" means "yes" for every agent and rule.
A rule whose value is not in the file (and could not be estimated) gets probability None and is
listed in not_checked (config/ada_rules.yaml: missing_values.policy "not_checked"). With
missing="infer" (config/pipeline.yaml: agents.missing_values; experiment E4) the agent's own guess
from the rest of the building is kept instead and the rule is listed in "inferred" (that key only
exists in this mode). "estimated": values the IFC reader estimated (config/derive.yaml).

`oracle_findings()` builds the same format from the rule engine (probabilities 0 / 1).
It is ONLY for testing the main agent and the UI before the GNNs are trained, and as an
upper bound in evaluation; it is clearly marked "oracle" everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ada.graph_builder import EDGE_FEATURES, FEATURE_VERSION, NODE_FEATURES, build_graph, to_pyg  # noqa: E402
from src.ada.rules import (DOOR_FIELD_RULE, DOOR_RULES, EXTERIOR, ROOM_FIELD, ROOM_RULE, door_side,  # noqa: E402
                           evaluate, load_rules)

AGENT_NAMES = ["door", "room", "route"]


def _unchecked(el: dict, rules: dict) -> list[str]:
    """Rules of a door / room whose value is missing, when such rules count as "not checked"."""
    if rules["missing_values"]["policy"] == "fail":
        return []
    if "connects" in el:
        return [r for r in DOOR_RULES if any(el.get(k) is None for k, rr in DOOR_FIELD_RULE.items() if rr == r)]
    return [ROOM_RULE] if el.get(ROOM_FIELD) is None else []


def load_agents(models_dir: str | Path = ROOT / "models") -> dict:
    """Load models/door.pt, room.pt, route.pt (written by src.agents.train)."""
    import torch

    from src.agents.models import MODEL_VERSION, build_agent

    models_dir = Path(models_dir)
    missing = [a for a in AGENT_NAMES if not (models_dir / f"{a}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"trained agents missing in {models_dir}: {missing}. "
                                f"Train them (python -m src.agents.train --agent ...) or use the oracle agents.")
    agents = {}
    for a in AGENT_NAMES:
        ck = torch.load(models_dir / f"{a}.pt", weights_only=False)
        c = ck["config"]
        if c.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError(f"{a}.pt was trained with feature version {c.get('feature_version')}, the code uses "
                               f"{FEATURE_VERSION}. Rebuild the dataset and retrain the agent.")
        if c.get("model_version", 1) != MODEL_VERSION:
            raise RuntimeError(f"{a}.pt is model version {c.get('model_version', 1)}, the code uses {MODEL_VERSION}. "
                               f"Retrain the agent (python -m src.agents.train --agent {a}).")
        m = build_agent(a, c["conv"], c["hidden"], c["layers"], c.get("norm", "standard"))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        m.thresholds = ck.get("thresholds", 0.5)  # chosen on the validation split by train.py
        agents[a] = m
    return agents


def rescale(p: float, t: float) -> float:
    """Monotone map that puts an agent's own decision threshold t (from the validation split) at
    0.5, with 0 -> 0 and 1 -> 1, so ">= 0.5 means yes" holds for every agent and rule downstream
    (main agent, LLM prompt, report)."""
    if t == 0.5:
        return p
    return p / (2 * t) if p < t else 0.5 + (p - t) / (2 * (1 - t))


def _values(obj: dict, keys: list[str]) -> dict:
    return {k: obj.get(k) for k in keys}


def _door(d: dict, probs: dict, rules: dict, rooms: dict, infer: bool = False) -> dict:
    """rooms: id -> room dict, for the wall each connected room has the door on (geometry, as in the graph)."""
    nc = _unchecked(d, rules)
    out = {"id": d["id"], "connects": d["connects"], "values": _values(d, EDGE_FEATURES),
           "fail_prob": {r: None if r in nc and not infer else round(p, 3) for r, p in probs.items()},
           "walls": {rid: door_side(d, rooms[rid], rules) or "?" for rid in d["connects"] if rid in rooms},
           "not_checked": [] if infer else nc, "estimated": d.get("estimated", {})}
    return out | ({"inferred": nc} if infer else {})


def _room(r: dict, turn_p: float, reach_p: float, rules: dict, infer: bool = False) -> dict:
    nc = _unchecked(r, rules)
    out = {"id": r["id"], "values": _values(r, NODE_FEATURES),
           "turning_fail_prob": None if nc and not infer else round(turn_p, 3), "reachable_prob": round(reach_p, 3),
           "not_checked": [] if infer else nc, "estimated": r.get("estimated", {})}
    return out | ({"inferred": nc} if infer else {})


def run_agents(building: dict, agents: dict, rules: dict | None = None, missing: str = "not_checked") -> dict:
    """Three GNN agents (in parallel, independent of each other) -> findings.
    missing: "not_checked" (a rule whose value is missing gets None) or "infer" (the agent's guess)."""
    import torch

    rules = rules or load_rules()
    g = build_graph(building)  # no labels at check time
    data = to_pyg(g)
    with torch.no_grad():
        door_p = torch.sigmoid(agents["door"](data)).tolist() if data.door_index.shape[1] else []
        turn_p = torch.sigmoid(agents["room"](data)).tolist()
        reach_p = torch.sigmoid(agents["route"](data)).tolist()
    thr = {a: getattr(agents[a], "thresholds", 0.5) for a in AGENT_NAMES}
    door_t = thr["door"] if isinstance(thr["door"], list) else [thr["door"]] * len(DOOR_RULES)
    door_p = [[rescale(p, t) for p, t in zip(row, door_t)] for row in door_p]
    turn_p = [rescale(p, thr["room"]) for p in turn_p]
    reach_p = [rescale(p, thr["route"]) for p in reach_p]
    doors_by_id = {d["id"]: d for d in building["doors"]}
    rooms_by_id = {r["id"]: r for r in building["rooms"]}
    infer = missing == "infer"
    doors = [_door(doors_by_id[did], dict(zip(DOOR_RULES, door_p[i])), rules, rooms_by_id, infer)
             for i, did in enumerate(data.door_ids)]
    rooms = [_room(rooms_by_id[n], turn_p[i], reach_p[i], rules, infer)
             for i, n in enumerate(data.node_names) if n != EXTERIOR]
    return {"source": "gnn", "doors": doors, "rooms": rooms, "data_issues": building.get("issues", [])}


def oracle_findings(building: dict, rules: dict | None = None) -> dict:
    """Same format from the rule engine. Testing / upper bound only - NOT the method."""
    rules = rules or load_rules()
    res = evaluate(building, rules)
    rooms_by_id = {r["id"]: r for r in building["rooms"]}
    doors = [_door(d, {r: float(any(f["rule"] == r for f in res["door_failures"][d["id"]])) for r in DOOR_RULES},
                   rules, rooms_by_id) for d in building["doors"]]
    rooms = [_room(r, float(bool(res["room_failures"][r["id"]])), float(res["labels"][r["id"]] == "reachable"), rules)
             for r in building["rooms"]]
    return {"source": "oracle", "doors": doors, "rooms": rooms, "data_issues": building.get("issues", [])}
