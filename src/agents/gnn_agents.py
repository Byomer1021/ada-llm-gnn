"""
Run the three GNN agents on one building and collect their findings for the main agent.

FINDINGS FORMAT (input of the main agent, also shown in the report)
{
  "source": "gnn" | "oracle",
  "doors": [{"id": "D2", "connects": ["R1", "R2"], "values": {...mm...},
             "fail_prob": {"ADA 404.2.3": 0.97, "ADA 404.2.5": 0.01, "ADA 404.2.7": 0.02, "ADA 404.2.4": 0.03}}],
  "rooms": [{"id": "R2", "values": {...mm...}, "turning_fail_prob": 0.02, "reachable_prob": 0.04}],
  "data_issues": ["..."]          # from the IFC reader
}
- door agent  -> doors[].fail_prob
- room agent  -> rooms[].turning_fail_prob
- route agent -> rooms[].reachable_prob

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
from src.ada.rules import DOOR_RULES, EXTERIOR, evaluate, load_rules  # noqa: E402

AGENT_NAMES = ["door", "room", "route"]


def load_agents(models_dir: str | Path = ROOT / "models") -> dict:
    """Load models/door.pt, room.pt, route.pt (written by src.agents.train)."""
    import torch

    from src.agents.models import build_agent

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
        m = build_agent(a, c["conv"], c["hidden"], c["layers"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        agents[a] = m
    return agents


def _values(obj: dict, keys: list[str]) -> dict:
    return {k: obj.get(k) for k in keys}


def run_agents(building: dict, agents: dict) -> dict:
    """Three GNN agents (in parallel, independent of each other) -> findings."""
    import torch

    g = build_graph(building)  # no labels at check time
    data = to_pyg(g)
    with torch.no_grad():
        door_p = torch.sigmoid(agents["door"](data)).tolist() if data.door_index.shape[1] else []
        turn_p = torch.sigmoid(agents["room"](data)).tolist()
        reach_p = torch.sigmoid(agents["route"](data)).tolist()
    doors_by_id = {d["id"]: d for d in building["doors"]}
    rooms_by_id = {r["id"]: r for r in building["rooms"]}
    doors = [{"id": did, "connects": doors_by_id[did]["connects"], "values": _values(doors_by_id[did], EDGE_FEATURES),
              "fail_prob": {r: round(p, 3) for r, p in zip(DOOR_RULES, door_p[i])}}
             for i, did in enumerate(data.door_ids)]
    rooms = [{"id": n, "values": _values(rooms_by_id[n], NODE_FEATURES),
              "turning_fail_prob": round(turn_p[i], 3), "reachable_prob": round(reach_p[i], 3)}
             for i, n in enumerate(data.node_names) if n != EXTERIOR]
    return {"source": "gnn", "doors": doors, "rooms": rooms, "data_issues": building.get("issues", [])}


def oracle_findings(building: dict, rules: dict | None = None) -> dict:
    """Same format from the rule engine. Testing / upper bound only - NOT the method."""
    rules = rules or load_rules()
    res = evaluate(building, rules)
    doors = [{"id": d["id"], "connects": d["connects"], "values": _values(d, EDGE_FEATURES),
              "fail_prob": {r: float(any(f["rule"] == r for f in res["door_failures"][d["id"]])) for r in DOOR_RULES}}
             for d in building["doors"]]
    rooms = [{"id": r["id"], "values": _values(r, NODE_FEATURES),
              "turning_fail_prob": float(bool(res["room_failures"][r["id"]])),
              "reachable_prob": float(res["labels"][r["id"]] == "reachable")} for r in building["rooms"]]
    return {"source": "oracle", "doors": doors, "rooms": rooms, "data_issues": building.get("issues", [])}
