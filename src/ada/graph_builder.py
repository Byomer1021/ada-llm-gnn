"""
Building dict -> graph (room = node, door = edge), the input of the three GNN agents.

Features are raw values only (mm; scaled in to_pyg, see SCALE). NEVER add answer
fields such as HandicapAccessible / ADA_Compliant / the verdict: they leak the label
(see tests/test_leakage.py). Missing values are 0 plus an is_missing flag.

Targets (only when `labels` is given, i.e. for training / scoring) come from the
labels stored in the verify step (src/ada/rules.py: make_labels). At check time
build_graph() is called without labels and the rule engine is not used.

  door agent  -> per door, 4 outputs: fails ADA 404.2.3 / 404.2.5 / 404.2.7 / 404.2.4
  room agent  -> per room, 1 output: turning space fails (ADA 304.3.1)
  route agent -> per room, 1 output: reachable from the entrance (uses door walls, ADA 403.5.1)
  building    -> accessible = every room reachable (what the main agent decides)

`to_pyg()` converts to a PyTorch Geometric Data object (needs torch + torch_geometric).
"""
from __future__ import annotations

import networkx as nx

from .rules import DOOR_RULES, EXTERIOR, door_side, load_rules

NODE_FEATURES = ["width_mm", "length_mm", "turning_space_diameter_mm"]
EDGE_FEATURES = ["clear_width_mm", "threshold_height_mm", "hardware_height_mm",
                 "maneuvering_pull_mm", "maneuvering_push_mm"]
FEATURE_VERSION = 2  # bump when features / scaling change; old checkpoints are then refused
# scale so every feature is about 0..3 (mm -> m; the threshold, ~0-40 mm, -> cm)
SCALE = {"threshold_height_mm": 10.0}
SIDES = ["S", "N", "W", "E", "?"]  # wall of the room the door is on; "?" = unknown / exterior
FORBIDDEN = {"HandicapAccessible", "ADA_Compliant", "AccessibleRoute", "WheelchairTurningSpace",
             "label", "labels", "intended_labels", "intended_verdict", "verdict", "failing_rules", "target"}


def _feats(obj: dict, keys: list[str]) -> tuple[list[float], list[float]]:
    vals, miss = [], []
    for k in keys:
        v = obj.get(k)
        vals.append(float(v) if v is not None else 0.0)
        miss.append(1.0 if v is None else 0.0)
    return vals, miss


def build_graph(building: dict, labels: dict | None = None, rules: dict | None = None) -> nx.Graph:
    rules = rules or load_rules()
    rooms = {r["id"]: r for r in building["rooms"]}
    g = nx.Graph(building_id=building.get("building_id"),
                 accessible=None if labels is None else int(labels["verdict"]["label"] == "accessible"))
    g.add_node(EXTERIOR, x=[0.0] * len(NODE_FEATURES), missing=[0.0] * len(NODE_FEATURES), is_exterior=1,
               turning_y=-1, reach_y=-1)  # exterior is not scored (-1 = masked)
    for r in building["rooms"]:
        x, m = _feats(r, NODE_FEATURES)
        g.add_node(r["id"], x=x, missing=m, is_exterior=0, global_id=r.get("global_id"),
                   turning_y=-1 if labels is None else labels["room_turning_fails"][r["id"]],
                   reach_y=-1 if labels is None else labels["room_reachable"][r["id"]])
    for d in building["doors"]:
        a, b = d["connects"]
        if a not in g or b not in g:
            continue  # door to an unknown space (reader reports it in issues)
        x, m = _feats(d, EDGE_FEATURES)
        side = {rid: (door_side(d, rooms[rid], rules) or "?") if rid in rooms else "?" for rid in (a, b)}
        g.add_edge(a, b, door_id=d["id"], x=x, missing=m, side=side, global_id=d.get("global_id"),
                   door_y=None if labels is None else [labels["door_rule_fails"][d["id"]][r] for r in DOOR_RULES])
    return g


def _onehot(side: str) -> list[float]:
    return [1.0 if side == s else 0.0 for s in SIDES]


def to_pyg(g: nx.Graph):
    """NetworkX graph -> torch_geometric.data.Data.

    x            [N, 7]   room width, length, turning (scaled) + 3 missing flags + is_exterior
    edge_index   [2, 2E]  both directions of every door
    edge_attr    [2E, 20] door values (scaled) + 5 missing flags + wall of the door in the source room
                          (one-hot S/N/W/E/?) + wall in the target room
    door_index   [2, E]   one column per door (for the door agent's head)
    door_attr    [E, 10]  door values + missing flags
    door_y       [E, 4]   1 = door fails that ADA rule (DOOR_RULES order); only with labels
    turning_y, reach_y [N]  room targets (-1 = exterior / no label)
    building_y   [1]      1 = accessible (-1 = no label)
    """
    import torch
    from torch_geometric.data import Data

    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    nscale = [SCALE.get(k, 1000.0) for k in NODE_FEATURES]
    escale = [SCALE.get(k, 1000.0) for k in EDGE_FEATURES]
    x = torch.tensor([[v / s for v, s in zip(g.nodes[n]["x"], nscale)] + g.nodes[n]["missing"] +
                      [g.nodes[n]["is_exterior"]] for n in nodes], dtype=torch.float)
    ei, ea, di, da, dy, dids = [], [], [], [], [], []
    for a, b, e in g.edges(data=True):
        vals = [v / s for v, s in zip(e["x"], escale)] + e["missing"]
        for u, v in ((a, b), (b, a)):  # undirected -> both directions
            ei.append([idx[u], idx[v]])
            ea.append(vals + _onehot(e["side"][u]) + _onehot(e["side"][v]))
        di.append([idx[a], idx[b]])
        da.append(vals)
        dy.append(e["door_y"] if e["door_y"] is not None else [-1] * len(DOOR_RULES))
        dids.append(e["door_id"])
    acc = g.graph.get("accessible")
    return Data(
        x=x,
        edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous().view(2, -1),
        edge_attr=torch.tensor(ea, dtype=torch.float).view(-1, 2 * len(EDGE_FEATURES) + 2 * len(SIDES)),
        door_index=torch.tensor(di, dtype=torch.long).t().contiguous().view(2, -1),
        door_attr=torch.tensor(da, dtype=torch.float).view(-1, 2 * len(EDGE_FEATURES)),
        door_y=torch.tensor(dy, dtype=torch.float).view(-1, len(DOOR_RULES)),
        turning_y=torch.tensor([g.nodes[n]["turning_y"] for n in nodes], dtype=torch.float),
        reach_y=torch.tensor([g.nodes[n]["reach_y"] for n in nodes], dtype=torch.float),
        building_y=torch.tensor([-1 if acc is None else acc], dtype=torch.float),
        node_names=nodes, door_ids=dids, building_id=g.graph.get("building_id"),
    )
