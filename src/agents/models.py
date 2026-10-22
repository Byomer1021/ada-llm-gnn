"""
The three GNN agents. They run in parallel on the same room-door graph; each one
checks its own ADA rules. Their outputs go to the main agent (main_agent.py).

  DoorAgent  : per door, 4 logits (fails ADA 404.2.3 / 404.2.5 / 404.2.7 / 404.2.4)
  RoomAgent  : per room, 1 logit  (turning space fails, ADA 304.3.1)
  RouteAgent : per room, 1 logit  (reachable from the entrance; door walls -> turn needed, ADA 403.5.1)

Message-passing layer (`conv`), chosen per experiment:
  gine     GINEConv, uses the door features on every message (default)
  gatv2    GATv2Conv with edge features
  mpnn_max own layer with MAX aggregation: "is there one bad door on the way?" is not
           diluted by averaging over good doors
  sage     GraphSAGE, ignores door features while passing messages (baseline)
Layers >= longest path from the entrance (stage 1: 2 rooms -> default 4 layers).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GINEConv, MessagePassing, SAGEConv

from src.ada.graph_builder import EDGE_FEATURES, NODE_FEATURES, SIDES
from src.ada.rules import DOOR_RULES

NODE_DIM = 2 * len(NODE_FEATURES) + 1
EDGE_DIM = 2 * len(EDGE_FEATURES) + 2 * len(SIDES)
DOOR_DIM = 2 * len(EDGE_FEATURES)
MODEL_VERSION = 3  # 2: InputNorm mean / std, 3: + clamp to the training range (older checkpoints are refused)
CONVS = ["gine", "gatv2", "mpnn_max", "sage"]
ROUTE_ONLY = ["transit"]  # route agent on the transit graph (TransitRouteAgent)
AGENTS = ["door", "room", "route"]


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class InputNorm(nn.Module):
    """Standardises the measured values (mm) with the TRAINING split's mean / std, after clamping
    them to the training range; missing flags, is_exterior and wall one-hots pass unchanged.

    Tried on the 231 teacher buildings (20-21.10.2026, 3 seeds, tiny test split, so read with
    care): raw mm / 1000 -> door 0.51, room 0.62, route 0.74 (F1 of the failing class);
    mean / std -> 0.68 / 0.67 / 0.84; a quantile (rank) transform -> 0.61 / 0.77 / 0.78.
    The clamp keeps the one real advantage of the ranks: values outside the training range
    (real files: a 7110 mm clearance) are not extrapolated by the network.
    Fitted once in train.py; the statistics are buffers, saved in the checkpoint.
    A missing value (0 + its flag) becomes 0 after scaling, i.e. the training mean, and the flag
    says it is missing. (Until 22.10.2026 it was clamped to the training MINIMUM, so a missing door
    width looked like the narrowest door; the synthetic data never has missing values, so no
    result before E4 changed. Same weights, so no MODEL_VERSION bump.)"""

    def __init__(self):
        super().__init__()
        for name, dim in (("x", NODE_DIM), ("e", EDGE_DIM), ("d", DOOR_DIM)):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))
            self.register_buffer(f"{name}_lo", torch.full((dim,), -1e9))
            self.register_buffer(f"{name}_hi", torch.full((dim,), 1e9))

    @torch.no_grad()
    def fit(self, graphs) -> "InputNorm":
        def stats(name, t, n, rows=None):  # value j is in column j, its missing flag in column n + j
            mean, std, lo, hi = (getattr(self, f"{name}_{s}").clone() for s in ("mean", "std", "lo", "hi"))
            for j in range(n):
                keep = t[:, n + j] == 0
                if rows is not None:
                    keep &= rows
                if int(keep.sum()) > 1:
                    v = t[keep, j]
                    mean[j], std[j], lo[j], hi[j] = v.mean(), v.std().clamp(min=1e-3), v.min(), v.max()
            for s, val in zip(("mean", "std", "lo", "hi"), (mean, std, lo, hi)):
                setattr(self, f"{name}_{s}", val)

        nv, ev = len(NODE_FEATURES), len(EDGE_FEATURES)
        x = torch.cat([g.x for g in graphs])
        stats("x", x, nv, rows=x[:, -1] == 0)  # rooms only, not EXTERIOR
        stats("e", torch.cat([g.edge_attr for g in graphs]), ev)
        stats("d", torch.cat([g.door_attr for g in graphs]), ev)
        return self

    def _scale(self, t, name, n):
        mean, std, lo, hi = (getattr(self, f"{name}_{s}") for s in ("mean", "std", "lo", "hi"))
        out = (torch.minimum(torch.maximum(t, lo), hi) - mean) / std
        return torch.cat([out[:, :n] * (1 - t[:, n:2 * n]), out[:, n:]], 1)  # missing value -> 0

    def forward(self, data):
        nv, ev = len(NODE_FEATURES), len(EDGE_FEATURES)
        return self._scale(data.x, "x", nv), self._scale(data.edge_attr, "e", ev), self._scale(data.door_attr, "d", ev)


class QuantileNorm(nn.Module):
    """Option `--norm quantile`: each measured value becomes its rank in the TRAINING split, mapped
    to [-1, 1] (piecewise linear between 65 quantiles; outside the training range: -1 / 1); flags and
    one-hots pass unchanged, a missing value becomes 0. Resolution follows the data: dense near the
    ADA limits (the prompts put many values there), so 1500 vs 1550 mm of maneuvering clearance,
    0.02 standard deviations apart after mean / std (values spread over 700-6000 mm), are many
    ranks apart. Knows nothing about the ADA limits. Same interface as InputNorm."""

    K = 65

    def __init__(self):
        super().__init__()
        for name, n in (("x", len(NODE_FEATURES)), ("e", len(EDGE_FEATURES)), ("d", len(EDGE_FEATURES))):
            self.register_buffer(f"{name}_q", torch.linspace(0, 1, self.K).repeat(n, 1))

    @torch.no_grad()
    def fit(self, graphs) -> "QuantileNorm":
        def quantiles(t, n, rows=None):  # value j is in column j, its missing flag in column n + j
            q = torch.linspace(0, 1, self.K).repeat(n, 1)
            for j in range(n):
                keep = t[:, n + j] == 0
                if rows is not None:
                    keep &= rows
                if int(keep.sum()) > 1:
                    q[j] = torch.quantile(t[keep, j], torch.linspace(0, 1, self.K))
            return q

        nv, ev = len(NODE_FEATURES), len(EDGE_FEATURES)
        x = torch.cat([g.x for g in graphs])
        self.x_q = quantiles(x, nv, rows=x[:, -1] == 0)  # rooms only, not EXTERIOR
        self.e_q = quantiles(torch.cat([g.edge_attr for g in graphs]), ev)
        self.d_q = quantiles(torch.cat([g.door_attr for g in graphs]), ev)
        return self

    def _rank(self, t, q):
        out, n = t.clone(), q.shape[0]
        for j in range(n):
            v, qj = t[:, j].contiguous(), q[j]
            i = torch.searchsorted(qj, v).clamp(1, self.K - 1)
            lo, hi = qj[i - 1], qj[i]
            u = (i - 1 + ((v - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1)) / (self.K - 1)
            u = torch.where(v <= qj[0], torch.zeros_like(u), torch.where(v >= qj[-1], torch.ones_like(u), u))
            out[:, j] = torch.where(t[:, n + j] > 0, torch.zeros_like(u), 2 * u - 1)
        return out

    def forward(self, data):
        return self._rank(data.x, self.x_q), self._rank(data.edge_attr, self.e_q), self._rank(data.door_attr, self.d_q)


NORMS = {"standard": InputNorm, "quantile": QuantileNorm}


class MaxMPNN(MessagePassing):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__(aggr="max")
        self.msg = mlp(2 * dim + edge_dim, dim, dim)
        self.upd = mlp(2 * dim, dim, dim)

    def forward(self, x, edge_index, edge_attr):
        return self.upd(torch.cat([x, self.propagate(edge_index, x=x, edge_attr=edge_attr)], -1))

    def message(self, x_i, x_j, edge_attr):
        return self.msg(torch.cat([x_i, x_j, edge_attr], -1))


class Encoder(nn.Module):
    def __init__(self, conv: str = "gine", hidden: int = 64, layers: int = 4):
        super().__init__()
        assert conv in CONVS, conv
        self.conv_name = conv
        self.inp = nn.Linear(NODE_DIM, hidden)
        self.edge = nn.Linear(EDGE_DIM, hidden)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if conv == "gine":
                self.convs.append(GINEConv(mlp(hidden, hidden, hidden), edge_dim=hidden))
            elif conv == "gatv2":
                self.convs.append(GATv2Conv(hidden, hidden, heads=1, edge_dim=hidden))
            elif conv == "mpnn_max":
                self.convs.append(MaxMPNN(hidden, hidden))
            else:
                self.convs.append(SAGEConv(hidden, hidden))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    def forward(self, x, edge_index, edge_attr):
        h = self.inp(x)
        e = self.edge(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index) if self.conv_name == "sage" else conv(h, edge_index, e)
            h = norm(h + torch.relu(out))  # residual
        return h


class DoorAgent(nn.Module):
    n_out = len(DOOR_RULES)

    def __init__(self, conv="gine", hidden=64, layers=2, norm="standard"):
        super().__init__()
        self.norm = NORMS[norm]()
        self.enc = Encoder(conv, hidden, layers)
        self.head = mlp(2 * hidden + DOOR_DIM, hidden, self.n_out)

    def forward(self, data):
        x, e, d = self.norm(data)
        h = self.enc(x, data.edge_index, e)
        a, b = data.door_index
        # symmetric in the two rooms, so the door's direction does not matter
        return self.head(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), d], -1))


class _NodeAgent(nn.Module):
    n_out = 1

    def __init__(self, conv="gine", hidden=64, layers=4, norm="standard"):
        super().__init__()
        self.norm = NORMS[norm]()
        self.enc = Encoder(conv, hidden, layers)
        self.head = mlp(hidden + NODE_DIM, hidden, 1)

    def forward(self, data):
        x, e, _ = self.norm(data)
        return self.head(torch.cat([self.enc(x, data.edge_index, e), x], -1)).squeeze(-1)


class RoomAgent(_NodeAgent):
    pass


class RouteAgent(_NodeAgent):
    pass


# ---------------------------------------------------------------- route agent on the transit graph
WALL = len(SIDES)
TRANSIT_NODE_DIM = DOOR_DIM + 1               # door values + missing flags, is_entrance
TRANSIT_EDGE_DIM = 2 * WALL + NODE_DIM        # wall in the room of the door we came in by and of the
                                              # door we leave by, + that room's features


def transit(x, edge_index, edge_attr):
    """The room-door graph -> the TRANSIT graph the route agent reasons on.

    A transit node is one direction of a door: "enter room t through door d" (edge u -> t of the
    room-door graph; entering EXTERIOR is left out). A transit edge joins "enter r through d1" to
    "enter t through d2" when d2 is another door of r: crossing r from d1 to d2. Its features carry
    the walls of r that d1 and d2 are on, together with r's features, so "does crossing r need a
    turn?" sits on ONE edge. On the room-door graph the two walls are on two different edges, and
    the route agent did not learn to compare them (22.10.2026: r1_small_straight 28 / 40 wrong).

    Works on batches: to_pyg writes both directions of a door next to each other, so door id =
    edge index // 2 also after batching. Returns (node features, transit edge_index, edge features,
    room each transit node enters)."""
    src, dst = edge_index
    is_ext = x[:, -1] > 0.5
    door = torch.arange(edge_index.shape[1], device=x.device) // 2
    keep = (~is_ext[dst]).nonzero().view(-1)                       # transit nodes = these edges
    tid = torch.full((edge_index.shape[1],), -1, dtype=torch.long, device=x.device)
    tid[keep] = torch.arange(len(keep), device=x.device)
    nodes = torch.cat([edge_attr[keep, :DOOR_DIM], is_ext[src[keep]].float().unsqueeze(1)], 1)
    # pairs (e1 into r, e2 out of r), another door, not leaving to EXTERIOR
    pair = (dst[keep].unsqueeze(1) == src[keep].unsqueeze(0)) & (door[keep].unsqueeze(1) != door[keep].unsqueeze(0))
    a, b = pair.nonzero(as_tuple=True)
    e1, e2 = keep[a], keep[b]
    wall_in, wall_out = slice(DOOR_DIM + WALL, DOOR_DIM + 2 * WALL), slice(DOOR_DIM, DOOR_DIM + WALL)
    feats = torch.cat([edge_attr[e1, wall_in], edge_attr[e2, wall_out], x[dst[e1]]], 1)
    return nodes, torch.stack([tid[e1], tid[e2]]), feats, dst[keep]


class TransitRouteAgent(nn.Module):
    """Route agent on the transit graph: max-aggregation message passing ("reachable if ANY way in is
    reachable"); a room's logit is the max over the transit nodes that enter it. Same input
    (the room-door graph), same output (one logit per node; EXTERIOR is masked) as RouteAgent."""
    n_out = 1

    def __init__(self, conv="transit", hidden=64, layers=4, norm="standard"):
        super().__init__()
        self.norm = NORMS[norm]()
        self.inp = nn.Linear(TRANSIT_NODE_DIM, hidden)
        self.edge = nn.Linear(TRANSIT_EDGE_DIM, hidden)
        self.convs = nn.ModuleList(MaxMPNN(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = mlp(hidden + TRANSIT_NODE_DIM, hidden, 1)

    def forward(self, data):
        x, e, _ = self.norm(data)
        tx, tei, te, into = transit(x, data.edge_index, e)
        h, ee = self.inp(tx), self.edge(te)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + torch.relu(conv(h, tei, ee)))
        logit = self.head(torch.cat([h, tx], -1)).squeeze(-1)
        out = torch.full((x.shape[0],), -10.0, device=x.device)  # no way in -> not reachable
        return out.scatter_reduce(0, into, logit, reduce="amax", include_self=True)


CLASSES = {"door": DoorAgent, "room": RoomAgent, "route": RouteAgent}


def build_agent(agent: str, conv: str = "gine", hidden: int = 64, layers: int | None = None,
                norm: str = "standard") -> nn.Module:
    layers = layers or (2 if agent == "door" else 4)
    if conv in ROUTE_ONLY:
        if agent != "route":
            raise ValueError(f"conv {conv!r} is for the route agent only")
        return TransitRouteAgent(conv=conv, hidden=hidden, layers=layers, norm=norm)
    return CLASSES[agent](conv=conv, hidden=hidden, layers=layers, norm=norm)


def targets(agent: str, data):
    """(target, mask) for one agent on a (batched) graph; mask drops exterior / unlabelled entries."""
    if agent == "door":
        y = data.door_y
        return y, y >= 0
    y = data.turning_y if agent == "room" else data.reach_y
    return y, y >= 0
