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
MODEL_VERSION = 2  # 2: InputNorm (checkpoints of version 1 have no normalisation statistics)
CONVS = ["gine", "gatv2", "mpnn_max", "sage"]
AGENTS = ["door", "room", "route"]


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class InputNorm(nn.Module):
    """Standardises the measured values (mm) with the TRAINING set's mean / std; missing flags,
    is_exterior and wall one-hots pass unchanged. Without it the network barely separates a
    door just below 815 mm from one just above (0.80 vs 0.83 after mm / 1000), while a
    decision tree on the same features does it easily. Fitted once in train.py; the statistics
    are buffers, so they are saved in the checkpoint and used at check time too."""

    def __init__(self):
        super().__init__()
        for name, dim in (("x", NODE_DIM), ("e", EDGE_DIM), ("d", DOOR_DIM)):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, graphs) -> "InputNorm":
        nv, ev = len(NODE_FEATURES), len(EDGE_FEATURES)

        def stats(t, n, rows=None):  # value j is in column j, its missing flag in column n + j
            mean, std = torch.zeros(t.shape[1]), torch.ones(t.shape[1])
            for j in range(n):
                keep = t[:, n + j] == 0
                if rows is not None:
                    keep &= rows
                if int(keep.sum()) > 1:
                    mean[j], std[j] = t[keep, j].mean(), t[keep, j].std().clamp(min=1e-3)
            return mean, std

        x = torch.cat([g.x for g in graphs])
        self.x_mean, self.x_std = stats(x, nv, rows=x[:, -1] == 0)  # rooms only, not EXTERIOR
        self.e_mean, self.e_std = stats(torch.cat([g.edge_attr for g in graphs]), ev)
        self.d_mean, self.d_std = stats(torch.cat([g.door_attr for g in graphs]), ev)
        return self

    def forward(self, data):
        return ((data.x - self.x_mean) / self.x_std, (data.edge_attr - self.e_mean) / self.e_std,
                (data.door_attr - self.d_mean) / self.d_std)


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

    def __init__(self, conv="gine", hidden=64, layers=2):
        super().__init__()
        self.norm = InputNorm()
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

    def __init__(self, conv="gine", hidden=64, layers=4):
        super().__init__()
        self.norm = InputNorm()
        self.enc = Encoder(conv, hidden, layers)
        self.head = mlp(hidden + NODE_DIM, hidden, 1)

    def forward(self, data):
        x, e, _ = self.norm(data)
        return self.head(torch.cat([self.enc(x, data.edge_index, e), x], -1)).squeeze(-1)


class RoomAgent(_NodeAgent):
    pass


class RouteAgent(_NodeAgent):
    pass


CLASSES = {"door": DoorAgent, "room": RoomAgent, "route": RouteAgent}


def build_agent(agent: str, conv: str = "gine", hidden: int = 64, layers: int | None = None) -> nn.Module:
    layers = layers or (2 if agent == "door" else 4)
    return CLASSES[agent](conv=conv, hidden=hidden, layers=layers)


def targets(agent: str, data):
    """(target, mask) for one agent on a (batched) graph; mask drops exterior / unlabelled entries."""
    if agent == "door":
        y = data.door_y
        return y, y >= 0
    y = data.turning_y if agent == "room" else data.reach_y
    return y, y >= 0
