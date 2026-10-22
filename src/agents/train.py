"""
Train one GNN agent on data/datasets/ (made by src.agents.dataset).

    python -m src.agents.train --agent door  --conv gine --seeds 0 1 2
    python -m src.agents.train --agent room  --conv gine --seeds 0 1 2
    python -m src.agents.train --agent route --conv mpnn_max --seeds 0 1 2

Main metric of each agent (success criterion: >= 0.85) = F1 of its FAILING class:
  door  -> "door fails" (any of its 4 rules)    (+ F1 per rule)
  room  -> "turning space fails"
  route -> "room not reachable"                 (+ building-level macro F1 if every room is used)
Macro F1 is logged too. Early stopping on validation main F1. Results: mean +- std over seeds.
After training, the decision threshold of every output (door: per rule) is chosen on the
VALIDATION split (calibrate()) and saved with the model; the test split is scored with it, and
at 0.5 for comparison.

Output: experiments/<time>_<agent>_<conv>/metrics.json and models/<agent>.pt (best seed on val),
which src/agents/gnn_agents.py loads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.models import AGENTS, CONVS, MODEL_VERSION, ROUTE_ONLY, build_agent, targets  # noqa: E402
from src.ada.graph_builder import FEATURE_VERSION  # noqa: E402
from src.ada.rules import DOOR_RULES  # noqa: E402

DATA = ROOT / "data" / "datasets"


def load_split():
    graphs = torch.load(DATA / "graphs.pt", weights_only=False)
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    by_id = {g.building_id: g for g in graphs}
    return {k: [by_id[i] for i in split[k] if i in by_id] for k in ("train", "val", "test")}


def f1(y, p, pos=1):
    return float(f1_score(y, p, pos_label=pos, zero_division=0))


def macro(y, p):
    return float(f1_score(y, p, average="macro", labels=[0, 1], zero_division=0))


@torch.no_grad()
def predict(model, agent, graphs):
    """(targets, probabilities) over a split: door [n, 4], room / route [n]."""
    model.eval()
    ys, ps = [], []
    for data in DataLoader(graphs, batch_size=256):
        y, mask = targets(agent, data)
        prob = torch.sigmoid(model(data))
        ys.append(y[mask].view(-1, y.shape[-1]) if agent == "door" else y[mask])
        ps.append(prob[mask].view(-1, y.shape[-1]) if agent == "door" else prob[mask])
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def calibrate(model, agent, graphs):
    """Decision threshold per output that maximises the failing class's F1 on `graphs` (the
    VALIDATION split, never the test split); ties -> the threshold closest to 0.5.
    Why: the loss weights the rare failing class up to 20x (pos_weight), which pushes the
    probabilities up, so 0.5 gives false alarms (22.10.2026: 16 of the 23 wrong building verdicts
    were ADA 404.2.4 called on compliant doors). Door: one threshold per rule."""
    y, prob = predict(model, agent, graphs)

    def best(yc, pc, pos):
        return float(max(THRESHOLD_GRID, key=lambda t: (f1(yc, (pc >= t).astype(int), pos), -abs(t - 0.5))))

    if agent == "door":
        return [best(y[:, j], prob[:, j], 1) for j in range(y.shape[1])]
    return best(y, prob, 0 if agent == "route" else 1)


@torch.no_grad()
def evaluate(model, agent, graphs, thr=0.5):
    """thr: one threshold, or one per door rule (as calibrate() returns)."""
    y, prob = predict(model, agent, graphs)
    p = (prob >= np.asarray(thr)).astype(int)
    if agent == "door":
        anyy, anyp = (y.max(1) > 0).astype(int), p.max(1)
        m = {"main_f1": f1(anyy, anyp), "macro_f1": macro(anyy, anyp)}
        m |= {f"f1_{r}": f1(y[:, i], p[:, i]) for i, r in enumerate(DOOR_RULES)}
    elif agent == "room":
        m = {"main_f1": f1(y, p), "macro_f1": macro(y, p)}
    else:  # route: failing class = not reachable (0)
        m = {"main_f1": f1(y, p, pos=0), "macro_f1": macro(y, p)}
    m["n"] = int(len(y))
    return m


def building_macro_f1(model, graphs, thr=0.5):
    """Route agent alone as a building classifier: accessible = every room predicted reachable."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for g in graphs:
            prob = torch.sigmoid(model(g))[g.reach_y >= 0]
            ys.append(int(g.building_y.item()))
            ps.append(int(bool((prob >= thr).all())))
    return macro(ys, ps)


def pos_weight(agent, graphs):
    y = torch.cat([targets(agent, g)[0][targets(agent, g)[1]].view(-1, 4 if agent == "door" else 1) for g in graphs])
    pos = y.sum(0)
    return ((len(y) - pos) / pos.clamp(min=1)).clamp(max=20)


def train_one(agent, conv, seed, sets, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_agent(agent, conv, args.hidden, args.layers, args.norm)
    model.norm.fit(sets["train"])  # statistics of the training split only
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pw = pos_weight(agent, sets["train"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw if agent == "door" else pw.squeeze())
    best, best_state, bad = -1.0, None, 0
    loader = DataLoader(sets["train"], batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for epoch in range(args.epochs):
        model.train()
        for data in loader:
            y, mask = targets(agent, data)
            out = model(data)
            loss = loss_fn(out[mask].view(-1, 4) if agent == "door" else out[mask],
                           y[mask].view(-1, 4) if agent == "door" else y[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = evaluate(model, agent, sets["val"])["main_f1"] if sets["val"] else 0.0
        if val > best:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    thr = calibrate(model, agent, sets["val"]) if sets["val"] else 0.5
    res = {"seed": seed, "epochs": epoch + 1, "val_main_f1": best, "thresholds": thr,
           "test": evaluate(model, agent, sets["test"], thr), "test_at_0.5": evaluate(model, agent, sets["test"])}
    if agent == "route":
        res["test"]["building_macro_f1"] = building_macro_f1(model, sets["test"], thr)
        res["test_at_0.5"]["building_macro_f1"] = building_macro_f1(model, sets["test"])
    return model, res


def main(args):
    sets = load_split()
    print(f"{args.agent} agent, {args.conv}: train {len(sets['train'])} / val {len(sets['val'])} / "
          f"test {len(sets['test'])} buildings")
    runs, best = [], None
    for seed in args.seeds:
        model, res = train_one(args.agent, args.conv, seed, sets, args)
        runs.append(res)
        print(f"  seed {seed}: epochs {res['epochs']}, val F1 {res['val_main_f1']:.3f}, thresholds "
              f"{res['thresholds']}, test {res['test']}")
        if best is None or res["val_main_f1"] > best[1]["val_main_f1"]:
            best = (model, res)
    keys = [k for k in runs[0]["test"] if k != "n"]
    summary, summary05 = ({k: {"mean": float(np.mean([r[part][k] for r in runs])),
                               "std": float(np.std([r[part][k] for r in runs]))} for k in keys}
                          for part in ("test", "test_at_0.5"))
    print("  test (mean +- std): " + ", ".join(f"{k} {v['mean']:.3f}+-{v['std']:.3f}" for k, v in summary.items()))
    print("  test at 0.5:        " + ", ".join(f"{k} {v['mean']:.3f}+-{v['std']:.3f}" for k, v in summary05.items()))
    cfg = {"agent": args.agent, "conv": args.conv, "hidden": args.hidden, "feature_version": FEATURE_VERSION,
           "model_version": MODEL_VERSION, "layers": args.layers or (2 if args.agent == "door" else 4),
           "norm": args.norm}
    out = ROOT / "experiments" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.agent}_{args.conv}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"config": cfg | vars(args), "runs": runs, "summary": summary,
                                                  "summary_at_0.5": summary05}, indent=1, default=str),
                                      encoding="utf-8")
    models_dir = ROOT / args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": best[0].state_dict(), "seed": best[1]["seed"],
                "val_main_f1": best[1]["val_main_f1"], "thresholds": best[1]["thresholds"]},
               models_dir / f"{args.agent}.pt")
    print(f"  saved {models_dir / (args.agent + '.pt')} and {out / 'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=AGENTS, required=True)
    ap.add_argument("--conv", choices=CONVS + ROUTE_ONLY, default="gine",
                    help="message-passing layer; transit = route agent on the transit graph")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=None, help="default: door 2, room/route 4")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--norm", choices=["standard", "quantile"], default="standard",
                    help="input scaling: mean / std clamped to the training range, or training-split ranks")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models-dir", default="models")
    main(ap.parse_args())
