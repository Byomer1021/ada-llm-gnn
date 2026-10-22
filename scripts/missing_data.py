"""
E4 - missing data: how do the if-else rule engine and the system (GNN agents + main agent) cope
when values are missing from the file?

On the TEST split, a fraction of the measured values is deleted at random after the IFC file is read,
as if the value were neither in the file nor estimable: door clear width, threshold, hardware height,
maneuvering pull / push, room turning space (never room width / length: geometry is always in a
file). The truth stays the label of the complete building: a building does not change because its
file is incomplete. Every system sees the same deleted values (paired).

Systems (building verdict):
  rule_not_checked   rule engine (if-else), missing value -> the rule is not checked (project default)
  rule_fail          rule engine, missing value -> the element fails (conservative)
  <set>              GNN agents of an --agent-sets entry + rule main agent (paths); missing -> not checked
  <set>_infer        the same agents, their guess from the rest of the building used for missing values
The rule engine is a BASELINE here (the plan: "rule engine vs system"), never part of the check.

    python scripts/missing_data.py                                   # agents in models/, 0 / 10 / 20 / 30 %
    python scripts/missing_data.py --agent-sets gnn=models gnn_aug=models/missing
Output: experiments/<time>_e4_missing/metrics.json (+ the printed table)
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import f1_score  # noqa: E402

from src.ada.graph_builder import EDGE_FEATURES  # noqa: E402
from src.ada.rules import ACCESSIBLE, NOT_ACCESSIBLE, ROOM_FIELD, evaluate, load_rules  # noqa: E402
from src.agents.dataset import load_building, records  # noqa: E402
from src.agents.gnn_agents import load_agents, run_agents  # noqa: E402
from src.agents.main_agent import load_config, rule_decide  # noqa: E402

LABELS = [ACCESSIBLE, NOT_ACCESSIBLE]


def drop_values(building: dict, frac: float, rng: random.Random) -> tuple[dict, int]:
    """Copy of the building with each measured value deleted (None) with probability frac."""
    b, n = copy.deepcopy(building), 0
    for el, keys in [(d, EDGE_FEATURES) for d in b["doors"]] + [(r, [ROOM_FIELD]) for r in b["rooms"]]:
        for k in keys:
            if el.get(k) is not None and rng.random() < frac:
                el[k], n = None, n + 1
    return b, n


def score(truth: list[str], pred: list[str]) -> dict:
    pairs = list(zip(truth, pred))
    return {"macro_f1": f1_score(truth, pred, average="macro", labels=LABELS, zero_division=0),
            "accuracy": sum(t == p for t, p in pairs) / len(pairs),
            "missed": sum(t == NOT_ACCESSIBLE and p == ACCESSIBLE for t, p in pairs),  # the dangerous error
            "false_alarms": sum(t == ACCESSIBLE and p == NOT_ACCESSIBLE for t, p in pairs)}


def main(a):
    R = load_rules()
    R_fail = copy.deepcopy(R)
    R_fail["missing_values"]["policy"] = "fail"
    cfg = load_config()
    thr = cfg["agents"]["threshold"]
    split = json.loads((ROOT / "data" / "datasets" / "split.json").read_text(encoding="utf-8"))
    test = set(split["test"])
    recs = [(role, r) for role, r in records(a.sources) if r["id"] in test]
    buildings = [load_building(role, r, from_json=False)[0] for role, r in recs]  # read back from the IFC files
    truth = [r["labels"]["verdict"]["label"] for _, r in recs]
    sets = dict(s.split("=", 1) for s in a.agent_sets)
    agents = {name: load_agents(ROOT / d) for name, d in sets.items()}
    print(f"{len(buildings)} test buildings; agent sets {sets}; fractions {a.fracs}; drop seeds {a.seeds}")

    runs = []
    for frac in a.fracs:
        for seed in (a.seeds if frac > 0 else a.seeds[:1]):
            rng = random.Random(f"e4-{seed}-{frac}")
            dropped = [drop_values(b, frac, rng) for b in buildings]
            pred = {"rule_not_checked": [evaluate(b, R)["verdict"]["label"] for b, _ in dropped],
                    "rule_fail": [evaluate(b, R_fail)["verdict"]["label"] for b, _ in dropped]}
            for name, ag in agents.items():
                for mode, key in (("not_checked", name), ("infer", f"{name}_infer")):
                    pred[key] = [rule_decide(run_agents(b, ag, R, missing=mode), thr, "paths")["label"]
                                 for b, _ in dropped]
            n_drop = sum(n for _, n in dropped)
            runs.append({"frac": frac, "seed": seed, "values_dropped": n_drop,
                         "buildings_with_a_drop": sum(n > 0 for _, n in dropped),
                         "systems": {k: score(truth, v) for k, v in pred.items()}})
            print(f"frac {frac} seed {seed}: {n_drop} values dropped in {runs[-1]['buildings_with_a_drop']} buildings; "
                  + ", ".join(f"{k} {v['macro_f1']:.3f}" for k, v in runs[-1]["systems"].items()))

    systems = list(runs[0]["systems"])
    table = {s: {str(f): {m: {"mean": statistics.mean(r["systems"][s][m] for r in runs if r["frac"] == f),
                              "std": statistics.pstdev([r["systems"][s][m] for r in runs if r["frac"] == f])}
                          for m in ("macro_f1", "accuracy", "missed", "false_alarms")}
                 for f in a.fracs} for s in systems}
    print(f"\nbuilding macro F1, mean +- std over drop seeds (missed failing buildings / false alarms), "
          f"{len(buildings)} buildings ({truth.count(NOT_ACCESSIBLE)} not accessible)")
    print(f"{'system':<22}" + "".join(f"{f'{f:.0%} missing':>26}" for f in a.fracs))
    for s in systems:
        cells = [table[s][str(f)] for f in a.fracs]
        print(f"{s:<22}" + "".join(f"{c['macro_f1']['mean']:>12.3f} +-{c['macro_f1']['std']:.3f} "
                                   f"({c['missed']['mean']:>3.0f}/{c['false_alarms']['mean']:>3.0f})" for c in cells))
    out = ROOT / "experiments" / f"{time.strftime('%Y%m%d_%H%M%S')}_e4_missing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"args": vars(a), "n": len(buildings), "table": table, "runs": runs},
                                                 indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="which values are dropped")
    ap.add_argument("--agent-sets", nargs="+", default=["gnn=models"], metavar="NAME=MODELS_DIR")
    ap.add_argument("--sources", nargs="+", default=["teacher_verified", "student_verified"])
    main(ap.parse_args())
