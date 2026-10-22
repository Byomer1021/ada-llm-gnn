"""
The accessibility check of one IFC file:
  IFC -> read (property sets + positions) -> graph -> 3 GNN agents (parallel) -> main agent (LLM) -> report

    python -m src.pipeline some.ifc                         # trained GNN agents, main agent from the config
    python -m src.pipeline some.ifc --provider rule         # no LLM (baseline)
    python -m src.pipeline some.ifc --agents oracle         # before the GNNs are trained (testing only)

The rule engine is NOT part of this check (it is only used in the verify step of generation,
and behind --agents oracle for testing).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ada.report import make_report  # noqa: E402
from src.agents.gnn_agents import load_agents, oracle_findings, run_agents  # noqa: E402
from src.agents.main_agent import decide, load_config  # noqa: E402

_AGENTS = {}


def get_agents(models_dir=None):
    key = str(models_dir or ROOT / load_config()["agents"]["models_dir"])
    if key not in _AGENTS:
        _AGENTS[key] = load_agents(key)
    return _AGENTS[key]


def check_building(building: dict, agents: str = "gnn", provider: str | None = None, cfg: dict | None = None,
                   clients: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if not building["rooms"]:  # "every room can be reached" would hold for no rooms at all
        raise ValueError("no rooms (IfcSpace) on the checked storey: the check needs the file's spaces")
    t0 = time.time()
    findings = oracle_findings(building) if agents == "oracle" else \
        run_agents(building, get_agents(ROOT / cfg["agents"]["models_dir"]),
                   missing=cfg["agents"].get("missing_values", "not_checked"))
    decision = decide(findings, provider=provider, cfg=cfg, clients=clients)
    return {"building": building, "findings": findings, "decision": decision,
            "report": make_report(building, decision, findings), "seconds": round(time.time() - t0, 2)}


def check_ifc(path: str, **kw) -> dict:
    from src.ada.ifc_reader import read_ifc
    b = read_ifc(str(path))
    b.setdefault("building_id", Path(path).stem)
    return check_building(b, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc")
    ap.add_argument("--agents", choices=["gnn", "oracle"], default="gnn")
    ap.add_argument("--provider", choices=["auto", "local", "openai", "rule"], default=None,
                    help="main agent (default: config/pipeline.yaml)")
    ap.add_argument("--out", default=None, help="write the report (.md) and the result (.json) here")
    a = ap.parse_args()
    res = check_ifc(a.ifc, agents=a.agents, provider=a.provider)
    print(res["report"])
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(a.ifc).stem
        (out / f"{stem}_report.md").write_text(res["report"], encoding="utf-8")
        (out / f"{stem}_result.json").write_text(json.dumps({k: v for k, v in res.items() if k != "report"},
                                                            indent=1, ensure_ascii=False, default=str), encoding="utf-8")
