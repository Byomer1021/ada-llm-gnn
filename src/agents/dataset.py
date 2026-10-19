"""
Verified / approved LLM buildings -> graph dataset for the three GNN agents.

For every record the IFC file written by generate.py is READ BACK with the IFC reader
(so the agents train on exactly what they will see on a real file), the graph is built,
and the targets are taken from the labels stored in the verify step (record["labels"]).

Split: by IFC file, 70 / 15 / 15, fixed seed (never by node or door).

    python -m src.agents.dataset                                  # teacher_approved + student_verified
    python -m src.agents.dataset --sources teacher_verified       # e.g. before the council step
    python -m src.agents.dataset --sources teacher_dryrun_approved   # fake data of --dry-run (testing only)
Output: data/datasets/graphs.pt, data/datasets/split.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ada.graph_builder import build_graph, to_pyg  # noqa: E402
from src.ada.rules import load_rules  # noqa: E402

LLM = ROOT / "data" / "llm"
OUT = ROOT / "data" / "datasets"


def records(sources: list[str]):
    seen = set()
    for src in sources:
        path = LLM / f"{src}.jsonl"
        if not path.exists():
            print(f"skip {path.name} (not found)")
            continue
        role = src.rsplit("_", 1)[0]  # IFC folder: teacher_approved -> teacher, teacher_dryrun_verified -> teacher_dryrun
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] not in seen and r.get("labels"):
                    seen.add(r["id"])
                    yield role, r


def load_building(role: str, rec: dict, from_json: bool) -> tuple[dict, str]:
    ifc = ROOT / "data" / "synthetic" / role / f"{rec['id']}.ifc"
    if from_json or not ifc.exists():
        return {**rec["building"], "building_id": rec["id"]}, "json"
    from src.ada.ifc_reader import read_ifc
    b = read_ifc(str(ifc))
    b["building_id"] = rec["id"]
    return b, "ifc"


def split_ids(ids: list[str], seed: int, frac=(0.7, 0.15, 0.15)) -> dict:
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n_tr, n_va = round(frac[0] * len(ids)), round(frac[1] * len(ids))
    return {"seed": seed, "train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}


def main(args):
    import torch

    rules = load_rules()
    graphs, skipped, from_ifc = [], [], 0
    for role, rec in records(args.sources):
        b, how = load_building(role, rec, args.from_json)
        lab = rec["labels"]
        if {r["id"] for r in b["rooms"]} != set(lab["room_reachable"]) or \
                {d["id"] for d in b["doors"]} != set(lab["door_rule_fails"]):
            skipped.append(rec["id"])  # IFC read back does not match the record
            continue
        graphs.append(to_pyg(build_graph(b, lab, rules)))
        from_ifc += how == "ifc"
    if not graphs:
        raise SystemExit("No graphs. Run src.llm.generate (with --write-ifc) first.")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, OUT / "graphs.pt")
    split = split_ids([g.building_id for g in graphs], args.seed)
    (OUT / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    acc = sum(int(g.building_y.item() == 1) for g in graphs)
    print(f"{len(graphs)} graphs ({from_ifc} read from IFC, {len(graphs) - from_ifc} from JSON); "
          f"accessible {acc}, not accessible {len(graphs) - acc}; skipped {len(skipped)}")
    print(f"split: train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])} -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["teacher_approved", "student_verified"],
                    help="record files in data/llm/ (without .jsonl)")
    ap.add_argument("--from-json", action="store_true", help="use the JSON building instead of reading the IFC")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
