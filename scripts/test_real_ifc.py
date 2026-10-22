"""
Final test on the REAL IFC files (data/real/**/*.ifc, never used for training). Since 22.10.2026:
data/real/ifc-bench/<project>/arc.ifc (huggingface.co/datasets/sylvainHellin/ifc-bench, CC BY 4.0;
each model keeps its own license.txt). Multi-storey files: only the entrance storey (the scope).

    python scripts/test_real_ifc.py                         # gnn agents, main agent from the config
    python scripts/test_real_ifc.py --provider rule --limit 10

For every file: read it, run the check, write the report, and record whether it worked.
Real files have no labels, so as a REFERENCE the rule engine is run on the same values
(agreement = how often the check gives the same verdict). Ground truth for a sample of the
files should be checked by hand (open question).

Output: experiments/<time>_real/ : summary.csv, reports/<file>_report.md, summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ada.ifc_reader import read_ifc  # noqa: E402
from src.ada.rules import evaluate  # noqa: E402
from src.pipeline import check_building  # noqa: E402


def main(a):
    folder = Path(a.folder)
    files = sorted(folder.rglob("*.ifc"))  # also data/real/<collection>/<project>/arc.ifc
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"No .ifc files in {a.folder}")
    out = ROOT / "experiments" / f"{time.strftime('%Y%m%d_%H%M%S')}_real"
    (out / "reports").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, f in enumerate(files, 1):
        name = "_".join(f.relative_to(folder).with_suffix("").parts)  # many files are called arc.ifc
        row = {"file": name, "read_ok": False, "storey": "", "storeys": 0, "rooms": 0, "doors": 0, "issues": 0,
               "verdict": "", "failing_rules": "", "reference_verdict": "", "agree": "", "seconds": "", "error": ""}
        t0 = time.time()
        try:
            b = read_ifc(str(f), storey=a.storey)
            b["building_id"] = name
            row.update(read_ok=True, storey=b["storey"]["name"] or "", storeys=b["storey"]["storeys_in_file"],
                       rooms=len(b["rooms"]), doors=len(b["doors"]), issues=len(b["issues"]))
            res = check_building(b, agents=a.agents, provider=a.provider)
            d = res["decision"]
            ref = evaluate(b)["verdict"]["label"]
            row.update(verdict=d["label"], reference_verdict=ref, agree=int(ref == d["label"]),
                       failing_rules="; ".join(f"{x['element']} {x['rule']}" for x in d["failing_rules"]))
            (out / "reports" / f"{name}_report.md").write_text(res["report"], encoding="utf-8")
        except Exception as e:
            row["error"] = f"{e.__class__.__name__}: {e}"
            (out / "reports" / f"{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"[{i}/{len(files)}] {name}: {row['rooms']} rooms, {row['doors']} doors on '{row['storey']}' "
              f"(of {row['storeys']}): " + (row["verdict"] or "ERROR " + row["error"]))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if r["verdict"]]
    s = {"files": len(rows), "read_ok": sum(r["read_ok"] for r in rows), "report_made": len(done),
         "files_with_data_issues": sum(1 for r in rows if r["issues"]),
         "not_accessible": sum(r["verdict"] == "not_accessible" for r in done),
         "agreement_with_rule_engine": sum(r["agree"] for r in done) / max(len(done), 1),
         "mean_seconds": sum(r["seconds"] for r in rows) / len(rows)}
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(ROOT / "data" / "real"))
    ap.add_argument("--agents", choices=["gnn", "oracle"], default="gnn")
    ap.add_argument("--provider", choices=["auto", "local", "openai", "rule"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--storey", default="entrance",
                    help='"entrance" (the scope is one storey), "all", or a storey name')
    main(ap.parse_args())
