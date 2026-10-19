"""
End-to-end demo on one 2-room building (no API key, no trained model needed):
  building JSON -> IFC file -> read IFC back -> graph -> 3 agents -> main agent -> report (+ 3D picture)

    python scripts/demo_pipeline.py                       # oracle agents + rule main agent (works anywhere)
    python scripts/demo_pipeline.py --agents gnn          # after training the GNN agents
    python scripts/demo_pipeline.py --provider auto       # local Qwen 7B, then GPT-4.1-mini
Outputs go to data/demo/.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ada.graph_builder import build_graph  # noqa: E402
from src.ada.ifc_reader import read_ifc  # noqa: E402
from src.ada.ifc_writer import write_ifc  # noqa: E402
from src.pipeline import check_building  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--agents", choices=["gnn", "oracle"], default="oracle")
ap.add_argument("--provider", choices=["auto", "local", "openai", "rule"], default="rule")
args = ap.parse_args()

OUT = ROOT / "data" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

# Same example as the slides: D2 is 710 mm wide (ADA needs >= 815 mm)
building = {
    "building_id": "demo_d2_narrow",
    "rooms": [
        {"id": "R1", "x_mm": 0, "y_mm": 0, "width_mm": 5000, "length_mm": 4800, "turning_space_diameter_mm": 1525},
        {"id": "R2", "x_mm": 5000, "y_mm": 0, "width_mm": 3600, "length_mm": 4800, "turning_space_diameter_mm": 1525},
    ],
    "doors": [
        {"id": "D1", "connects": ["EXTERIOR", "R1"], "x_mm": 2500, "y_mm": 0, "clear_width_mm": 915,
         "threshold_height_mm": 0, "hardware_height_mm": 1000, "hardware_type": "Lever",
         "maneuvering_pull_mm": 1550, "maneuvering_push_mm": 1250},
        {"id": "D2", "connects": ["R1", "R2"], "x_mm": 5000, "y_mm": 2400, "clear_width_mm": 710,
         "threshold_height_mm": 0, "hardware_height_mm": 1000, "hardware_type": "Lever",
         "maneuvering_pull_mm": 1550, "maneuvering_push_mm": 1250},
    ],
}

ifc_path = OUT / "demo.ifc"
write_ifc(building, str(ifc_path))
print("1. IFC written:", ifc_path)

b = read_ifc(str(ifc_path))
b["building_id"] = building["building_id"]
print("2. IFC read back:", len(b["rooms"]), "rooms,", len(b["doors"]), "doors, issues:", b["issues"])

g = build_graph(b)
print("3. Graph:", g.number_of_nodes(), "nodes (rooms + EXTERIOR),", g.number_of_edges(), "edges (doors)")

res = check_building(b, agents=args.agents, provider=args.provider)
print(f"4. Agents ({res['findings']['source']}) + main agent ({res['decision']['provider']}):",
      res["decision"]["label"], res["decision"]["failing_rules"])

(OUT / "demo_report.md").write_text(res["report"], encoding="utf-8")
(OUT / "demo_result.json").write_text(json.dumps({k: v for k, v in res.items() if k != "report"}, indent=2,
                                                 ensure_ascii=False, default=str), encoding="utf-8")
print("5. Report:\n")
print(res["report"])

try:
    sys.path.insert(0, str(ROOT / "tools"))
    from render_3d import render
    failing = {f["element"] for f in res["decision"]["failing_rules"]}
    print("6. 3D picture:", render(str(ifc_path), str(OUT / "demo_3d.png"), failing=failing))
except Exception as e:  # matplotlib missing etc.
    print("6. 3D picture skipped:", e)
