"""E4 (scripts/missing_data.py): which values are deleted, and that the original building is kept."""
import importlib.util
import random
from pathlib import Path

from src.ada.graph_builder import EDGE_FEATURES
from tests.test_rules import two_rooms

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("missing_data", ROOT / "scripts" / "missing_data.py")
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)


def test_nothing_dropped_at_zero():
    b = two_rooms()
    same, n = md.drop_values(b, 0.0, random.Random(0))
    assert n == 0 and same == b


def test_all_measured_values_dropped_at_one_geometry_kept():
    b = two_rooms()
    gone, n = md.drop_values(b, 1.0, random.Random(0))
    assert n == 2 * len(EDGE_FEATURES) + 2  # 2 doors x 5 values + 2 turning spaces
    assert all(d[k] is None for d in gone["doors"] for k in EDGE_FEATURES)
    assert all(r["turning_space_diameter_mm"] is None for r in gone["rooms"])
    assert [(r["width_mm"], r["length_mm"]) for r in gone["rooms"]] == [(r["width_mm"], r["length_mm"]) for r in b["rooms"]]
    assert b["doors"][0]["clear_width_mm"] is not None  # a copy: the original is untouched


def test_score_counts_missed_and_false_alarms():
    s = md.score(["not_accessible", "accessible", "accessible"], ["accessible", "not_accessible", "accessible"])
    assert s["missed"] == 1 and s["false_alarms"] == 1 and abs(s["accuracy"] - 1 / 3) < 1e-9
