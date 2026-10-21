"""Real files lack ADA properties: estimates from standard IFC data / geometry, the rest "not checked"."""
import copy

from src.ada.ifc_reader import estimate_clear_width, estimate_missing, load_derive
from src.ada.report import make_report
from src.ada.rules import evaluate
from src.agents.gnn_agents import oracle_findings
from src.agents.main_agent import decide
from tests.test_rules import RULES, two_rooms

D = load_derive()
SWING = {"overall_width_mm": 1000, "lining_thickness_mm": 50, "panel_share": 1.0, "panel_depth_mm": 40,
         "operation": "SINGLE_SWING_RIGHT", "panel_operations": ["SWINGING"]}


def test_clear_width_formula_on_the_door_types_seen_in_real_files():
    # values as in the KIT ArchiCAD files (checked 20.10.2026)
    assert estimate_clear_width(SWING, D)[0] == 860                                   # (1000 - 100) * 1 - 40
    assert estimate_clear_width({**SWING, "overall_width_mm": 885}, D)[0] == 745      # house door: fails 815
    double = {**SWING, "overall_width_mm": 1800, "panel_share": 0.5, "operation": "DOUBLE_DOOR_SINGLE_SWING"}
    assert estimate_clear_width(double, D)[0] == 810                                  # one leaf must give 815
    sliding = {"overall_width_mm": 2010, "lining_thickness_mm": 50, "operation": "DOUBLE_DOOR_SLIDING",
               "panel_operations": ["SLIDING"]}
    assert estimate_clear_width(sliding, D)[0] == 955                                 # no leaf in the way
    bare = {"overall_width_mm": 1010, "operation": "NOTDEFINED"}                      # config defaults
    assert estimate_clear_width(bare, D)[0] == 1010 - 2 * D["door"]["lining_thickness_mm"] - D["door"]["leaf_thickness_mm"]
    assert estimate_clear_width({}, D)[0] is None


def _real_like():
    """two_rooms() as a real export would give it: geometry and OverallWidth, no ADA properties."""
    b = two_rooms()
    for d in b["doors"]:
        for k in ("clear_width_mm", "threshold_height_mm", "hardware_height_mm", "maneuvering_pull_mm",
                  "maneuvering_push_mm"):
            d[k] = None
        d["ifc"] = dict(SWING)
    for r in b["rooms"]:
        r["turning_space_diameter_mm"] = None
    return b


def test_estimates_fill_what_geometry_allows():
    b = estimate_missing(_real_like(), RULES, D)
    d1, d2 = b["doors"]
    r1, r2 = b["rooms"]
    assert d1["clear_width_mm"] == d2["clear_width_mm"] == 860
    # D1 on the south wall of R1 (depth 4800) / outside (6000); D2: R1 east wall (5000), R2 west wall (3600)
    assert (d1["maneuvering_pull_mm"], d1["maneuvering_push_mm"]) == (4800, 6000)
    assert (d2["maneuvering_pull_mm"], d2["maneuvering_push_mm"]) == (3600, 5000)  # pull = shallower side
    assert r1["turning_space_diameter_mm"] == 4800 and r2["turning_space_diameter_mm"] == 3600
    assert d1["hardware_height_mm"] is None and d1["threshold_height_mm"] is None  # no standard field
    assert set(d2["estimated"]) == {"clear_width_mm", "maneuvering_pull_mm", "maneuvering_push_mm"}
    assert any("D2: clear_width_mm estimated 860 mm" in i for i in b["issues"])


def test_a_property_always_wins():
    b = _real_like()
    b["doors"][1]["clear_width_mm"] = 700
    estimate_missing(b, RULES, D)
    assert b["doors"][1]["clear_width_mm"] == 700 and "clear_width_mm" not in b["doors"][1]["estimated"]


def test_estimates_can_be_switched_off():
    b = estimate_missing(_real_like(), RULES, {**D, "enabled": False})
    assert b["doors"][0]["clear_width_mm"] is None


def test_not_checked_flows_from_findings_to_decision_and_report():
    b = estimate_missing(_real_like(), RULES, D)
    v = evaluate(b, RULES)["verdict"]
    assert v["label"] == "accessible" and v["failing_rules"] == []
    assert {(n["element"], n["rule"]) for n in v["not_checked"]} == {
        ("D1", "ADA 404.2.5"), ("D1", "ADA 404.2.7"), ("D2", "ADA 404.2.5"), ("D2", "ADA 404.2.7")}
    f = oracle_findings(b, RULES)
    assert f["doors"][0]["fail_prob"]["ADA 404.2.7"] is None and f["doors"][0]["fail_prob"]["ADA 404.2.3"] == 0.0
    assert f["doors"][0]["not_checked"] == ["ADA 404.2.5", "ADA 404.2.7"]
    dec = decide(f, provider="rule")
    assert dec["label"] == "accessible" and len(dec["not_checked"]) == 4
    rep = make_report(b, dec, f)
    assert "## Not checked (value not in the IFC file)" in rep and "ADA 404.2.7" in rep
    assert "## Estimated values" in rep and "pull side unknown" in rep


def test_estimated_failure_is_reported_as_estimated():
    b = _real_like()
    b["doors"][1]["ifc"] = {**SWING, "overall_width_mm": 885}  # D2 -> 745 mm, fails ADA 404.2.3
    estimate_missing(b, RULES, D)
    f = oracle_findings(b, RULES)
    dec = decide(f, provider="rule")
    assert dec["label"] == "not_accessible" and dec["failing_rules"] == [{"element": "D2", "rule": "ADA 404.2.3"}]
    assert "clear_width 745 (estimated)" in make_report(b, dec, f)


def test_fail_policy_still_marks_missing_values_as_failing():
    rules = copy.deepcopy(RULES)
    rules["missing_values"]["policy"] = "fail"
    b = estimate_missing(_real_like(), rules, D)
    f = oracle_findings(b, rules)
    assert f["doors"][0]["not_checked"] == [] and f["doors"][0]["fail_prob"]["ADA 404.2.7"] == 1.0
    assert decide(f, provider="rule")["label"] == "not_accessible"
