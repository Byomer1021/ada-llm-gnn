"""Rule engine: every 2-room scenario of the project must give the expected labels."""
import copy

import pytest

from src.ada.rules import NOT_REACHABLE as NR, REACHABLE as R, door_side, evaluate, load_rules

RULES = load_rules()


def door(did, a, b, x, y, **kw):
    d = {"id": did, "connects": [a, b], "x_mm": x, "y_mm": y, "clear_width_mm": 915, "threshold_height_mm": 0,
         "hardware_height_mm": 1000, "hardware_type": "Lever", "maneuvering_pull_mm": 1550, "maneuvering_push_mm": 1250}
    d.update(kw)
    return d


def two_rooms(d1_xy=(2500, 0), r1=None, r2=None, d1=None, d2=None):
    """EXTERIOR -D1- R1 -D2- R2 ; R2 east of R1; D2 on the shared (east) wall of R1."""
    b = {"rooms": [{"id": "R1", "x_mm": 0, "y_mm": 0, "width_mm": 5000, "length_mm": 4800, "turning_space_diameter_mm": 1525},
                   {"id": "R2", "x_mm": 5000, "y_mm": 0, "width_mm": 3600, "length_mm": 4800, "turning_space_diameter_mm": 1525}],
         "doors": [door("D1", "EXTERIOR", "R1", *d1_xy), door("D2", "R1", "R2", 5000, 2400)]}
    for obj, upd in ((b["rooms"][0], r1), (b["rooms"][1], r2), (b["doors"][0], d1), (b["doors"][1], d2)):
        obj.update(upd or {})
    return b


SCENARIOS = [
    ("all ok", two_rooms(), {"R1": R, "R2": R}),
    ("D2 narrow", two_rooms(d2={"clear_width_mm": 710}), {"R1": R, "R2": NR}),
    ("D2 boundary 815 passes", two_rooms(d2={"clear_width_mm": 815}), {"R1": R, "R2": R}),
    ("D2 814 fails", two_rooms(d2={"clear_width_mm": 814}), {"R1": R, "R2": NR}),
    ("D2 threshold", two_rooms(d2={"threshold_height_mm": 25}), {"R1": R, "R2": NR}),
    ("D2 hardware high", two_rooms(d2={"hardware_height_mm": 1300}), {"R1": R, "R2": NR}),
    ("D2 maneuvering pull", two_rooms(d2={"maneuvering_pull_mm": 1200}), {"R1": R, "R2": NR}),
    ("D2 maneuvering push", two_rooms(d2={"maneuvering_push_mm": 1000}), {"R1": R, "R2": NR}),
    ("D1 narrow", two_rooms(d1={"clear_width_mm": 700}), {"R1": NR, "R2": NR}),
    ("R1 small, turn needed (D1 south, D2 east)", two_rooms(r1={"turning_space_diameter_mm": 1300}), {"R1": R, "R2": NR}),
    ("R1 small, straight (D1 west, D2 east)", two_rooms(d1_xy=(0, 2400), r1={"turning_space_diameter_mm": 1300}),
     {"R1": R, "R2": R}),
    ("R2 small", two_rooms(r2={"turning_space_diameter_mm": 1300}), {"R1": R, "R2": R}),
    ("D2 value missing -> not checked", two_rooms(d2={"clear_width_mm": None}), {"R1": R, "R2": R}),
]
FAIL_POLICY = copy.deepcopy(RULES)
FAIL_POLICY["missing_values"]["policy"] = "fail"


@pytest.mark.parametrize("name,building,expected", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_scenarios(name, building, expected):
    assert evaluate(building, RULES)["labels"] == expected


def test_small_room_gets_warning_not_label():
    res = evaluate(two_rooms(r2={"turning_space_diameter_mm": 1300}), RULES)
    assert res["labels"]["R2"] == R
    assert any("cannot turn around" in w for w in res["warnings"]["R2"])


def test_reason_names_the_rule():
    res = evaluate(two_rooms(d2={"clear_width_mm": 710}), RULES)
    assert "ADA 404.2.3" in res["reasons"]["R2"][0]


def test_unknown_door_position_is_conservative():
    b = two_rooms(r1={"turning_space_diameter_mm": 1300}, d1_xy=(0, 2400))
    b["doors"][0]["x_mm"] = None  # position unknown
    assert evaluate(b, RULES)["labels"]["R2"] == NR


def test_door_side():
    room = {"x_mm": 0, "y_mm": 0, "width_mm": 5000, "length_mm": 4800}
    assert door_side({"x_mm": 2500, "y_mm": 0}, room, RULES) == "S"
    assert door_side({"x_mm": 5000, "y_mm": 2400}, room, RULES) == "E"
    assert door_side({"x_mm": 0, "y_mm": 2400}, room, RULES) == "W"
    assert door_side({"x_mm": 2500, "y_mm": 2400}, room, RULES) is None  # middle of the room


def test_three_rooms_do_not_break_the_code():
    """Code must not be hard-wired to 2 rooms (stage 2 will have more)."""
    b = two_rooms()
    b["rooms"].append({"id": "R3", "x_mm": 8600, "y_mm": 0, "width_mm": 3000, "length_mm": 4800,
                       "turning_space_diameter_mm": 1525})
    b["doors"].append(door("D3", "R2", "R3", 8600, 2400, clear_width_mm=700))
    assert evaluate(b, RULES)["labels"] == {"R1": R, "R2": R, "R3": NR}
    b2 = copy.deepcopy(b)
    b2["doors"][2]["clear_width_mm"] = 900
    b2["rooms"][1]["turning_space_diameter_mm"] = 1200  # R2 small, D2 west / D3 east -> straight
    assert evaluate(b2, RULES)["labels"] == {"R1": R, "R2": R, "R3": R}


# ---------------------------------------------------------------- building verdict (the output)
A, NA = "accessible", "not_accessible"
VERDICTS = [  # (scenario name, label, failing (element, rule) pairs)
    ("all ok", A, set()),
    ("D2 narrow", NA, {("D2", "ADA 404.2.3")}),
    ("D2 boundary 815 passes", A, set()),
    ("D2 814 fails", NA, {("D2", "ADA 404.2.3")}),
    ("D2 threshold", NA, {("D2", "ADA 404.2.5")}),
    ("D2 hardware high", NA, {("D2", "ADA 404.2.7")}),
    ("D2 maneuvering pull", NA, {("D2", "ADA 404.2.4")}),
    ("D2 maneuvering push", NA, {("D2", "ADA 404.2.4")}),
    ("D1 narrow", NA, {("D1", "ADA 404.2.3")}),
    ("R1 small, turn needed (D1 south, D2 east)", NA, {("R1", "ADA 304.3.1")}),
    ("R1 small, straight (D1 west, D2 east)", A, set()),
    ("R2 small", A, set()),
    ("D2 value missing -> not checked", A, set()),
]
BUILDINGS = {name: b for name, b, _ in SCENARIOS}


def test_missing_value_policies():
    """Default "not_checked": the rule is reported apart, nothing fails. "fail": conservative (E4)."""
    b = BUILDINGS["D2 value missing -> not checked"]
    v = evaluate(b, RULES)["verdict"]
    assert v["label"] == A and v["failing_rules"] == [] and v["warnings"] == []
    assert v["not_checked"] == [{"element": "D2", "rule": "ADA 404.2.3", "field": "clear_width_mm"}]
    res = evaluate(b, FAIL_POLICY)
    assert res["labels"]["R2"] == NR and res["verdict"]["not_checked"] == []
    assert {(f["element"], f["rule"]) for f in res["verdict"]["failing_rules"]} == {("D2", "ADA 404.2.3")}
    small = two_rooms(r1={"turning_space_diameter_mm": None})  # R1 crossed with a turn: unknown -> may pass
    assert evaluate(small, RULES)["labels"]["R2"] == R
    assert evaluate(small, FAIL_POLICY)["labels"]["R2"] == NR


@pytest.mark.parametrize("name,label,failing", VERDICTS, ids=[v[0] for v in VERDICTS])
def test_building_verdict(name, label, failing):
    v = evaluate(BUILDINGS[name], RULES)["verdict"]
    assert v["label"] == label
    assert {(f["element"], f["rule"]) for f in v["failing_rules"]} == failing


def test_non_blocking_failure_is_a_warning():
    v = evaluate(two_rooms(r2={"turning_space_diameter_mm": 1300}), RULES)["verdict"]
    assert v["label"] == A and any("R2" in w for w in v["warnings"])


def test_unreachable_room_without_a_failing_rule_is_not_accessible():
    """Found on real files: no entrance door / a room without doors used to give "accessible"."""
    no_entrance = two_rooms()
    no_entrance["doors"] = no_entrance["doors"][1:]  # only D2 (R1 - R2) is left
    res = evaluate(no_entrance, RULES)
    assert res["verdict"]["label"] == NA and res["verdict"]["failing_rules"] == []
    assert res["verdict"]["unreachable_rooms"] == ["R1", "R2"]
    assert "no entrance door found" in res["reasons"]["R1"][0]
    no_doors = {"rooms": two_rooms()["rooms"], "doors": []}
    assert evaluate(no_doors, RULES)["verdict"]["label"] == NA
    lonely = two_rooms()
    lonely["doors"] = lonely["doors"][:1]  # R2 has no door at all
    v = evaluate(lonely, RULES)["verdict"]
    assert v["label"] == NA and v["unreachable_rooms"] == ["R2"]


def test_warnings_can_be_made_to_count():
    rules = copy.deepcopy(RULES)
    rules["verdict"]["warnings_make_not_accessible"] = True
    v = evaluate(two_rooms(r2={"turning_space_diameter_mm": 1300}), rules)["verdict"]
    assert v["label"] == NA and v["failing_rules"][0]["rule"] == "ADA 304.3.1"


def test_make_labels():
    from src.ada.rules import make_labels
    lab = make_labels(two_rooms(d2={"clear_width_mm": 710}), RULES)
    assert lab["verdict"]["label"] == NA
    assert lab["door_rule_fails"]["D2"]["ADA 404.2.3"] == 1 and sum(lab["door_rule_fails"]["D1"].values()) == 0
    assert lab["room_reachable"] == {"R1": 1, "R2": 0}
    assert lab["room_turning_fails"] == {"R1": 0, "R2": 0}
