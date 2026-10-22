"""IFC writer -> IFC reader must give back the same values and the same labels."""
import json
import random

import pytest

from src.ada.ifc_reader import read_ifc
from src.ada.ifc_writer import write_ifc
from src.ada.rules import door_side, evaluate, load_rules
from src.llm import generate as gen
from tests.test_rules import SCENARIOS

RULES = load_rules()


@pytest.mark.parametrize("schema", ["IFC4", "IFC2X3"])
@pytest.mark.parametrize("name,building,expected", [s for s in SCENARIOS if "missing" not in s[0]],
                         ids=[s[0] for s in SCENARIOS if "missing" not in s[0]])
def test_roundtrip_labels(tmp_path, schema, name, building, expected):
    p = tmp_path / "b.ifc"
    write_ifc(building, str(p), schema=schema)
    back = read_ifc(str(p))
    assert back["issues"] == []
    assert evaluate(back, RULES)["labels"] == expected


@pytest.mark.parametrize("d2_wall", ["S", "N", "W", "E"])
def test_roundtrip_every_orientation(tmp_path, d2_wall):
    """R2 on each side of R1: the reader gives back the same door walls (route agent features) and labels.
    R1 cannot turn around and D1 / D2 are on adjacent walls, so R2 must come back not reachable."""
    d1_wall = next(w for w in "SNWE" if w not in (d2_wall, gen.OPPOSITE_WALL[d2_wall]))
    target = {"layout": {"D1": d1_wall, "D2": d2_wall}, "sizes": {"R1": [4200, 3600], "R2": [3100, 4500]},
              "hardware": {"D1": "Lever", "D2": "Push bar"},
              "values": [{"element": "R1", "field": "turning_space_diameter_mm", "value": 1400, "fails": True}]}
    b = json.loads(gen.split_answer(gen.fake_answer(gen.SCENARIOS[0], target, random.Random(0)))[1])
    p = tmp_path / "b.ifc"
    write_ifc(b, str(p))
    back = read_ifc(str(p))
    assert back["issues"] == []
    for bb in (b, back):
        r1 = next(r for r in bb["rooms"] if r["id"] == "R1")
        walls = {d["id"]: door_side(d, r1, RULES) for d in bb["doors"]}
        assert walls == {"D1": d1_wall, "D2": d2_wall}
    assert evaluate(back, RULES)["labels"] == evaluate(b, RULES)["labels"]
    assert evaluate(back, RULES)["labels"]["R2"] == "not_reachable"


def test_only_the_entrance_storey_is_read(tmp_path):
    """Scope is one storey: a second storey's rooms must not land in the same plan."""
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.root
    p = tmp_path / "b.ifc"
    write_ifc(SCENARIOS[0][1], str(p))
    m = ifcopenshell.open(str(p))
    ground = m.by_type("IfcBuildingStorey")[0]
    upper = ifcopenshell.api.root.create_entity(m, ifc_class="IfcBuildingStorey", name="First floor")
    upper.Elevation = 3.0
    ifcopenshell.api.aggregate.assign_object(m, relating_object=m.by_type("IfcBuilding")[0], products=[upper])
    ifcopenshell.api.aggregate.assign_object(m, relating_object=upper, products=[
        ifcopenshell.api.root.create_entity(m, ifc_class="IfcSpace", name="R9")])
    m.write(str(p))
    back = read_ifc(str(p))
    assert sorted(r["id"] for r in back["rooms"]) == ["R1", "R2"] and len(back["doors"]) == 2
    assert back["storey"]["name"] == ground.Name and back["storey"]["how"] == "lowest storey with an external door"
    assert any("2 storeys" in i for i in back["issues"])
    assert "R9" in [r["id"] for r in read_ifc(str(p), storey="all")["rooms"]]
    assert [r["id"] for r in read_ifc(str(p), storey="First floor")["rooms"]] == ["R9"]
    assert evaluate(back, RULES)["verdict"]["label"] == "accessible"  # "all ok", unchanged by the upper storey


def test_repeated_names_stay_separate(tmp_path):
    """Real files repeat Names (ArchiCAD: 14 doors called 'Tür-005'); no door may replace another."""
    import ifcopenshell
    p = tmp_path / "b.ifc"
    write_ifc(SCENARIOS[1][1], str(p))  # D2 narrow: not accessible because of D2
    m = ifcopenshell.open(str(p))
    for d in m.by_type("IfcDoor"):
        d.Name = "Tuer-005"
    m.write(str(p))
    back = read_ifc(str(p))
    assert sorted(d["id"] for d in back["doors"]) == ["Tuer-005#1", "Tuer-005#2"]
    v = evaluate(back, RULES)["verdict"]
    assert v["label"] == "not_accessible" and [f["rule"] for f in v["failing_rules"]] == ["ADA 404.2.3"]


def test_values_read_exactly(tmp_path):
    b = SCENARIOS[1][1]  # D2 narrow
    p = tmp_path / "b.ifc"
    write_ifc(b, str(p))
    back = read_ifc(str(p))
    d2 = next(d for d in back["doors"] if d["id"] == "D2")
    assert d2["clear_width_mm"] == 710
    assert d2["connects"] == ["R1", "R2"]
    r1 = next(r for r in back["rooms"] if r["id"] == "R1")
    assert round(r1["width_mm"]) == 5000 and round(r1["length_mm"]) == 4800


def test_missing_value_estimated_or_not_checked(tmp_path):
    """D2 has no ClearWidth_mm property: the reader estimates it from IfcDoor.OverallWidth; without
    estimates it is reported as not in the file and the rule is not checked (or fails, policy "fail")."""
    from src.ada.ifc_reader import estimate_clear_width, load_derive
    from tests.test_rules import FAIL_POLICY
    b = [s for s in SCENARIOS if "missing" in s[0]][0][1]
    p = tmp_path / "b.ifc"
    write_ifc(b, str(p))
    back = read_ifc(str(p))
    d2 = next(d for d in back["doors"] if d["id"] == "D2")
    assert "clear_width_mm" in d2["estimated"]
    assert d2["clear_width_mm"] == estimate_clear_width(d2["ifc"], load_derive())[0]
    assert 800 < d2["clear_width_mm"] < 900  # writer: OverallWidth = 900 mm default + 90 mm frame
    raw = read_ifc(str(p), derive=False)
    assert any("D2: ClearWidth_mm not in the file" in i for i in raw["issues"])
    assert evaluate(raw, RULES)["verdict"]["not_checked"][0]["element"] == "D2"
    assert evaluate(raw, FAIL_POLICY)["labels"]["R2"] == "not_reachable"


def test_synthetic_files_are_never_estimated(tmp_path):
    """Our files carry every ADA property; the estimates must not touch them."""
    p = tmp_path / "b.ifc"
    write_ifc(SCENARIOS[0][1], str(p))
    back = read_ifc(str(p))
    assert all(not e.get("estimated") for e in back["rooms"] + back["doors"])
    assert back["issues"] == []
