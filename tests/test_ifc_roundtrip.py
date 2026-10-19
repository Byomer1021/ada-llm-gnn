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


def test_missing_value_reported(tmp_path):
    b = [s for s in SCENARIOS if "missing" in s[0]][0][1]
    p = tmp_path / "b.ifc"
    write_ifc(b, str(p))
    back = read_ifc(str(p))
    assert any("ClearWidth_mm missing" in i for i in back["issues"])
    assert evaluate(back, RULES)["labels"]["R2"] == "not_reachable"
