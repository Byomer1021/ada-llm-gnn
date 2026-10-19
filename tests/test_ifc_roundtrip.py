"""IFC writer -> IFC reader must give back the same values and the same labels."""
import pytest

from src.ada.ifc_reader import read_ifc
from src.ada.ifc_writer import write_ifc
from src.ada.rules import evaluate, load_rules
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
