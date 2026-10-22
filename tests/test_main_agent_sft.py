"""Fine-tuned main agent's training answers: same verdict as the labels, a valid answer, the right reason."""
import json

import pytest

from src.ada.rules import make_labels
from src.agents.gnn_agents import oracle_findings
from src.agents.main_agent import _parse
from src.agents.main_agent_sft import answer
from tests.test_rules import RULES, SCENARIOS, two_rooms


@pytest.mark.parametrize("name,building,_", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_answer_is_the_label_and_parses(name, building, _):
    labels = make_labels(building, RULES)
    a = answer(oracle_findings(building, RULES), labels, RULES)
    assert list(a)[0] == "explanation"  # reasoning before the verdict
    d = _parse(json.dumps(a))
    assert d["label"] == labels["verdict"]["label"]
    assert d["failing_rules"] == labels["verdict"]["failing_rules"]
    assert ("Every room can be reached" in a["explanation"]) == (d["label"] == "accessible")


def test_explanation_names_the_reason():
    small = {"turning_space_diameter_mm": 1300}
    turn = two_rooms(r1=small)                      # D1 south, D2 east of R1: adjacent walls
    straight = two_rooms(r1=small, d1_xy=(0, 2400))  # D1 west, D2 east: opposite walls
    narrow = two_rooms(d2={"clear_width_mm": 710})
    expl = lambda b: answer(oracle_findings(b, RULES), make_labels(b, RULES), RULES)["explanation"]
    assert "adjacent walls" in expl(turn) and "R2 cannot be reached" in expl(turn)
    assert "crossed in a straight line" in expl(straight)
    assert "D2 (R1 to R2) fails ADA 404.2.3 (clear width 710 mm < 815 mm)" in expl(narrow)


def test_wrong_agents_are_named():
    b = two_rooms()
    f = oracle_findings(b, RULES)
    f["doors"][1]["fail_prob"]["ADA 404.2.4"] = 0.9  # the door agent is wrong about D2
    f["rooms"][1]["reachable_prob"] = 0.1             # the route agent is wrong about R2
    e = answer(f, make_labels(b, RULES), RULES)["explanation"]
    assert "door agent's 0.90 for D2 ADA 404.2.4 is wrong" in e and "route agent disagrees on R2" in e
