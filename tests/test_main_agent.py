"""Main agent: baseline rules, LLM answer parsing, and the local -> GPT-4.1-mini fallback."""
import json

import pytest

from src.ada.rules import evaluate
from src.agents.gnn_agents import oracle_findings
from src.agents.main_agent import EXAMPLES, _parse, decide, load_config, messages, rule_decide
from tests.test_rules import RULES, SCENARIOS, two_rooms


@pytest.mark.parametrize("reachability", ["paths", "route"])
@pytest.mark.parametrize("name,building,_", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_rule_main_agent_with_perfect_agents_matches_verdict(name, building, _, reachability):
    v = evaluate(building, RULES)["verdict"]
    d = rule_decide(oracle_findings(building, RULES), reachability=reachability)
    assert d["label"] == v["label"]
    assert {(f["element"], f["rule"]) for f in d["failing_rules"]} == \
           {(f["element"], f["rule"]) for f in v["failing_rules"]}


def test_findings_carry_the_door_walls():
    f = oracle_findings(two_rooms(), RULES)  # D1 on the south wall of R1, D2 between R1 (east) and R2 (west)
    assert {d["id"]: d["walls"] for d in f["doors"]} == {"D1": {"R1": "S"}, "D2": {"R1": "E", "R2": "W"}}


def test_paths_straight_crossing_blocked_by_a_narrow_room():
    # R1 cannot turn; D1 west / D2 east = opposite walls, but R1 is only 900 mm across (< 915, ADA 403.5.1)
    b = two_rooms(d1_xy=(0, 450), r1={"turning_space_diameter_mm": 800, "length_mm": 900},
                  d2={"y_mm": 450})
    b["rooms"][1]["length_mm"] = 900
    v = evaluate(b, RULES)["verdict"]
    d = rule_decide(oracle_findings(b, RULES))
    assert v["label"] == d["label"] == "not_accessible"
    assert d["failing_rules"] == [{"element": "R1", "rule": "ADA 403.5.1"}] == \
           [{"element": f["element"], "rule": f["rule"]} for f in v["failing_rules"]]


def test_paths_override_a_wrong_route_agent_and_say_so():
    f = oracle_findings(two_rooms(r1={"turning_space_diameter_mm": 1300}, d1_xy=(0, 2400)), RULES)  # straight
    for r in f["rooms"]:
        r["reachable_prob"] = 0.0  # a route agent that thinks nothing can be reached
    d = rule_decide(f)
    assert d["label"] == "accessible"
    assert sum("route agent disagrees" in w for w in d["warnings"]) == 2
    assert rule_decide(f, reachability="route")["label"] == "not_accessible"


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create."""

    def __init__(self, answers=None, error=None):
        self.answers, self.error, self.calls = list(answers or []), error, 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        if self.error:
            raise self.error
        msg = type("M", (), {"content": self.answers.pop(0)})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


GOOD = json.dumps({"label": "not_accessible", "failing_rules": [{"element": "D2", "rule": "ADA 404.2.3"}],
                   "warnings": [], "explanation": "D2 is too narrow, so R2 cannot be reached."})
FINDINGS = oracle_findings(SCENARIOS[1][1])


def test_parse_accepts_think_and_code_fences():
    d = _parse("<think>reasoning</think>\n```json\n" + GOOD + "\n```")
    assert d["label"] == "not_accessible" and d["failing_rules"][0]["rule"] == "ADA 404.2.3"


@pytest.mark.parametrize("bad", ['{"label": "maybe", "failing_rules": []}',
                                 '{"label": "accessible", "failing_rules": [{"element": "D2", "rule": "ADA 999"}]}',
                                 "not json"])
def test_parse_rejects_bad_answers(bad):
    with pytest.raises(Exception):
        _parse(bad)


def test_local_model_used_first():
    d = decide(FINDINGS, "auto", load_config(), clients={"local": FakeClient([GOOD]), "openai": FakeClient([GOOD])})
    assert d["provider"].startswith("local:") and "fallback_reason" not in d


def test_fallback_when_local_is_down():
    local, remote = FakeClient(error=ConnectionError("no server")), FakeClient([GOOD])
    d = decide(FINDINGS, "auto", load_config(), clients={"local": local, "openai": remote})
    assert d["provider"].startswith("openai:") and "no server" in d["fallback_reason"]


def test_fallback_when_local_gives_invalid_json_twice():
    local, remote = FakeClient(["oops", '{"label": "x"}']), FakeClient([GOOD])
    d = decide(FINDINGS, "auto", load_config(), clients={"local": local, "openai": remote})
    assert local.calls == 2 and d["provider"].startswith("openai:")


def test_retry_once_on_invalid_json():
    local = FakeClient(["oops", GOOD])
    d = decide(FINDINGS, "local", load_config(), clients={"local": local})
    assert local.calls == 2 and d["label"] == "not_accessible"


def test_few_shot_examples_agree_with_the_rule_main_agent():
    """The worked examples must teach the same logic as the path search, and be valid answers."""
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    assert {ex["answer"]["label"] for ex in examples} == {"accessible", "not_accessible"}
    for ex in examples:
        d, a = rule_decide(ex["findings"]), _parse(json.dumps(ex["answer"]))
        assert a["label"] == d["label"], ex["case"]
        assert {(f["element"], f["rule"]) for f in a["failing_rules"]} == \
            {(f["element"], f["rule"]) for f in d["failing_rules"]}, ex["case"]


def test_few_shot_messages_put_the_examples_before_the_building():
    n = len(json.loads(EXAMPLES.read_text(encoding="utf-8")))
    zero, few = messages(FINDINGS), messages(FINDINGS, few_shot=True)
    assert [m["role"] for m in zero] == ["system", "user"]
    assert len(few) == 2 + 2 * n and [m["role"] for m in few[1:3]] == ["user", "assistant"]
    assert few[0] == zero[0] and few[-1] == zero[-1]


def test_both_options_down_raises():
    with pytest.raises(RuntimeError):
        decide(FINDINGS, "auto", load_config(),
               clients={"local": FakeClient(error=ConnectionError()), "openai": FakeClient(error=ConnectionError())})
