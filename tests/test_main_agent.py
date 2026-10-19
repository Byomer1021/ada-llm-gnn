"""Main agent: baseline rules, LLM answer parsing, and the local -> GPT-4.1-mini fallback."""
import json

import pytest

from src.ada.rules import evaluate
from src.agents.gnn_agents import oracle_findings
from src.agents.main_agent import _parse, decide, load_config, rule_decide
from tests.test_rules import RULES, SCENARIOS


@pytest.mark.parametrize("name,building,_", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_rule_main_agent_with_perfect_agents_matches_verdict(name, building, _):
    v = evaluate(building, RULES)["verdict"]
    d = rule_decide(oracle_findings(building, RULES))
    assert d["label"] == v["label"]
    assert {(f["element"], f["rule"]) for f in d["failing_rules"]} == \
           {(f["element"], f["rule"]) for f in v["failing_rules"]}


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


def test_both_options_down_raises():
    with pytest.raises(RuntimeError):
        decide(FINDINGS, "auto", load_config(),
               clients={"local": FakeClient(error=ConnectionError()), "openai": FakeClient(error=ConnectionError())})
