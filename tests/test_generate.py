"""LLM generation: dry-run answers are verified against the target verdict."""
import argparse
import json
import random

import src.llm.generate as gen
from src.llm.generate import SCENARIOS, fake_answer, make_user_prompt, split_answer, validate
from tests.test_rules import RULES


def test_correct_answers_pass_and_wrong_label_fails():
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        sc, c, target, _ = make_user_prompt(rng)
        _, text = split_answer(fake_answer(sc, c, rng))
        b = json.loads(text)
        problems = validate(b, target, RULES)
        if b["intended_verdict"]["label"] == target["label"]:
            assert problems == [], (sc["id"], problems)
            seen.add(sc["id"])
        else:  # simulated model mistake
            assert any("intended_verdict" in p for p in problems)
    assert seen == {s["id"] for s in SCENARIOS}


def test_wrong_building_is_rejected():
    rng = random.Random(1)
    sc = next(s for s in SCENARIOS if s["id"] == "d2_width")
    _, text = split_answer(fake_answer(sc, sc["causes"][0], rng))
    b = json.loads(text)
    b["doors"][1]["clear_width_mm"] = 900  # now everything passes, but the target says D2 fails
    target = {"label": "not_accessible", "failing_rules": [{"element": "D2", "rule": "ADA 404.2.3"}]}
    assert any("rule engine" in p for p in validate(b, target, RULES))


def _dry_args(seed, n=3):
    return argparse.Namespace(role="teacher", provider="nemotron", model=None, base_url=None, n=n,
                              temperature=1.0, max_tokens=100, seed=seed, write_ifc=False, dry_run=True)


def test_seed_is_stored_and_reuse_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gen, "OUT_DIR", tmp_path)
    raw = tmp_path / "teacher_dryrun_raw.jsonl"
    gen.generate(_dry_args(4))
    assert "WARNING" not in capsys.readouterr().out
    assert gen.used_seeds(raw) == {4}
    gen.generate(_dry_args(4))
    assert "seed 4 was already used" in capsys.readouterr().out
    gen.generate(_dry_args(5))
    assert "WARNING" not in capsys.readouterr().out
    assert [json.loads(x)["seed"] for x in raw.read_text(encoding="utf-8").splitlines()] == [4] * 6 + [5] * 3


def test_dry_run_stays_out_of_the_real_data(tmp_path, monkeypatch):
    from src.agents import dataset
    monkeypatch.setattr(gen, "OUT_DIR", tmp_path)
    monkeypatch.setattr(dataset, "LLM", tmp_path)
    gen.generate(_dry_args(0, n=10))
    gen.export_sft(dry_run=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "student_dryrun_sft.jsonl", "teacher_dryrun_raw.jsonl", "teacher_dryrun_verified.jsonl"]
    recs = [json.loads(x) for x in (tmp_path / "teacher_dryrun_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["model"] == "dry-run" and r["id"].startswith("teacher_dryrun_") for r in recs)
    # the dataset finds the fake records (and their IFC folder) only when asked for them by name
    assert {role for role, _ in dataset.records(["teacher_dryrun_verified"])} == {"teacher_dryrun"}
    assert list(dataset.records(["teacher_approved", "teacher_verified", "student_verified"])) == []
