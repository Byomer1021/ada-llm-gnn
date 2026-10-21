"""LLM generation: prompts, dry-run answers and their verification against the target verdict."""
import argparse
import json
import random
from collections import Counter

import src.llm.generate as gen
from src.ada.rules import door_failures, room_failures
from src.llm.generate import SCENARIOS, fake_answer, make_user_prompt, split_answer, validate
from tests.test_rules import RULES

COMPLIANT_DOOR = {"id": "D", "clear_width_mm": 900, "threshold_height_mm": 5, "hardware_height_mm": 1000,
                  "maneuvering_pull_mm": 1600, "maneuvering_push_mm": 1300}


def test_correct_answers_pass_and_wrong_label_fails():
    rng = random.Random(0)
    seen = set()
    for _ in range(300):
        sc, c, target, _ = make_user_prompt(rng, RULES)
        _, text = split_answer(fake_answer(sc, target, rng))
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
    sc, c, target, _ = make_user_prompt(rng, RULES)
    while sc["id"] != "d2_width":
        sc, c, target, _ = make_user_prompt(rng, RULES)
    b = json.loads(split_answer(fake_answer(sc, target, rng))[1])
    b["doors"][1]["clear_width_mm"] = 900  # now everything passes, but the target says D2 fails
    assert any("rule engine" in p for p in validate(b, target, RULES))


def test_required_values_fail_or_comply_as_asked():
    """Checked with the rule engine itself, so the prompt values follow config/ada_rules.yaml."""
    rng = random.Random(2)
    gaps = {True: [], False: []}
    for _ in range(500):
        _, _, target, _ = make_user_prompt(rng, RULES)
        for v in target["values"]:
            if v["element"].startswith("D"):
                fails = door_failures({**COMPLIANT_DOOR, v["field"]: v["value"]}, RULES)
            else:
                fails = room_failures({"id": v["element"], "turning_space_diameter_mm": v["value"]}, RULES)
            assert bool(fails) == v["fails"], v
            gaps[v["fails"]].append(abs(v["value"] - float(v["limit"].split()[1])))
    assert all(g <= 30 for g in gaps[False])  # compliant values are close to the limit
    near = sum(g <= 30 for g in gaps[True]) / len(gaps[True])
    assert 0.35 < near < 0.65  # about half of the failing values are just past the limit


def test_layouts_cover_every_orientation_and_follow_the_scenario():
    rng = random.Random(3)
    seen = Counter()
    for _ in range(400):
        sc, _, target, prompt = make_user_prompt(rng, RULES)
        lay = target["layout"]
        opposite = lay["D1"] == gen.OPPOSITE_WALL[lay["D2"]]
        assert lay["D1"] != lay["D2"]
        assert sc["placement"] != gen.OPP or opposite
        assert sc["placement"] != gen.ADJ or not opposite
        assert (gen.OPP if opposite else gen.ADJ) in prompt
        seen["D1" + lay["D1"]] += 1
        seen["D2" + lay["D2"]] += 1
    assert all(seen[d + w] > 0 for d in ("D1", "D2") for w in "SNWE")


def test_scenario_filter_and_every_door_value_can_fail_on_both_doors():
    rng = random.Random(4)
    prompts = [make_user_prompt(rng, RULES, ["d1_fails"]) for _ in range(300)]
    assert {sc["id"] for sc, *_ in prompts} == {"d1_fails"}
    d1_fields = {v["field"] for _, _, t, _ in prompts for v in t["values"] if v["fails"]}
    d2_fields = {c["field"] for s in SCENARIOS if s["id"].startswith("d2_") for c in s["causes"]}
    assert d1_fields == d2_fields == set(gen.DOOR_FIELDS)  # no "the entrance door never fails X" shortcut


def _two_rooms(d1_pull=5000, d2_pull=3000, d2_push=4000):
    """R1 4000 x 3500 and R2 east of it (3000 x 3500). D1 on the south wall of R1 (depth 3500 inside,
    6000 allowed outside); D2 on the shared wall (depth 4000 in R1, 3000 in R2)."""
    door = lambda did, a, b, x, y, pull, push: {
        "id": did, "connects": [a, b], "x_mm": x, "y_mm": y, "clear_width_mm": 900, "threshold_height_mm": 5,
        "hardware_height_mm": 1000, "hardware_type": "Lever", "maneuvering_pull_mm": pull, "maneuvering_push_mm": push}
    return {"rooms": [{"id": "R1", "x_mm": 0, "y_mm": 0, "width_mm": 4000, "length_mm": 3500,
                       "turning_space_diameter_mm": 1600},
                      {"id": "R2", "x_mm": 4000, "y_mm": 0, "width_mm": 3000, "length_mm": 3500,
                       "turning_space_diameter_mm": 1600}],
            "doors": [door("D1", "EXTERIOR", "R1", 2000, 0, d1_pull, 3000),
                      door("D2", "R1", "R2", 4000, 1750, d2_pull, d2_push)],
            "intended_verdict": {"label": "accessible", "failing_rules": []}}


def test_maneuvering_clearance_must_fit_the_room():
    target = {"label": "accessible", "failing_rules": [], "layout": {"D1": "S", "D2": "E"}}
    # each clearance as deep as the room on its side: prompt v4 rejected two such buildings by mistake
    assert validate(_two_rooms(), target, RULES) == []
    deep = validate(_two_rooms(d2_push=4100), target, RULES)  # deeper than R1 (4000) and R2 (3000)
    assert any("clearance of D2 is deeper than the room" in p for p in deep)
    outside = validate(_two_rooms(d1_pull=6500), target, RULES)  # deeper than outside (6000) and R1
    assert any("clearance of D1 is deeper than the room" in p for p in outside)


def test_layout_of_the_prompt_must_be_followed():
    target = {"label": "accessible", "failing_rules": [], "layout": {"D1": "W", "D2": "E"}}
    assert "door D1 is not on the west wall of R1 as asked" in validate(_two_rooms(), target, RULES)


def _dry_args(seed, n=3):
    return argparse.Namespace(role="teacher", provider="nemotron", model=None, base_url=None, n=n, temperature=1.0,
                              max_tokens=100, seed=seed, workers=1, scenario=None, tag=None, write_ifc=False,
                              dry_run=True)


def test_tag_keeps_e1_baseline_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "OUT_DIR", tmp_path)
    gen.generate(argparse.Namespace(**{**vars(_dry_args(0)), "role": "student", "tag": "student_base"}))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["student_base_dryrun_raw.jsonl",
                                                          "student_base_dryrun_verified.jsonl"]


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
    assert list(dataset.records(["teacher_verified", "student_verified"])) == []
