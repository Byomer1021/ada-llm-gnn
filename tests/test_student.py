"""Student training data: prompt-completion format, split by building, long examples dropped (no GPU needed)."""
import json

from src.llm.train_student import drop_long, load_config, load_examples, split


def _sft(tmp_path, n=20):
    p = tmp_path / "sft.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"teacher_{i}", "messages": [
                {"role": "system", "content": "sys"}, {"role": "user", "content": f"design {i}"},
                {"role": "assistant", "content": f"<think>\nwhy {i}\n</think>\n{{}}"}]}) + "\n")
    return p


def test_loss_only_on_the_answer(tmp_path):
    ex = load_examples(_sft(tmp_path))
    assert [m["role"] for m in ex[0]["prompt"]] == ["system", "user"]
    assert [m["role"] for m in ex[0]["completion"]] == ["assistant"]


def test_split_is_fixed_and_by_building(tmp_path):
    ex = load_examples(_sft(tmp_path))
    tr, va = split(ex, 0.1, seed=0)
    assert len(va) == 2 and len(tr) == 18
    assert not {e["id"] for e in tr} & {e["id"] for e in va}
    assert [e["id"] for e in split(list(reversed(ex)), 0.1, seed=0)[1]] == [e["id"] for e in va]


def test_long_examples_are_dropped_not_cut(tmp_path):
    ex = load_examples(_sft(tmp_path, n=3))
    kept, dropped = drop_long(ex, {"teacher_0": 100, "teacher_1": 5000, "teacher_2": 4096}, max_length=4096)
    assert dropped == ["teacher_1"]
    assert [e["completion"] for e in kept] == [ex[0]["completion"], ex[2]["completion"]]  # unchanged


def test_config_has_what_the_script_reads():
    cfg = load_config()
    assert {"base_model", "data", "val_frac", "seed", "max_length", "load_in_4bit", "lora", "train",
            "output_dir", "served_name"} <= set(cfg)
