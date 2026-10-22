"""
Distillation, step 3: fine-tune the STUDENT (Qwen2.5-7B-Instruct) with LoRA on the teacher's
verified answers (reasoning trace + building JSON).

Input:  data/llm/student_sft.jsonl   (python -m src.llm.generate --export-sft)
Config: config/student.yaml (base model, LoRA, training); command-line flags override it.
The real run needs a CUDA GPU (Colab / lab machine); 4-bit QLoRA fits the 7B model on 16 GB.

    pip install -r requirements-student.txt
    python -m src.llm.train_student
    python -m src.llm.train_student --epochs 2 --max-length 8192
    # CPU smoke test of the code path (tiny random model, 2 steps, no 4-bit, fake data):
    python -m src.llm.generate --dry-run --n 20 && python -m src.llm.generate --export-sft --dry-run
    python -m src.llm.train_student --base-model trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
        --no-4bit --max-steps 2 --data data/llm/student_dryrun_sft.jsonl

The loss covers the teacher's answer only (prompt-completion format), never the system or user
prompt. Examples longer than max_length are DROPPED, never cut: a cut answer would teach the
student to stop inside the JSON. The split is by teacher building with a fixed seed.

Output: models/student/<time>/adapter/  (LoRA weights + tokenizer)
        experiments/<time>_student/metrics.json  (config, seed, split, token lengths, losses)
Serve the adapter with vLLM (no merge needed), then generate with the student:
    vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
        --lora-modules qwen-ada=models/student/<time>/adapter --port 8000
    python -m src.llm.generate --role student --provider local --model qwen-ada --n 2000 --workers 8 --write-ifc
--merge also writes models/student/<time>/merged/ (the full model, e.g. for Ollama; needs ~16 GB RAM).

The same script fine-tunes the main agent: --config config/main_agent_sft.yaml
(data from python -m src.agents.main_agent_sft; served as qwen-main, scored with scripts/evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config" / "student.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_examples(path: str | Path) -> list[dict]:
    """student_sft.jsonl -> TRL prompt-completion examples: prompt = system + user, completion = answer."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r["messages"]
            if msgs[-1]["role"] != "assistant":
                raise ValueError(f"{path}, line {i}: the last message must be the teacher's answer")
            out.append({"id": r.get("id", f"line{i}"), "prompt": msgs[:-1], "completion": msgs[-1:]})
    return out


def split(examples: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Fixed-seed split by teacher building (one example = one building)."""
    ex = sorted(examples, key=lambda e: e["id"])
    random.Random(seed).shuffle(ex)
    n_val = max(1, round(val_frac * len(ex))) if val_frac > 0 and len(ex) > 1 else 0
    return ex[n_val:], ex[:n_val]


def _rel(p: Path) -> str:
    """Path relative to the project if it is inside it, else absolute."""
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def n_tokens(tok, example: dict) -> int:
    """Token count of the whole chat (transformers 5 returns a dict with input_ids, older versions a list)."""
    ids = tok.apply_chat_template(example["prompt"] + example["completion"], tokenize=True)
    return len(ids["input_ids"] if hasattr(ids, "keys") else ids)


def drop_long(examples: list[dict], n_tokens: dict, max_length: int) -> tuple[list[dict], list[str]]:
    """Examples that fit in max_length tokens, and the ids of the dropped ones."""
    kept = [e for e in examples if n_tokens[e["id"]] <= max_length]
    return kept, [e["id"] for e in examples if n_tokens[e["id"]] > max_length]


def main(args):
    import torch
    import peft
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(args.config)
    base = args.base_model or cfg["base_model"]
    seed = cfg["seed"] if args.seed is None else args.seed
    max_length = args.max_length or cfg["max_length"]
    four_bit = cfg["load_in_4bit"] and not args.no_4bit
    epochs = args.epochs or cfg["train"]["epochs"]
    cuda = torch.cuda.is_available()
    if four_bit and not cuda:
        raise SystemExit("4-bit QLoRA needs a CUDA GPU. For a CPU smoke test add --no-4bit (and a tiny --base-model).")

    data = ROOT / (args.data or cfg["data"])
    if not data.exists():
        raise SystemExit(f"{data} not found. Run: python -m src.llm.generate --export-sft")
    tok = AutoTokenizer.from_pretrained(base)
    examples = load_examples(data)
    n_tok = {e["id"]: n_tokens(tok, e) for e in examples}
    train, val = split(examples, cfg["val_frac"], seed)
    train, dropped_train = drop_long(train, n_tok, max_length)
    val, dropped_val = drop_long(val, n_tok, max_length)
    lengths = sorted(n_tok.values())
    print(f"{len(examples)} examples from {data.name}; tokens mean {statistics.mean(lengths):.0f}, "
          f"max {lengths[-1]}; train {len(train)} / val {len(val)}; dropped (> {max_length} tokens): "
          f"{len(dropped_train) + len(dropped_val)}")
    if not train:
        raise SystemExit("No training examples left.")

    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if four_bit else None
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, quantization_config=quant,
                                                 device_map={"": 0} if cuda else None)
    lc = cfg["lora"]
    lora = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      target_modules=lc["target_modules"], task_type="CAUSAL_LM")

    run = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / (args.output_dir or cfg["output_dir"]) / run
    t = cfg["train"]
    sft = SFTConfig(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=epochs, max_steps=args.max_steps or -1,
        per_device_train_batch_size=t["batch_size"], per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["scheduler"], warmup_steps=t["warmup_ratio"],  # a float < 1 = ratio of all steps
        max_length=max_length, completion_only_loss=True,  # loss on the teacher's answer only
        gradient_checkpointing=t["gradient_checkpointing"] and cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        eval_strategy="epoch" if val else "no", save_strategy="epoch", save_total_limit=1,
        logging_steps=1, seed=seed, report_to="none")
    cols = lambda exs: Dataset.from_list([{"prompt": e["prompt"], "completion": e["completion"]} for e in exs])
    trainer = SFTTrainer(model=model, args=sft, train_dataset=cols(train), eval_dataset=cols(val) if val else None,
                         processing_class=tok, peft_config=lora)
    t0 = time.time()
    trainer.train()
    final_eval = trainer.evaluate() if val else {}
    adapter = out_dir / "adapter"
    trainer.save_model(str(adapter))
    tok.save_pretrained(str(adapter))
    seconds = round(time.time() - t0, 1)

    metrics = {
        "config": cfg | {"base_model": base, "seed": seed, "max_length": max_length, "load_in_4bit": four_bit,
                         "data": _rel(data)} | {"train": t | {"epochs": epochs}},
        "args": vars(args), "device": torch.cuda.get_device_name(0) if cuda else "cpu", "dtype": str(dtype),
        "versions": {m.__name__: m.__version__ for m in (torch, transformers, peft, trl)},
        "examples": {"total": len(examples), "train": len(train), "val": len(val),
                     "dropped_too_long": dropped_train + dropped_val, "val_ids": [e["id"] for e in val]},
        "tokens": {"mean": statistics.mean(lengths), "median": statistics.median(lengths), "max": lengths[-1]},
        "final_eval": final_eval, "log_history": trainer.state.log_history, "seconds": seconds,
        "adapter": _rel(adapter)}
    exp = ROOT / "experiments" / f"{run}_{cfg.get('experiment', 'student')}"
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    print(f"Trained in {seconds} s; eval loss {final_eval.get('eval_loss')}. Adapter: {adapter}\n"
          f"Metrics: {exp / 'metrics.json'}")

    if args.merge:
        import gc

        from peft import PeftModel
        del trainer, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        full = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16 if cuda else torch.float32,
                                                    device_map="cpu")
        merged = PeftModel.from_pretrained(full, str(adapter)).merge_and_unload()
        merged.save_pretrained(str(out_dir / "merged"))
        tok.save_pretrained(str(out_dir / "merged"))
        print(f"Merged model: {out_dir / 'merged'}")
    name = cfg["served_name"]
    print(f"Serve:  vllm serve {base} --enable-lora --max-lora-rank {lc['r']} --lora-modules {name}={adapter} "
          f"--port 8000")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", default=None, help="SFT file, relative to the project (default: config)")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true", help="16/32-bit weights instead of 4-bit QLoRA")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--merge", action="store_true", help="also save the merged full model")
    main(ap.parse_args())
