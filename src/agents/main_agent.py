"""
Main agent: an LLM that combines the three GNN agents' findings into the final verdict
(accessible / not_accessible + failing ADA rules + warnings + a short explanation).

Two options (config/pipeline.yaml, main_agent):
  1. local Qwen 7B (Ollama or vLLM, OpenAI-compatible API)          provider "local"
  2. GPT-4.1-mini over the OpenAI API, if option 1 is not enough    provider "openai"
  "auto" tries 1, then 2.  "rule" = fixed combination rules, no LLM (baseline, offline).

    decision = decide(findings)                      # provider from the config
    decision = decide(findings, provider="rule")

DECISION FORMAT
{"label": "accessible" | "not_accessible", "failing_rules": [{"element": "D2", "rule": "ADA 404.2.3"}],
 "warnings": [...], "explanation": "...", "provider": "local:qwen2.5:7b" | "openai:gpt-4.1-mini" | "rule",
 "not_checked": [{"element": "D1", "rule": "ADA 404.2.7"}]}
"not_checked" (value not in the file) is copied from the findings, not decided by the LLM:
it is a fact about the input, not a judgement.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ada.rules import ACCESSIBLE, DOOR_RULES, EXTERIOR, NOT_ACCESSIBLE, ROOM_RULE  # noqa: E402

SYSTEM_PROMPT = (ROOT / "prompts" / "main_agent_system.txt").read_text(encoding="utf-8")
CONFIG = ROOT / "config" / "pipeline.yaml"
KNOWN_RULES = set(DOOR_RULES) | {ROOM_RULE, "ADA 403.5.1"}


def load_config(path: str | Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- baseline without an LLM
def rule_decide(findings: dict, thr: float = 0.5) -> dict:
    """Fixed combination: reachability from the route agent, causes from the door / room agents."""
    rooms = {r["id"]: r for r in findings["rooms"]}
    reach = {rid: r["reachable_prob"] >= thr for rid, r in rooms.items()} | {EXTERIOR: True}
    # None = rule not checked (value not in the file): never counts as failing
    door_fails = {d["id"]: [r for r, p in d["fail_prob"].items() if p is not None and p >= thr]
                  for d in findings["doors"]}
    turn_fails = {rid for rid, r in rooms.items() if (r["turning_fail_prob"] or 0) >= thr}
    failing, blamed = [], set()
    for d in findings["doors"]:
        a, b = d["connects"]
        if a not in reach or b not in reach or reach[a] == reach[b]:
            continue
        inside = a if reach[a] else b  # the reachable side of a door into an unreachable room
        if door_fails[d["id"]]:
            failing += [{"element": d["id"], "rule": r} for r in door_fails[d["id"]]]
            blamed.add(d["id"])
        elif inside in turn_fails:  # door is fine, so the room before it could not be crossed
            failing.append({"element": inside, "rule": ROOM_RULE})
            blamed.add(inside)
    unreachable = [rid for rid in rooms if not reach[rid]]
    failing = [dict(t) for t in dict.fromkeys(tuple(f.items()) for f in failing)]
    warnings = [f"{rid}: turning space may fail ({ROOM_RULE}); reachable but cannot turn around"
                for rid in turn_fails if rid not in blamed]
    warnings += [f"{did}: fails {', '.join(r)} but blocks no room" for did, r in door_fails.items()
                 if r and did not in blamed]
    expl = "All rooms can be reached from the entrance." if not unreachable else \
        f"Not reachable from the entrance: {', '.join(unreachable)}."
    if unreachable and not failing:
        warnings.append("route agent predicts unreachable rooms but no failing element was found on the way")
    return {"label": NOT_ACCESSIBLE if unreachable else ACCESSIBLE, "failing_rules": failing,
            "warnings": warnings, "explanation": expl, "provider": "rule"}


# ---------------------------------------------------------------- LLM
def _parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = text.find("{"), text.rfind("}")
    d = json.loads(text[i:j + 1])
    if d.get("label") not in (ACCESSIBLE, NOT_ACCESSIBLE):
        raise ValueError(f"bad label {d.get('label')!r}")
    fr = d.get("failing_rules") or []
    if not isinstance(fr, list) or not all(isinstance(f, dict) and "element" in f and "rule" in f for f in fr):
        raise ValueError("bad failing_rules")
    unknown = {f["rule"] for f in fr} - KNOWN_RULES
    if unknown:
        raise ValueError(f"unknown rules {unknown}")
    return {"label": d["label"], "failing_rules": [{"element": f["element"], "rule": f["rule"]} for f in fr],
            "warnings": [str(w) for w in d.get("warnings") or []], "explanation": str(d.get("explanation", ""))}


def _client(opt: dict, timeout: float):
    from openai import OpenAI
    key = os.environ.get(opt["api_key_env"]) if opt.get("api_key_env") else "EMPTY"
    if not key:
        raise RuntimeError(f"{opt['api_key_env']} is not set")
    return OpenAI(base_url=opt.get("base_url"), api_key=key, timeout=timeout, max_retries=1)


def llm_decide(findings: dict, opt: dict, cfg: dict, client=None, tries: int = 2) -> dict:
    client = client or _client(opt, cfg.get("timeout_s", 120))
    user = "Findings of the three agents:\n" + json.dumps(findings, ensure_ascii=False, indent=1)
    last = None
    for _ in range(tries):
        resp = client.chat.completions.create(
            model=opt["model"], temperature=cfg.get("temperature", 0), max_tokens=cfg.get("max_tokens", 1500),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
        try:
            return _parse(resp.choices[0].message.content)
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise ValueError(f"invalid answer after {tries} tries: {last}")


def not_checked(findings: dict) -> list[dict]:
    return [{"element": e["id"], "rule": r} for e in findings["doors"] + findings["rooms"]
            for r in e.get("not_checked", [])]


def decide(findings: dict, provider: str | None = None, cfg: dict | None = None, clients: dict | None = None) -> dict:
    """clients: optional {"local": client, "openai": client} (tests inject fakes here)."""
    d = _decide(findings, provider, cfg, clients)
    d["not_checked"] = not_checked(findings)
    return d


def _decide(findings: dict, provider: str | None, cfg: dict | None, clients: dict | None) -> dict:
    cfg = cfg or load_config()
    m = cfg["main_agent"]
    provider = provider or m["provider"]
    if provider == "rule":
        return rule_decide(findings, cfg["agents"]["threshold"])
    order = {"auto": ["local", "openai"], "local": ["local"], "openai": ["openai"]}[provider]
    errors = []
    for name in order:
        opt = m["local"] if name == "local" else m["fallback"]
        try:
            d = llm_decide(findings, opt, m, client=(clients or {}).get(name))
            d["provider"] = f"{name}:{opt['model']}"
            if errors:
                d["fallback_reason"] = "; ".join(errors)
            return d
        except Exception as e:  # server not running, no key, invalid JSON, ...
            errors.append(f"{name} failed ({e.__class__.__name__}: {e})")
    raise RuntimeError("main agent failed: " + "; ".join(errors))
