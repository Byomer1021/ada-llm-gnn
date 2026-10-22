"""
Score the whole check (3 GNN agents + main agent) on the TEST split of the synthetic data.

    python scripts/evaluate.py                            # gnn agents, main agent from the config
    python scripts/evaluate.py --provider rule            # baseline main agent without an LLM
    python scripts/evaluate.py --provider local           # Qwen 7B only
    python scripts/evaluate.py --agents oracle            # main agent alone, perfect agents (upper bound)
    python scripts/evaluate.py --limit 20                 # quick try (LLM calls cost time / money)
    # a hosted model in the main agent's "local" slot (e.g. the NVIDIA API, same key as the teacher)
    python scripts/evaluate.py --provider local --llm-url https://integrate.api.nvidia.com/v1 \
        --llm-model mistralai/mistral-7b-instruct-v0.3 --llm-key-env NVIDIA_API_KEY --workers 4

Metrics (success criterion: main agent's verdict macro F1 >= 0.90 on unseen buildings):
  building verdict: accuracy, macro F1, F1 per class, confusion matrix,
                    "always the most common class" baseline
  failing rules:    precision / recall / F1 over (element, rule) pairs, exact-match rate
Output: experiments/<time>_eval_<agents>_<provider>/ (metrics.json + predictions.jsonl)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from src.agents.dataset import load_building, records  # noqa: E402
from src.agents.gnn_agents import oracle_findings, run_agents  # noqa: E402
from src.agents.main_agent import decide, load_config  # noqa: E402
from src.pipeline import get_agents  # noqa: E402

LABELS = ["accessible", "not_accessible"]


def main(a):
    split = json.loads((ROOT / "data" / "datasets" / "split.json").read_text(encoding="utf-8"))
    test_ids = set(split["test"])
    recs = [(role, r) for role, r in records(a.sources) if r["id"] in test_ids]
    if a.limit:
        recs = recs[:a.limit]
    cfg = load_config()
    if a.reachability:
        cfg["main_agent"]["reachability"] = a.reachability
    if a.models_dir:
        cfg["agents"]["models_dir"] = a.models_dir
    # the "local" slot of the main agent can point at any OpenAI-compatible server, e.g. a hosted model
    loc = cfg["main_agent"]["local"]
    loc.update({k: v for k, v in (("base_url", a.llm_url), ("model", a.llm_model), ("api_key_env", a.llm_key_env)) if v})
    if a.max_tokens:
        cfg["main_agent"]["max_tokens"] = a.max_tokens
    if a.timeout:
        cfg["main_agent"]["timeout_s"] = a.timeout
    reach = cfg["main_agent"].get("reachability", "paths")
    who = a.provider or "config"
    if a.provider == "local":
        who += "_" + loc["model"].split("/")[-1]
    out = ROOT / "experiments" / f"{time.strftime('%Y%m%d_%H%M%S')}_eval_{a.agents}_{who}_{reach}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. the agents' findings, one building after the other (fast, CPU)
    agents = None if a.agents == "oracle" else get_agents(ROOT / cfg["agents"]["models_dir"])
    items = []
    for role, r in recs:
        b, _ = load_building(role, r, from_json=False)
        items.append((r, oracle_findings(b) if agents is None else run_agents(b, agents)))
    print(f"findings for {len(items)} buildings; main agent: {who}")

    # 2. the main agent's decisions; LLM calls go out --workers at a time. An answer that stays
    #    invalid after the retries is kept as label "invalid" and counts as wrong, never dropped.
    def one(item):
        try:
            return decide(item[1], provider=a.provider, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            return {"label": "invalid", "failing_rules": [], "warnings": [], "explanation": "",
                    "provider": "failed", "error": f"{e.__class__.__name__}: {e}"[:400], "not_checked": []}

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        decisions = list(ex.map(one, items))

    y, p, tp, fp, fn, exact, used = [], [], 0, 0, 0, 0, Counter()
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, ((r, findings), d) in enumerate(zip(items, decisions), 1):
            truth = r["labels"]["verdict"]
            y.append(truth["label"])
            p.append(d["label"])
            gold = {(x["element"], x["rule"]) for x in truth["failing_rules"]}
            pred = {(x["element"], x["rule"]) for x in d["failing_rules"]}
            tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
            exact += gold == pred
            used[d["provider"]] += 1
            f.write(json.dumps({"id": r["id"], "truth": truth, "decision": d, "findings": findings},
                               ensure_ascii=False) + "\n")
            print(f"[{i}/{len(recs)}] {r['id']}: truth {truth['label']}, predicted {d['label']}"
                  + ("" if truth["label"] == d["label"] else "   <-- wrong"))
    majority = Counter(y).most_common(1)[0][0]
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    m = {
        "n": len(y), "agents": a.agents, "provider": a.provider or "config", "reachability": reach,
        "models_dir": cfg["agents"]["models_dir"],
        "llm": {k: loc[k] for k in ("base_url", "model")} if a.provider in ("local", "auto") else None,
        "providers_used": dict(used), "invalid_answers": p.count("invalid"),
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
        "f1_per_class": dict(zip(LABELS, f1_score(y, p, average=None, labels=LABELS, zero_division=0).tolist())),
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y, p, labels=LABELS).tolist()},
        "baseline_majority_macro_f1": f1_score(y, [majority] * len(y), average="macro", labels=LABELS,
                                               zero_division=0),
        "failing_rules": {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                          "exact_match": exact / max(len(y), 1)},
    }
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(m, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", choices=["gnn", "oracle"], default="gnn")
    ap.add_argument("--provider", choices=["auto", "local", "openai", "rule"], default=None)
    ap.add_argument("--sources", nargs="+", default=["teacher_verified", "student_verified"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reachability", choices=["paths", "route"], default=None,
                    help="rule main agent: trace paths over the door / room agents, or use the route agent "
                         "(default: config/pipeline.yaml)")
    ap.add_argument("--models-dir", default=None, help="trained agents to use (default: config/pipeline.yaml)")
    ap.add_argument("--llm-url", default=None, help="main agent 'local' slot: OpenAI-compatible base URL")
    ap.add_argument("--llm-model", default=None, help="main agent 'local' slot: model name")
    ap.add_argument("--llm-key-env", default=None, help="main agent 'local' slot: env var holding the API key")
    ap.add_argument("--max-tokens", type=int, default=None, help="main agent answer limit (reasoning models need more)")
    ap.add_argument("--workers", type=int, default=1, help="main agent calls at the same time")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per main agent call (reasoning models are slow)")
    main(ap.parse_args())
