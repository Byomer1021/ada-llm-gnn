"""
Score the whole check (3 GNN agents + main agent) on the TEST split of the synthetic data.

    python scripts/evaluate.py                            # gnn agents, main agent from the config
    python scripts/evaluate.py --provider rule            # baseline main agent without an LLM
    python scripts/evaluate.py --provider local           # Qwen 7B only
    python scripts/evaluate.py --agents oracle            # main agent alone, perfect agents (upper bound)
    python scripts/evaluate.py --limit 20                 # quick try (LLM calls cost time / money)

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

from src.agents.dataset import load_building, records  # noqa: E402
from src.pipeline import check_building  # noqa: E402

LABELS = ["accessible", "not_accessible"]


def main(a):
    split = json.loads((ROOT / "data" / "datasets" / "split.json").read_text(encoding="utf-8"))
    test_ids = set(split["test"])
    recs = [(role, r) for role, r in records(a.sources) if r["id"] in test_ids]
    if a.limit:
        recs = recs[:a.limit]
    out = ROOT / "experiments" / f"{time.strftime('%Y%m%d_%H%M%S')}_eval_{a.agents}_{a.provider or 'config'}"
    out.mkdir(parents=True, exist_ok=True)
    y, p, tp, fp, fn, exact, used = [], [], 0, 0, 0, 0, Counter()
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, (role, r) in enumerate(recs, 1):
            b, _ = load_building(role, r, from_json=False)
            res = check_building(b, agents=a.agents, provider=a.provider)
            d, truth = res["decision"], r["labels"]["verdict"]
            y.append(truth["label"])
            p.append(d["label"])
            gold = {(x["element"], x["rule"]) for x in truth["failing_rules"]}
            pred = {(x["element"], x["rule"]) for x in d["failing_rules"]}
            tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
            exact += gold == pred
            used[d["provider"]] += 1
            f.write(json.dumps({"id": r["id"], "truth": truth, "decision": d, "findings": res["findings"]},
                               ensure_ascii=False) + "\n")
            print(f"[{i}/{len(recs)}] {r['id']}: truth {truth['label']}, predicted {d['label']}"
                  + ("" if truth["label"] == d["label"] else "   <-- wrong"))
    majority = Counter(y).most_common(1)[0][0]
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    m = {
        "n": len(y), "agents": a.agents, "provider": a.provider or "config", "providers_used": dict(used),
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
    ap.add_argument("--sources", nargs="+", default=["teacher_approved", "student_verified"])
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
