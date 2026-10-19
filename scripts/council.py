"""
Architecture council review of the TEACHER's buildings (verify step, after the rule engine).

1) packet:  every verified teacher building -> a 3D picture + one row in a review sheet
       python scripts/council.py packet
   Output: data/council/pictures/<id>.png and data/council/council_sheet.csv
   Send the pictures (and the IFC files, for an IFC viewer) and the sheet to the council.
   Each reviewer fills "approved" (yes / no) and "comment" in their own copy of the sheet.
   (When the survey questions are decided, add one column per question to the sheet.)

2) collect: read the filled sheets back
       python scripts/council.py collect data/council/filled/*.csv
   A building is approved if at least --min-share of its reviewers said yes (default 0.5 = at least half).
   Output: data/llm/teacher_approved.jsonl (used by `generate.py --export-sft` and the dataset)
   and the approval rate (success criterion: >= 90% of the teacher's buildings approved).

--dry-run (both commands): use the fake records of `generate.py --dry-run` (teacher_dryrun_*),
write to data/council/dryrun/ and teacher_dryrun_approved.jsonl; the real files are not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

LLM = ROOT / "data" / "llm"
OUT = ROOT / "data" / "council"
FIELDS = ["id", "picture", "ifc_file", "intended_result", "failing_rules", "warnings", "approved", "comment"]


def _records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tag(args) -> str:
    return "teacher_dryrun" if args.dry_run else "teacher"


def _out(args) -> Path:
    return OUT / "dryrun" if args.dry_run else OUT


def packet(args):
    from render_3d import render  # needs matplotlib

    from src.ada.ifc_writer import write_ifc

    tag, out = _tag(args), _out(args)
    recs = _records(LLM / f"{tag}_verified.jsonl")
    (out / "pictures").mkdir(parents=True, exist_ok=True)
    ifc_dir = ROOT / "data" / "synthetic" / tag
    ifc_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "council_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for i, r in enumerate(recs, 1):
            ifc = ifc_dir / f"{r['id']}.ifc"
            if not ifc.exists():
                write_ifc({**r["building"], "building_id": r["id"]}, str(ifc))
            png = out / "pictures" / f"{r['id']}.png"
            if not png.exists() or args.force:
                render(str(ifc), str(png))
            v = r["labels"]["verdict"]
            w.writerow({"id": r["id"], "picture": f"pictures/{png.name}", "ifc_file": str(ifc.relative_to(ROOT)),
                        "intended_result": v["label"],
                        "failing_rules": "; ".join(f"{x['element']} {x['rule']}" for x in v["failing_rules"]),
                        "warnings": "; ".join(v["warnings"]), "approved": "", "comment": ""})
            print(f"[{i}/{len(recs)}] {r['id']}")
    print(f"Sheet: {out / 'council_sheet.csv'}  Pictures: {out / 'pictures'}")


def collect(args):
    votes = defaultdict(list)
    comments = defaultdict(list)
    for sheet in args.sheets:
        with open(sheet, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                a = (row.get("approved") or "").strip().lower()
                if a in ("yes", "y", "evet", "e", "1", "true"):
                    votes[row["id"]].append(1)
                elif a in ("no", "n", "hayir", "hayır", "h", "0", "false"):
                    votes[row["id"]].append(0)
                if row.get("comment"):
                    comments[row["id"]].append(row["comment"])
    tag = _tag(args)
    recs = {r["id"]: r for r in _records(LLM / f"{tag}_verified.jsonl")}
    approved = 0
    with open(LLM / f"{tag}_approved.jsonl", "w", encoding="utf-8") as g:
        for bid, rec in recs.items():
            v = votes.get(bid, [])
            ok = bool(v) and sum(v) / len(v) >= args.min_share
            if ok:
                approved += 1
                g.write(json.dumps({**rec, "council": {"votes": v, "comments": comments[bid]}},
                                   ensure_ascii=False) + "\n")
    reviewed = sum(1 for b in recs if votes.get(b))
    rate = approved / max(len(recs), 1)
    print(f"Reviewed {reviewed}/{len(recs)} buildings; approved {approved} ({100 * rate:.1f}%). Target: >= 90%.")
    if reviewed < len(recs):
        print("Note: buildings without any vote count as NOT approved.")
    print(f"Wrote {LLM / f'{tag}_approved.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("packet")
    p1.add_argument("--force", action="store_true", help="re-render pictures that already exist")
    p2 = sub.add_parser("collect")
    p2.add_argument("sheets", nargs="+", help="filled CSV sheets, one per reviewer")
    p2.add_argument("--min-share", type=float, default=0.5, help="share of 'yes' votes needed (default 0.5)")
    for p in (p1, p2):
        p.add_argument("--dry-run", action="store_true", help="fake records of generate.py --dry-run")
    a = ap.parse_args()
    packet(a) if a.cmd == "packet" else collect(a)
