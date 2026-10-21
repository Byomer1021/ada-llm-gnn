"""Main agent's decision + agents' findings -> accessibility report (Markdown)."""
from __future__ import annotations

from .rules import ACCESSIBLE, load_rules

LIMITS = None


def _limit(rule: str) -> str:
    global LIMITS
    if LIMITS is None:
        R = load_rules()
        d, lo_hi = R["door"], R["door"]["hardware_height_mm"]
        LIMITS = {"ADA 404.2.3": f"clear width >= {d['clear_width_min_mm']} mm",
                  "ADA 404.2.5": f"threshold <= {d['threshold_max_mm']} mm",
                  "ADA 404.2.7": f"hardware height {lo_hi[0]}-{lo_hi[1]} mm",
                  "ADA 404.2.4": f"maneuvering clearance pull >= {d['maneuvering_pull_min_mm']} mm, "
                                 f"push >= {d['maneuvering_push_min_mm']} mm",
                  "ADA 304.3.1": f"turning space >= {R['room']['turning_diameter_min_mm']} mm",
                  "ADA 403.5.1": f"straight passage >= {R['passage']['straight_min_width_mm']} mm"}
    return LIMITS.get(rule, "")


FIELD_OF = {"ADA 404.2.3": ["clear_width_mm"], "ADA 404.2.5": ["threshold_height_mm"],
            "ADA 404.2.7": ["hardware_height_mm"], "ADA 404.2.4": ["maneuvering_pull_mm", "maneuvering_push_mm"],
            "ADA 304.3.1": ["turning_space_diameter_mm"], "ADA 403.5.1": ["width_mm", "length_mm"]}


def _value(findings: dict, element: str, rule: str) -> str:
    for e in findings["doors"] + findings["rooms"]:
        if e["id"] == element:
            est = e.get("estimated", {})
            vals = [(k, e["values"].get(k)) for k in FIELD_OF.get(rule, []) if k in e["values"]]
            return ", ".join(f"{k.replace('_mm', '')} {'not in the file' if v is None else f'{v:g}'}"
                             f"{' (estimated)' if k in est else ''}" for k, v in vals)
    return ""


def _door_result(d: dict) -> str:
    bad = [r for r, p in d["fail_prob"].items() if p is not None and p >= 0.5]
    out = "fails " + ", ".join(bad) if bad else "passes"
    return out + (f" (not checked: {', '.join(d['not_checked'])})" if d.get("not_checked") else "")


def make_report(building: dict, decision: dict, findings: dict, title: str | None = None) -> str:
    name = title or building.get("building_id") or "building"
    ok = decision["label"] == ACCESSIBLE
    lines = [f"# Accessibility report: {name}", "",
             f"**Result: {'ACCESSIBLE' if ok else 'NOT ACCESSIBLE'}** for a wheelchair user (2010 ADA Standards)", ""]
    if decision.get("explanation"):
        lines += [decision["explanation"], ""]
    if decision["failing_rules"]:
        lines += ["## Failing ADA rules", "", "| Element | ADA rule | Requirement | Value in the IFC (mm) |",
                  "|---|---|---|---|"]
        for f in decision["failing_rules"]:
            lines.append(f"| {f['element']} | {f['rule']} | {_limit(f['rule'])} | {_value(findings, f['element'], f['rule'])} |")
        lines.append("")
    if decision.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {w}" for w in decision["warnings"]] + [""]
    if decision.get("not_checked"):
        by_rule: dict[str, list[str]] = {}
        for nc in decision["not_checked"]:
            by_rule.setdefault(nc["rule"], []).append(nc["element"])
        lines += ["## Not checked (value not in the IFC file)", "",
                  "The result above holds for the rules that could be checked. These could not:", ""]
        lines += [f"- {rule} ({_limit(rule)}): {', '.join(els)}" for rule, els in by_rule.items()] + [""]
    est = [(e["id"], k, e["values"].get(k), how) for e in findings["doors"] + findings["rooms"]
           for k, how in e.get("estimated", {}).items()]
    if est:
        lines += ["## Estimated values (not in the file, estimated from its geometry / IFC data)", "",
                  "| Element | Value | mm | How |", "|---|---|---|---|"]
        lines += [f"| {el} | {k.replace('_mm', '')} | {v:g} | {how} |" for el, k, v, how in est] + [""]
    lines += ["## What the agents found", "",
              "| Element | Agent | Result |", "|---|---|---|"]
    for d in findings["doors"]:
        lines.append(f"| {d['id']} ({' - '.join(d['connects'])}) | door | {_door_result(d)} |")
    for r in findings["rooms"]:
        tp = r["turning_fail_prob"]
        lines.append(f"| {r['id']} | room | " + ("turning space not checked" if tp is None else
                     f"turning space {'fails' if tp >= 0.5 else 'ok'} (p = {tp:.2f})") + " |")
        lines.append(f"| {r['id']} | route | {'reachable' if r['reachable_prob'] >= 0.5 else 'not reachable'} "
                     f"(p = {r['reachable_prob']:.2f}) |")
    if findings.get("data_issues"):
        lines += ["", "## Data issues (read from the IFC file)"] + [f"- {i}" for i in findings["data_issues"]]
    src = "rule engine (ORACLE, for testing only)" if findings["source"] == "oracle" else "GNN agents"
    lines += ["", f"_Agents: {src}. Main agent: {decision.get('provider', '?')}._"]
    if decision.get("fallback_reason"):
        lines.append(f"_Fallback used: {decision['fallback_reason']}_")
    return "\n".join(lines) + "\n"
