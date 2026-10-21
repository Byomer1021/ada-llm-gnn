"""
ADA rule engine.

Input: a building dict (see BUILDING FORMAT below).
Output: the building-level VERDICT ("accessible" / "not_accessible" + the failing
ADA rules) and, internally, which rooms can be reached from the entrance.

Where it is used: ONLY in the verify step of IFC generation (it checks the LLM's
buildings and writes their labels). The accessibility check itself is done by the
GNN agents + the main agent (src/agents/); the rule engine is not part of it.

BUILDING FORMAT (shared by the LLM output, the IFC writer and the IFC reader)
{
  "rooms": [{"id": "R1", "x_mm": 0, "y_mm": 0, "width_mm": 5000, "length_mm": 4800,
             "turning_space_diameter_mm": 1525}],
  "doors": [{"id": "D1", "connects": ["EXTERIOR", "R1"], "x_mm": 2500, "y_mm": 0,
             "clear_width_mm": 915, "threshold_height_mm": 0, "hardware_height_mm": 1000,
             "hardware_type": "Lever", "maneuvering_pull_mm": 1550, "maneuvering_push_mm": 1250}]
}
- x_mm / y_mm of a room: lower-left corner; width along x, length along y.
- x_mm / y_mm of a door: door centre, lying on a room wall.
- A value may be None (missing in the IFC file).

RULES (decisions of the project)
1. Door passes if clear width, threshold, hardware height and both maneuvering
   clearances meet ADA. A failing door cannot be passed.
2. A room "fails" if its turning space < 1525 mm (ADA 304.3.1).
3. A failing room is still REACHABLE itself if a valid path leads to it; the
   report adds a warning "cannot turn around".
4. A failing room can be PASSED THROUGH only if no turn is needed: the entry and
   exit doors are on opposite walls and the room is at least 915 mm wide across
   the direction of travel (ADA 403.5.1). Otherwise it blocks the rooms behind it.
5. If a door's wall cannot be determined, assume a turn is needed (conservative).
6. Missing values (config `missing_values.policy`):
   "not_checked" (default): a value that is not in the file (and could not be estimated
   by the IFC reader) does not fail its rule; the rule is listed as NOT CHECKED for that
   element, apart from the failing rules and the warnings.
   "fail": the element fails (conservative; compared in the missing-data experiment E4).
7. BUILDING VERDICT: "accessible" if every room is reachable from the entrance,
   otherwise "not_accessible". The failing rules are the ones that block a room.
   Failures that block nothing (e.g. the last room cannot turn around) are
   warnings; config `verdict.warnings_make_not_accessible` can make them count.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import yaml

EXTERIOR = "EXTERIOR"
REACHABLE, NOT_REACHABLE = "reachable", "not_reachable"          # room level (internal)
ACCESSIBLE, NOT_ACCESSIBLE = "accessible", "not_accessible"      # building level (the output)
DOOR_RULES = ["ADA 404.2.3", "ADA 404.2.5", "ADA 404.2.7", "ADA 404.2.4"]  # door agent outputs
ROOM_RULE = "ADA 304.3.1"                                                  # room agent output
DOOR_FIELD_RULE = {"clear_width_mm": "ADA 404.2.3", "threshold_height_mm": "ADA 404.2.5",
                   "hardware_height_mm": "ADA 404.2.7", "maneuvering_pull_mm": "ADA 404.2.4",
                   "maneuvering_push_mm": "ADA 404.2.4"}
ROOM_FIELD = "turning_space_diameter_mm"
OPPOSITE = {("S", "N"), ("N", "S"), ("W", "E"), ("E", "W")}

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "ada_rules.yaml"


def load_rules(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ element checks
def door_failures(door: dict, R: dict) -> list[dict]:
    """List of failed rules for one door (empty list = door passes)."""
    d = R["door"]
    lo, hi = d["hardware_height_mm"]
    checks = [
        ("clear_width_mm", "ADA 404.2.3", lambda v: v >= d["clear_width_min_mm"], f">= {d['clear_width_min_mm']}"),
        ("threshold_height_mm", "ADA 404.2.5", lambda v: v <= d["threshold_max_mm"], f"<= {d['threshold_max_mm']}"),
        ("hardware_height_mm", "ADA 404.2.7", lambda v: lo <= v <= hi, f"{lo}-{hi}"),
        ("maneuvering_pull_mm", "ADA 404.2.4", lambda v: v >= d["maneuvering_pull_min_mm"], f">= {d['maneuvering_pull_min_mm']}"),
        ("maneuvering_push_mm", "ADA 404.2.4", lambda v: v >= d["maneuvering_push_min_mm"], f">= {d['maneuvering_push_min_mm']}"),
    ]
    out = []
    for key, rule, ok, limit in checks:
        v = door.get(key)
        if v is None:
            if R["missing_values"]["policy"] == "fail":
                out.append({"element": door["id"], "field": key, "rule": rule, "value": None, "limit": limit,
                            "text": f"{door['id']}: {key} missing"})
            continue
        if not ok(v):
            out.append({"element": door["id"], "field": key, "rule": rule, "value": v, "limit": limit,
                        "text": f"{door['id']}: {key} = {v:g} mm, needs {limit} ({rule})"})
    return out


def room_failures(room: dict, R: dict) -> list[dict]:
    v = room.get("turning_space_diameter_mm")
    lim = R["room"]["turning_diameter_min_mm"]
    if v is None:
        if R["missing_values"]["policy"] == "fail":
            return [{"element": room["id"], "field": "turning_space_diameter_mm", "rule": "ADA 304.3.1",
                     "value": None, "limit": f">= {lim}", "text": f"{room['id']}: turning space missing"}]
        return []
    if v < lim:
        return [{"element": room["id"], "field": "turning_space_diameter_mm", "rule": "ADA 304.3.1",
                 "value": v, "limit": f">= {lim}",
                 "text": f"{room['id']}: turning space {v:g} mm < {lim} mm (ADA 304.3.1)"}]
    return []


# ------------------------------------------------------------------ geometry helpers
def door_side(door: dict, room: dict, R: dict) -> str | None:
    """Which wall of the room the door is on: 'S', 'N', 'W', 'E', or None if unknown."""
    try:
        x, y = float(door["x_mm"]), float(door["y_mm"])
        x0, y0 = float(room["x_mm"]), float(room["y_mm"])
        x1, y1 = x0 + float(room["width_mm"]), y0 + float(room["length_mm"])
    except (KeyError, TypeError, ValueError):
        return None
    tol = R["passage"]["door_wall_tolerance_mm"]
    cand = []
    if x0 - tol <= x <= x1 + tol:
        cand += [("S", abs(y - y0)), ("N", abs(y - y1))]
    if y0 - tol <= y <= y1 + tol:
        cand += [("W", abs(x - x0)), ("E", abs(x - x1))]
    cand = [c for c in cand if c[1] <= tol]
    return min(cand, key=lambda c: c[1])[0] if cand else None


def room_depth(door: dict, room: dict, R: dict) -> float | None:
    """Depth of the room perpendicular to the door's wall (None if the wall is unknown)."""
    wall = door_side(door, room, R)
    if wall is None:
        return None
    v = room.get("length_mm") if wall in ("S", "N") else room.get("width_mm")
    return float(v) if v is not None else None


def not_checked(building: dict, R: dict) -> list[dict]:
    """Rules that could not be checked because their value is missing (policy "not_checked")."""
    if R["missing_values"]["policy"] == "fail":
        return []
    out = [{"element": d["id"], "rule": rule, "field": k}
           for d in building["doors"] for k, rule in DOOR_FIELD_RULE.items() if d.get(k) is None]
    out += [{"element": r["id"], "rule": ROOM_RULE, "field": ROOM_FIELD}
            for r in building["rooms"] if r.get(ROOM_FIELD) is None]
    return out


def can_pass(room: dict, door_in: dict, door_out: dict, R: dict) -> tuple[bool, dict | None]:
    """Can a wheelchair go through `room`, entering by door_in and leaving by door_out?
    Returns (ok, cause); cause is a failure dict like door_failures() gives."""
    rf = room_failures(room, R)
    if not rf:
        return True, None
    s_in, s_out = door_side(door_in, room, R), door_side(door_out, room, R)
    turn = rf[0]
    if s_in is None or s_out is None:
        return False, {**turn, "text": f"{turn['text']}; door position unknown, a turn is assumed (conservative)"}
    if (s_in, s_out) not in OPPOSITE:
        return False, {**turn, "text": f"{turn['text']}; {door_in['id']} and {door_out['id']} are on adjacent "
                                       f"walls, so a turn is needed"}
    across = room["width_mm"] if s_in in ("S", "N") else room["length_mm"]
    lim = R["passage"]["straight_min_width_mm"]
    if across is None or across < lim:
        return False, {"element": room["id"], "field": "width_mm", "rule": "ADA 403.5.1", "value": across,
                       "limit": f">= {lim}", "text": f"{turn['text']}; straight passage through {room['id']} "
                                                    f"narrower than {lim} mm (ADA 403.5.1)"}
    return True, None


# ------------------------------------------------------------------ main entry point
def evaluate(building: dict, R: dict | None = None) -> dict:
    """
    Returns {
      "verdict":  {"label": "accessible" | "not_accessible", "failing_rules": [...], "warnings": [...],
                   "unreachable_rooms": [...]},            # building level = THE OUTPUT
      "labels":   {room_id: "reachable" | "not_reachable"},  # room level, internal (route agent target)
      "reasons":  {room_id: [text, ...]},     # why a room is not reachable
      "causes":   {room_id: [failure dict, ...]},  # same, structured
      "warnings": {room_id: [text, ...]},     # e.g. reachable but cannot turn around
      "door_failures": {door_id: [failure dict, ...]},
      "room_failures": {room_id: [failure dict, ...]},
    }
    """
    R = R or load_rules()
    rooms = {r["id"]: r for r in building["rooms"]}
    doors = {d["id"]: d for d in building["doors"]}
    dfail = {did: door_failures(d, R) for did, d in doors.items()}
    by_room = {rid: [] for rid in rooms} | {EXTERIOR: []}
    for d in doors.values():
        a, b = d["connects"]
        if a in by_room and b in by_room:
            by_room[a].append(d)
            by_room[b].append(d)

    blocked: dict[str, list[dict]] = {rid: [] for rid in rooms}
    seen, reached = set(), set()
    q = deque()
    # start: entrance doors
    for d in by_room[EXTERIOR]:
        tgt = d["connects"][1] if d["connects"][0] == EXTERIOR else d["connects"][0]
        if dfail[d["id"]]:
            blocked[tgt] += dfail[d["id"]]
        else:
            q.append((tgt, d["id"]))
    # search over (room, door we came in through) so "do we need to turn?" can be answered
    while q:
        rid, din = q.popleft()
        if (rid, din) in seen:
            continue
        seen.add((rid, din))
        reached.add(rid)
        for d in by_room[rid]:
            if d["id"] == din:
                continue
            tgt = d["connects"][1] if d["connects"][0] == rid else d["connects"][0]
            if tgt == EXTERIOR:
                continue
            if dfail[d["id"]]:
                blocked[tgt] += dfail[d["id"]]
                continue
            ok, cause = can_pass(rooms[rid], doors[din], d, R)
            if not ok:
                blocked[tgt].append({**cause, "text": f"cannot pass through {rid}: {cause['text']}"})
                continue
            q.append((tgt, d["id"]))

    labels, reasons, causes, warnings, rfail = {}, {}, {}, {}, {}
    for rid, room in rooms.items():
        labels[rid] = REACHABLE if rid in reached else NOT_REACHABLE
        rf = rfail[rid] = room_failures(room, R)
        warnings[rid] = [f"{f['text']}: reachable but cannot turn around" for f in rf] if rid in reached else \
                        [f["text"] for f in rf]
        if rid in reached:
            reasons[rid], causes[rid] = [], []
        else:
            causes[rid] = _unique(blocked[rid])
            reasons[rid] = [c["text"] for c in causes[rid]] or \
                ["no entrance door found (no door connects a room to EXTERIOR)" if not by_room[EXTERIOR] else
                 "no accessible path from the entrance (all neighbouring rooms are unreachable)"]
    result = {"labels": labels, "reasons": reasons, "causes": causes, "warnings": warnings,
              "door_failures": dfail, "room_failures": rfail}
    result["verdict"] = building_verdict(result, R)
    result["verdict"]["not_checked"] = not_checked(building, R)
    return result


def _unique(fails: list[dict]) -> list[dict]:
    seen, out = set(), []
    for f in fails:
        k = (f["element"], f["rule"], f["field"])
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def building_verdict(result: dict, R: dict) -> dict:
    """Room reachability -> building verdict (accessible / not_accessible + failing rules)."""
    unreachable = [rid for rid, lab in result["labels"].items() if lab == NOT_REACHABLE]
    failing = _unique([c for rid in unreachable for c in result["causes"][rid]])
    used = {(f["element"], f["rule"], f["field"]) for f in failing}
    # failures that block nothing: rooms that cannot turn around, failing doors off every path
    other = [f for fs in list(result["room_failures"].values()) + list(result["door_failures"].values())
             for f in fs if (f["element"], f["rule"], f["field"]) not in used]
    other = _unique(other)
    if R.get("verdict", {}).get("warnings_make_not_accessible") and other:
        failing, other = _unique(failing + other), []
    # every room must be reachable: an unreachable room makes the building not accessible even
    # when no ADA rule is to blame (no entrance door, a room without doors)
    return {"label": NOT_ACCESSIBLE if unreachable or failing else ACCESSIBLE,
            "failing_rules": [{k: f[k] for k in ("element", "rule", "field", "value", "limit", "text")}
                              for f in failing],
            "warnings": [f["text"] for f in other],
            "unreachable_rooms": unreachable}


def make_labels(building: dict, R: dict | None = None) -> dict:
    """All training targets of one building, written in the VERIFY step and stored with the
    record (data/llm/*_verified.jsonl). The GNN agents and the main agent learn / are scored
    against these; nothing in the accessibility check calls the rule engine."""
    R = R or load_rules()
    res = evaluate(building, R)
    return {
        "verdict": {"label": res["verdict"]["label"],
                    "failing_rules": [{"element": f["element"], "rule": f["rule"]}
                                      for f in res["verdict"]["failing_rules"]],
                    "warnings": res["verdict"]["warnings"]},
        # door agent: one 0/1 per door rule (1 = fails)
        "door_rule_fails": {did: {rule: int(any(f["rule"] == rule for f in fs)) for rule in DOOR_RULES}
                            for did, fs in res["door_failures"].items()},
        # room agent: 1 = turning space fails
        "room_turning_fails": {rid: int(bool(fs)) for rid, fs in res["room_failures"].items()},
        # route agent: 1 = room reachable from the entrance
        "room_reachable": {rid: int(lab == REACHABLE) for rid, lab in res["labels"].items()},
    }
