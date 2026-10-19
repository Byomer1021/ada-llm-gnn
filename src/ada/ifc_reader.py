"""
IFC file -> building dict (see rules.py). Works on our synthetic files and is
written to be tolerant with real files.

Values come from PROPERTY SETS (no geometric measurement of widths etc.).
Positions come from the file's own placement/geometry data (where a door and a
room are), which every IFC file that opens in a viewer contains. Positions are
only used to know which wall a door is on (needed for "is a turn needed?").

Which rooms a door connects:
1. IfcRelSpaceBoundary (door <-> space) if the file has it; one space = entrance door
2. otherwise: rooms whose footprint edge is near the door centre

Anything missing becomes None and is listed in "issues"; the rule engine then
treats it conservatively.
"""
from __future__ import annotations

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel
import ifcopenshell.util.unit as uunit
import numpy as np

EXTERIOR = "EXTERIOR"

# property names we accept (first match wins); add aliases here for real files
DOOR_KEYS = {
    "clear_width_mm": ["ClearWidth_mm", "ClearWidth"],
    "threshold_height_mm": ["ThresholdHeight_mm", "ThresholdHeight"],
    "hardware_height_mm": ["HardwareHeight_mm", "HandleHeight_mm", "HardwareHeight"],
    "maneuvering_pull_mm": ["ManeuveringClearancePull_mm"],
    "maneuvering_push_mm": ["ManeuveringClearancePush_mm"],
}
ROOM_KEYS = {"turning_space_diameter_mm": ["TurningSpaceDiameter_mm", "TurningDiameter_mm"]}


def _find(psets: dict, names: list[str]):
    for props in psets.values():
        for n in names:
            if n in props and props[n] is not None:
                return props[n]
    return None


def _bbox_xy(settings, element):
    """World-coordinate footprint (xmin, ymin, xmax, ymax) in metres, or None."""
    try:
        v = np.array(ifcopenshell.geom.create_shape(settings, element).geometry.verts).reshape(-1, 3)
        return v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()
    except Exception:
        return None


def read_ifc(path: str) -> dict:
    model = ifcopenshell.open(path)
    to_m = uunit.calculate_unit_scale(model)          # file length unit -> metres
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    issues = []

    rooms, space_by_id = [], {}
    for sp in model.by_type("IfcSpace"):
        rid = sp.Name or sp.GlobalId
        bb = _bbox_xy(s, sp)
        ps = uel.get_psets(sp)
        r = {"id": rid, "global_id": sp.GlobalId}
        if bb:
            r.update(x_mm=bb[0] * 1000, y_mm=bb[1] * 1000, width_mm=(bb[2] - bb[0]) * 1000, length_mm=(bb[3] - bb[1]) * 1000)
        else:
            r.update(x_mm=None, y_mm=None, width_mm=None, length_mm=None)
            issues.append(f"{rid}: no geometry, position unknown")
        for k, names in ROOM_KEYS.items():
            v = _find(ps, names)
            r[k] = float(v) if v is not None else None
            if v is None:
                issues.append(f"{rid}: {names[0]} missing")
        rooms.append(r)
        space_by_id[sp.id()] = rid

    # door -> spaces from space boundaries
    door_spaces: dict[int, set] = {}
    for rel in model.by_type("IfcRelSpaceBoundary"):
        el, sp = rel.RelatedBuildingElement, rel.RelatingSpace
        if el is not None and sp is not None and el.is_a("IfcDoor"):
            door_spaces.setdefault(el.id(), set()).add(space_by_id.get(sp.id()))

    doors = []
    for d in model.by_type("IfcDoor"):
        did = d.Name or d.GlobalId
        ps = uel.get_psets(d)
        bb = _bbox_xy(s, d)
        rec = {"id": did, "global_id": d.GlobalId}
        if bb:
            rec.update(x_mm=(bb[0] + bb[2]) / 2 * 1000, y_mm=(bb[1] + bb[3]) / 2 * 1000)
        else:
            rec.update(x_mm=None, y_mm=None)
            issues.append(f"{did}: no geometry, position unknown")
        for k, names in DOOR_KEYS.items():
            v = _find(ps, names)
            rec[k] = float(v) if v is not None else None
            if v is None:
                issues.append(f"{did}: {names[0]} missing")
        ht = _find(ps, ["HardwareType"])
        rec["hardware_type"] = ht

        conn = sorted(x for x in door_spaces.get(d.id(), set()) if x)
        if not conn and bb:  # fallback: rooms whose edge is close to the door centre
            cx, cy = rec["x_mm"], rec["y_mm"]
            for r in rooms:
                if r["x_mm"] is None:
                    continue
                x0, y0, x1, y1 = r["x_mm"], r["y_mm"], r["x_mm"] + r["width_mm"], r["y_mm"] + r["length_mm"]
                near = (x0 - 400 <= cx <= x1 + 400) and (y0 - 400 <= cy <= y1 + 400) and \
                       min(abs(cx - x0), abs(cx - x1), abs(cy - y0), abs(cy - y1)) <= 400
                if near:
                    conn.append(r["id"])
            issues.append(f"{did}: no space boundary, rooms found from positions")
        if len(conn) == 1:
            rec["connects"] = [EXTERIOR, conn[0]]
        elif len(conn) == 2:
            rec["connects"] = conn
        else:
            issues.append(f"{did}: could not determine connected rooms ({conn})")
            continue
        doors.append(rec)

    return {"source": str(path), "unit_to_m": to_m, "rooms": rooms, "doors": doors, "issues": issues}
