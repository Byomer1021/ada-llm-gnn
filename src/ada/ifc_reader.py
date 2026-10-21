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

Real files carry no ADA property set. For them `estimate_missing()` (config/derive.yaml)
estimates what the standard IFC data allows: clear width from IfcDoor.OverallWidth and
the lining / panel properties, maneuvering clearances from the room depths, turning space
from the room's shorter side. Estimated values are listed per element in "estimated".
Anything still missing stays None and is listed in "issues"; its rule is then "not checked"
(config/ada_rules.yaml: missing_values.policy).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel
import ifcopenshell.util.unit as uunit
import numpy as np
import yaml

from .rules import load_rules, room_depth

EXTERIOR = "EXTERIOR"
DERIVE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "derive.yaml"

# property names we accept (first match wins); add aliases here for real files
DOOR_KEYS = {
    "clear_width_mm": ["ClearWidth_mm", "ClearWidth"],
    "threshold_height_mm": ["ThresholdHeight_mm", "ThresholdHeight"],
    "hardware_height_mm": ["HardwareHeight_mm", "HandleHeight_mm", "HardwareHeight"],
    "maneuvering_pull_mm": ["ManeuveringClearancePull_mm"],
    "maneuvering_push_mm": ["ManeuveringClearancePush_mm"],
}
ROOM_KEYS = {"turning_space_diameter_mm": ["TurningSpaceDiameter_mm", "TurningDiameter_mm"]}


def _unique_ids(names: list[str]) -> list[str]:
    """IFC Names need not be unique (ArchiCAD example: 14 of 77 doors named 'Tür-005'). Repeated names
    get #1, #2, ... in file order, so no door or room silently replaces another one."""
    count, seen, out = Counter(names), Counter(), []
    for n in names:
        if count[n] > 1:
            seen[n] += 1
            out.append(f"{n}#{seen[n]}")
        else:
            out.append(n)
    return out


def _find(psets: dict, names: list[str]):
    for props in psets.values():
        for n in names:
            if n in props and props[n] is not None:
                return props[n]
    return None


def load_derive(path: str | Path = DERIVE_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _predefined(element, cls: str) -> list:
    """Predefined property sets (e.g. IfcDoorLiningProperties) of an element and of its type."""
    out = []
    t = uel.get_type(element)
    for p in (getattr(t, "HasPropertySets", None) or []) if t else []:
        if p.is_a(cls):
            out.append(p)
    for rel in getattr(element, "IsDefinedBy", None) or []:
        p = getattr(rel, "RelatingPropertyDefinition", None)
        if p is not None and not isinstance(p, tuple) and p.is_a(cls):
            out.append(p)
    return out


def _ifc_door_data(door, to_mm: float) -> dict:
    """Standard IFC door data that the estimates use (lengths in mm, panel shares as 0..1)."""
    t = uel.get_type(door)
    op = getattr(door, "OperationType", None) or (getattr(t, "OperationType", None) if t else None)
    lin = _predefined(door, "IfcDoorLiningProperties")
    pan = _predefined(door, "IfcDoorPanelProperties")
    mm = lambda v: float(v) * to_mm if v is not None else None
    shares = [float(p.PanelWidth) for p in pan if p.PanelWidth is not None and 0 < float(p.PanelWidth) <= 1]
    return {"overall_width_mm": mm(getattr(door, "OverallWidth", None)),
            "lining_thickness_mm": next((mm(p.LiningThickness) for p in lin if p.LiningThickness), None),
            "threshold_mm": next((mm(p.ThresholdThickness) for p in lin if p.ThresholdThickness is not None), None),
            "panel_share": max(shares) if shares else None,
            "panel_depth_mm": next((mm(p.PanelDepth) for p in pan if p.PanelDepth), None),
            "operation": str(op) if op else None,
            "panel_operations": sorted({str(p.PanelOperation) for p in pan if p.PanelOperation})}


def estimate_clear_width(ifc: dict, D: dict) -> tuple[float | None, str]:
    """(clear width, how) from the standard door data; see config/derive.yaml."""
    ow = ifc.get("overall_width_mm")
    if not ow:
        return None, ""
    lining = ifc.get("lining_thickness_mm") or D["door"]["lining_thickness_mm"]
    op = (ifc.get("operation") or "").upper()
    share = ifc.get("panel_share") or (0.5 if op.startswith("DOUBLE") else 1.0)
    moves = " ".join([op] + ifc.get("panel_operations", [])).upper()
    sliding = "SLIDING" in moves or "FOLDING" in moves
    leaf = 0.0 if sliding else (ifc.get("panel_depth_mm") or D["door"]["leaf_thickness_mm"])
    v = (ow - 2 * lining) * share - leaf
    return round(v), (f"OverallWidth {ow:.0f} - 2 x lining {lining:.0f}, x leaf share {share:g}"
                      f"{'' if sliding else f', - open leaf {leaf:.0f}'}")


def estimate_missing(building: dict, R: dict | None = None, D: dict | None = None) -> dict:
    """Fill the ADA values a file lacks with estimates from standard IFC data and geometry.
    Changes `building` in place: values, element["estimated"] = {field: how}, issues."""
    R, D = R or load_rules(), D or load_derive()
    if not D.get("enabled", True):
        return building
    issues = building.setdefault("issues", [])
    rooms = {r["id"]: r for r in building["rooms"]}

    def put(el, field, value, how):
        el[field] = float(value)
        el.setdefault("estimated", {})[field] = how
        issues.append(f"{el['id']}: {field} estimated {value:.0f} mm ({how})")

    for r in building["rooms"]:
        if r.get("turning_space_diameter_mm") is None and D["room"]["turning_from_short_side"] \
                and r.get("width_mm") and r.get("length_mm"):
            put(r, "turning_space_diameter_mm", min(r["width_mm"], r["length_mm"]),
                "shorter side of the empty room, upper bound")
    for d in building["doors"]:
        ifc = d.get("ifc") or {}
        if d.get("clear_width_mm") is None:
            v, how = estimate_clear_width(ifc, D)
            if v is not None:
                put(d, "clear_width_mm", v, how)
        if d.get("threshold_height_mm") is None and ifc.get("threshold_mm") is not None:
            put(d, "threshold_height_mm", ifc["threshold_mm"], "IfcDoorLiningProperties.ThresholdThickness")
        if d.get("maneuvering_pull_mm") is None or d.get("maneuvering_push_mm") is None:
            depths = [D["maneuvering"]["exterior_depth_mm"] if s == EXTERIOR else
                      room_depth(d, rooms[s], R) if s in rooms else None for s in d["connects"]]
            if None not in depths:
                shallow, deep = sorted(depths)
                if d.get("maneuvering_pull_mm") is None:
                    put(d, "maneuvering_pull_mm", shallow, "depth of the shallower side (pull side unknown)")
                if d.get("maneuvering_push_mm") is None:
                    put(d, "maneuvering_push_mm", deep, "depth of the deeper side")
    return building


def _bbox_xy(settings, element):
    """World-coordinate footprint (xmin, ymin, xmax, ymax) in metres, or None."""
    try:
        v = np.array(ifcopenshell.geom.create_shape(settings, element).geometry.verts).reshape(-1, 3)
        return v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()
    except Exception:
        return None


def read_ifc(path: str, derive: bool = True) -> dict:
    """derive=False: property sets only, no estimates (e.g. to measure what a file carries)."""
    model = ifcopenshell.open(path)
    to_m = uunit.calculate_unit_scale(model)          # file length unit -> metres
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    issues = []

    rooms, space_by_id = [], {}
    spaces = model.by_type("IfcSpace")
    for sp, rid in zip(spaces, _unique_ids([sp.Name or sp.GlobalId for sp in spaces])):
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
        rooms.append(r)
        space_by_id[sp.id()] = rid

    # door -> spaces from space boundaries
    door_spaces: dict[int, set] = {}
    for rel in model.by_type("IfcRelSpaceBoundary"):
        el, sp = rel.RelatedBuildingElement, rel.RelatingSpace
        if el is not None and sp is not None and el.is_a("IfcDoor"):
            door_spaces.setdefault(el.id(), set()).add(space_by_id.get(sp.id()))

    doors = []
    ifc_doors = model.by_type("IfcDoor")
    for d, did in zip(ifc_doors, _unique_ids([d.Name or d.GlobalId for d in ifc_doors])):
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
        rec["hardware_type"] = _find(ps, ["HardwareType"])
        rec["ifc"] = _ifc_door_data(d, to_m * 1000)

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

    b = {"source": str(path), "unit_to_m": to_m, "rooms": rooms, "doors": doors, "issues": issues}
    if derive:
        estimate_missing(b)
    for el, keys in [(r, ROOM_KEYS) for r in rooms] + [(d, DOOR_KEYS) for d in doors]:
        for k, names in keys.items():
            if el.get(k) is None:
                issues.append(f"{el['id']}: {names[0]} not in the file")
    return b
