"""
Building dict (see rules.py) -> IFC file, with IfcOpenShell.

What is written:
- IfcProject > IfcSite > IfcBuilding > IfcBuildingStorey (units: metres)
- one IfcWall per unique room edge (shared edges are written once), centred on the edge
- for every door: IfcOpeningElement in its wall + IfcDoor filling it
- one IfcSpace per room (named R1, R2, ... no room types)
- property sets:
    door  -> ADA_Accessibility (ClearWidth_mm, ThresholdHeight_mm, HardwareHeight_mm,
             HardwareType, ManeuveringClearancePull_mm, ManeuveringClearancePush_mm),
             Pset_DoorCommon (IsExternal)
    room  -> ADA_Accessibility (TurningSpaceDiameter_mm), Pset_SpaceCommon (IsExternal),
             Qto_SpaceBaseQuantities (NetFloorArea, Height)
- IfcRelSpaceBoundary for every door-room and wall-room pair (this is how the
  reader knows which rooms a door connects)

Deliberately NOT written: yes/no answer fields such as HandicapAccessible or
ADA_Compliant. They would leak the label into the GNN input.
"""
from __future__ import annotations

import numpy as np
import ifcopenshell
import ifcopenshell.api
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.owner
import ifcopenshell.api.owner.settings
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import ifcopenshell.guid
from ifcopenshell.util.shape_builder import ShapeBuilder

WALL_T = 0.20   # wall thickness (m)
H = 3.00        # storey height (m)
DOOR_H = 2.10   # door height (m)
FRAME = 0.09    # frame allowance added to the clear width for the geometric opening (m)
EXTERIOR = "EXTERIOR"


def _matrix(x, y, z=0.0, rot_deg=0.0):
    m = np.eye(4)
    a = np.radians(rot_deg)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
    m[:3, 3] = [x, y, z]
    return m


def _edges(room):
    """Room edges in metres: (orientation, fixed coordinate, start, end)."""
    x0, y0 = room["x_mm"] / 1000, room["y_mm"] / 1000
    x1, y1 = x0 + room["width_mm"] / 1000, y0 + room["length_mm"] / 1000
    return [("h", round(y0, 3), round(x0, 3), round(x1, 3)), ("h", round(y1, 3), round(x0, 3), round(x1, 3)),
            ("v", round(x0, 3), round(y0, 3), round(y1, 3)), ("v", round(x1, 3), round(y0, 3), round(y1, 3))]


def write_ifc(building: dict, path: str, schema: str = "IFC4", name: str = "ADA synthetic building") -> str:
    model = ifcopenshell.api.project.create_file(version=schema)
    if schema.upper() == "IFC2X3":  # IFC2x3 requires an owner history (who created the file)
        person = ifcopenshell.api.owner.add_person(model, identification="ADA", family_name="Generator", given_name="ADA")
        org = ifcopenshell.api.owner.add_organisation(model, identification="GTU", name="Gebze Technical University")
        user = ifcopenshell.api.owner.add_person_and_organisation(model, person=person, organisation=org)
        app = ifcopenshell.api.owner.add_application(model)
        ifcopenshell.api.owner.settings.get_user = lambda f: user
        ifcopenshell.api.owner.settings.get_application = lambda f: app
    project = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name=name)
    ifcopenshell.api.unit.assign_unit(  # metres; the default would be millimetres
        model,
        length={"is_metric": True, "raw": "METERS"},
        area={"is_metric": True, "raw": "SQUARE_METERS"},
        volume={"is_metric": True, "raw": "CUBIC_METERS"},
    )
    ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(model, context_type="Model", context_identifier="Body",
                                                target_view="MODEL_VIEW", parent=ctx)
    site = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name="Site")
    bldg = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuilding", name=building.get("building_id", "Building"))
    storey = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuildingStorey", name="Ground floor")
    ifcopenshell.api.aggregate.assign_object(model, relating_object=project, products=[site])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=site, products=[bldg])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=bldg, products=[storey])
    for p in (site, bldg, storey):
        ifcopenshell.api.geometry.edit_object_placement(model, product=p)

    def pset(product, pname, props):
        ifcopenshell.api.pset.edit_pset(model, pset=ifcopenshell.api.pset.add_pset(model, product=product, name=pname),
                                        properties=props)

    # ---------- walls (one per unique edge)
    walls = {}  # edge -> IfcWall
    exterior_edges = {}
    for room in building["rooms"]:
        for e in _edges(room):
            exterior_edges[e] = exterior_edges.get(e, 0) + 1
    for i, e in enumerate(exterior_edges):
        ori, c, s, t = e
        w = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall", name=f"W{i + 1}", predefined_type="SOLIDWALL")
        rep = ifcopenshell.api.geometry.add_wall_representation(model, context=body, length=t - s, height=H, thickness=WALL_T)
        ifcopenshell.api.geometry.assign_representation(model, product=w, representation=rep)
        m = _matrix(s, c - WALL_T / 2) if ori == "h" else _matrix(c + WALL_T / 2, s, 0, 90)
        ifcopenshell.api.geometry.edit_object_placement(model, product=w, matrix=m)
        ifcopenshell.api.spatial.assign_container(model, relating_structure=storey, products=[w])
        pset(w, "Pset_WallCommon", {"IsExternal": exterior_edges[e] == 1})
        walls[e] = w

    # ---------- spaces
    builder = ShapeBuilder(model)
    spaces = {}
    for room in building["rooms"]:
        w_m, l_m = room["width_mm"] / 1000, room["length_mm"] / 1000
        sp = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSpace", name=room["id"], predefined_type="INTERNAL")
        solid = builder.extrude(builder.rectangle(size=np.array([w_m, l_m])), magnitude=H)
        ifcopenshell.api.geometry.assign_representation(model, product=sp, representation=builder.get_representation(body, [solid]))
        ifcopenshell.api.geometry.edit_object_placement(model, product=sp, matrix=_matrix(room["x_mm"] / 1000, room["y_mm"] / 1000))
        ifcopenshell.api.aggregate.assign_object(model, relating_object=storey, products=[sp])
        pset(sp, "Pset_SpaceCommon", {"IsExternal": False})
        if room.get("turning_space_diameter_mm") is not None:
            pset(sp, "ADA_Accessibility", {"TurningSpaceDiameter_mm": float(room["turning_space_diameter_mm"])})
        q = ifcopenshell.api.pset.add_qto(model, product=sp, name="Qto_SpaceBaseQuantities")
        ifcopenshell.api.pset.edit_qto(model, qto=q, properties={"NetFloorArea": round(w_m * l_m, 3), "Height": H})
        spaces[room["id"]] = sp

    def boundary(sp, element, kind):
        model.create_entity("IfcRelSpaceBoundary", GlobalId=ifcopenshell.guid.new(), Name=f"{sp.Name}-{element.Name}",
                            RelatingSpace=sp, RelatedBuildingElement=element,
                            PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary=kind)

    for room in building["rooms"]:
        for e in _edges(room):
            boundary(spaces[room["id"]], walls[e], "EXTERNAL" if exterior_edges[e] == 1 else "INTERNAL")

    # ---------- doors
    key_map = {"clear_width_mm": "ClearWidth_mm", "threshold_height_mm": "ThresholdHeight_mm",
               "hardware_height_mm": "HardwareHeight_mm", "maneuvering_pull_mm": "ManeuveringClearancePull_mm",
               "maneuvering_push_mm": "ManeuveringClearancePush_mm"}
    for d in building["doors"]:
        x, y = d["x_mm"] / 1000, d["y_mm"] / 1000
        # host wall = the edge the door centre lies on
        host, best = None, 1e9
        for e, w in walls.items():
            ori, c, s, t = e
            dist = abs(y - c) if ori == "h" else abs(x - c)
            along = x if ori == "h" else y
            if s - 0.01 <= along <= t + 0.01 and dist < best:
                host, best, host_e = w, dist, e
        if host is None or best > 0.45:
            raise ValueError(f"door {d['id']} is not on any wall")
        ori = host_e[0]
        dw = (d.get("clear_width_mm") or 900) / 1000 + FRAME
        if ori == "h":
            op_m, d_m = _matrix(x - dw / 2, host_e[1] - WALL_T / 2 - 0.05), _matrix(x - dw / 2, host_e[1] - WALL_T / 2)
        else:
            op_m = _matrix(host_e[1] + WALL_T / 2 + 0.05, y - dw / 2, 0, 90)
            d_m = _matrix(host_e[1] + WALL_T / 2, y - dw / 2, 0, 90)
        op = ifcopenshell.api.root.create_entity(model, ifc_class="IfcOpeningElement", name=f"{d['id']} opening")
        op_rep = ifcopenshell.api.geometry.add_wall_representation(model, context=body, length=dw, height=DOOR_H, thickness=WALL_T + 0.10)
        ifcopenshell.api.geometry.assign_representation(model, product=op, representation=op_rep)
        ifcopenshell.api.geometry.edit_object_placement(model, product=op, matrix=op_m)
        ifcopenshell.api.feature.add_feature(model, feature=op, element=host)

        door = ifcopenshell.api.root.create_entity(model, ifc_class="IfcDoor", name=d["id"], predefined_type="DOOR")
        door.OverallWidth, door.OverallHeight = dw, DOOR_H
        rep = ifcopenshell.api.geometry.add_door_representation(model, context=body, overall_height=DOOR_H, overall_width=dw,
                                                                operation_type="SINGLE_SWING_LEFT")
        ifcopenshell.api.geometry.assign_representation(model, product=door, representation=rep)
        ifcopenshell.api.geometry.edit_object_placement(model, product=door, matrix=d_m)
        ifcopenshell.api.spatial.assign_container(model, relating_structure=storey, products=[door])
        ifcopenshell.api.feature.add_filling(model, opening=op, element=door)
        pset(door, "Pset_DoorCommon", {"IsExternal": EXTERIOR in d["connects"]})
        props = {ifc_k: float(d[k]) for k, ifc_k in key_map.items() if d.get(k) is not None}
        if d.get("hardware_type"):
            props["HardwareType"] = d["hardware_type"]
        pset(door, "ADA_Accessibility", props)
        for rid in d["connects"]:
            if rid != EXTERIOR:
                boundary(spaces[rid], door, "EXTERNAL" if EXTERIOR in d["connects"] else "INTERNAL")

    model.write(path)
    return path
