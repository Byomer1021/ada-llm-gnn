"""
ADA standartlarına uygun, tek katlı, 2 odalı örnek IFC (IFC4) üretici.

Plan (metre, duvar kalınlığı 0.20 m, kat yüksekliği 3.00 m):

    y=5.2 +-------------------------------+-----------------+
          |                               |                 |
          |   ODA 1  (Giriş / Oturma)      D2   ODA 2        |
          |   5.00 x 4.80 m               |  (Yatak/Ofis)   |
          |                               |  3.60 x 4.80 m  |
    y=0   +--------D1---------------------+-----------------+
          x=0                            x=5.4             x=9.2
                  ^ dış giriş kapısı

ADA 2010 kontrolleri (kullanılan değerler):
  - 404.2.3 Kapı net geçiş genişliği   >= 815 mm (32")   -> 915 mm (36")
  - 404.2.5 Eşik yüksekliği             <= 13 mm (1/2")   -> 0 mm
  - 404.2.7 Kapı kolu yüksekliği        865–1220 mm       -> 1000 mm, kol (lever) tipi
  - 404.2.4 Çekme tarafı manevra alanı  >= 1525 mm (60")  -> oda derinliği yeterli
  - 304.3.1 Dönüş alanı (çember)        >= 1525 mm (60")  -> her odada sağlanıyor
  - 403.3   Zemin eğimi                 <= 1:48           -> düz zemin (0)
"""

import numpy as np
import ifcopenshell
import ifcopenshell.api
import ifcopenshell.api.root
import ifcopenshell.api.unit
import ifcopenshell.api.context
import ifcopenshell.api.project
import ifcopenshell.api.aggregate
import ifcopenshell.api.spatial
import ifcopenshell.api.geometry
import ifcopenshell.api.feature
import ifcopenshell.api.pset
from ifcopenshell.util.shape_builder import ShapeBuilder

WALL_T = 0.20      # duvar kalınlığı (m)
H = 3.00           # kat yüksekliği (m)
DOOR_W = 1.00      # kapı boşluğu (kasa dahil, m)
DOOR_H = 2.10      # kapı yüksekliği (m)
CLEAR_W_MM = 915   # net geçiş genişliği (36")

model = ifcopenshell.api.project.create_file(version="IFC4")
project = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name="ADA Ornek Proje")
ifcopenshell.api.unit.assign_unit(  # uzunluk birimi: metre
    model,
    length={"is_metric": True, "raw": "METERS"},
    area={"is_metric": True, "raw": "SQUARE_METERS"},
    volume={"is_metric": True, "raw": "CUBIC_METERS"},
)

ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
body = ifcopenshell.api.context.add_context(
    model, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=ctx
)

site = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name="Arsa")
building = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuilding", name="ADA Uyumlu Konut")
storey = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuildingStorey", name="Zemin Kat")
ifcopenshell.api.aggregate.assign_object(model, relating_object=project, products=[site])
ifcopenshell.api.aggregate.assign_object(model, relating_object=site, products=[building])
ifcopenshell.api.aggregate.assign_object(model, relating_object=building, products=[storey])
for p in (site, building, storey):
    ifcopenshell.api.geometry.edit_object_placement(model, product=p)


def matrix(x, y, z=0.0, rot_deg=0.0):
    m = np.eye(4)
    a = np.radians(rot_deg)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
    m[:3, 3] = [x, y, z]
    return m


def wall(name, x, y, length, rot):
    w = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWall", name=name, predefined_type="SOLIDWALL")
    rep = ifcopenshell.api.geometry.add_wall_representation(model, context=body, length=length, height=H, thickness=WALL_T)
    ifcopenshell.api.geometry.assign_representation(model, product=w, representation=rep)
    ifcopenshell.api.geometry.edit_object_placement(model, product=w, matrix=matrix(x, y, 0, rot))
    ifcopenshell.api.spatial.assign_container(model, relating_structure=storey, products=[w])
    ifcopenshell.api.pset.edit_pset(
        model, pset=ifcopenshell.api.pset.add_pset(model, product=w, name="Pset_WallCommon"),
        properties={"IsExternal": name.startswith("Dis"), "LoadBearing": name.startswith("Dis")},
    )
    return w


# Duvarlar (yerel X = duvar boyu, yerel Y = kalınlık)
w_south = wall("Dis Duvar Guney", 0.0, 0.0, 9.2, 0)
w_north = wall("Dis Duvar Kuzey", 0.0, 5.0, 9.2, 0)
w_west = wall("Dis Duvar Bati", 0.2, 0.2, 4.8, 90)
w_east = wall("Dis Duvar Dogu", 9.2, 0.2, 4.8, 90)
w_part = wall("Ara Duvar", 5.4, 0.2, 4.8, 90)


def door(name, host, open_matrix, door_matrix, external):
    # Duvarda boşluk (IfcOpeningElement)
    op = ifcopenshell.api.root.create_entity(model, ifc_class="IfcOpeningElement", name=f"{name} Boslugu")
    op_rep = ifcopenshell.api.geometry.add_wall_representation(model, context=body, length=DOOR_W, height=DOOR_H, thickness=WALL_T + 0.10)
    ifcopenshell.api.geometry.assign_representation(model, product=op, representation=op_rep)
    ifcopenshell.api.geometry.edit_object_placement(model, product=op, matrix=open_matrix)
    ifcopenshell.api.feature.add_feature(model, feature=op, element=host)

    # Kapı
    d = ifcopenshell.api.root.create_entity(model, ifc_class="IfcDoor", name=name, predefined_type="DOOR")
    d.OverallWidth, d.OverallHeight = DOOR_W, DOOR_H
    d.OperationType = "SINGLE_SWING_LEFT"
    d_rep = ifcopenshell.api.geometry.add_door_representation(
        model, context=body, overall_height=DOOR_H, overall_width=DOOR_W, operation_type="SINGLE_SWING_LEFT"
    )
    ifcopenshell.api.geometry.assign_representation(model, product=d, representation=d_rep)
    ifcopenshell.api.geometry.edit_object_placement(model, product=d, matrix=door_matrix)
    ifcopenshell.api.spatial.assign_container(model, relating_structure=storey, products=[d])
    ifcopenshell.api.feature.add_filling(model, opening=op, element=d)

    ifcopenshell.api.pset.edit_pset(
        model, pset=ifcopenshell.api.pset.add_pset(model, product=d, name="Pset_DoorCommon"),
        properties={"IsExternal": external, "HandicapAccessible": True},
    )
    ifcopenshell.api.pset.edit_pset(
        model, pset=ifcopenshell.api.pset.add_pset(model, product=d, name="ADA_Accessibility"),
        properties={
            "ClearWidth_mm": float(CLEAR_W_MM),
            "ThresholdHeight_mm": 0.0,
            "HardwareHeight_mm": 1000.0,
            "HardwareType": "Lever",
            "ManeuveringClearancePull_mm": 1525.0,
            "ManeuveringClearancePush_mm": 1220.0,
            "AutomaticDoor": False,
        },
    )
    return d


# D1: güney duvarında dış giriş kapısı (x = 2.0 – 3.0)
d1 = door("D1 Giris Kapisi", w_south, matrix(2.0, -0.05), matrix(2.0, 0.0), external=True)
# D2: ara duvarda oda geçiş kapısı (y = 2.0 – 3.0); duvar 90° döndüğü için kalınlık -X yönünde
d2 = door("D2 Oda Gecis Kapisi", w_part, matrix(5.45, 2.0, 0, 90), matrix(5.4, 2.0, 0, 90), external=False)

# Mekânlar (IfcSpace)
builder = ShapeBuilder(model)


def space(name, long_name, x, y, w, l):
    s = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSpace", name=name, predefined_type="INTERNAL")
    s.LongName = long_name
    profile = builder.rectangle(size=np.array([w, l]))
    solid = builder.extrude(profile, magnitude=H)
    rep = builder.get_representation(body, [solid])
    ifcopenshell.api.geometry.assign_representation(model, product=s, representation=rep)
    ifcopenshell.api.geometry.edit_object_placement(model, product=s, matrix=matrix(x, y))
    ifcopenshell.api.aggregate.assign_object(model, relating_object=storey, products=[s])
    ifcopenshell.api.pset.edit_pset(
        model, pset=ifcopenshell.api.pset.add_pset(model, product=s, name="Pset_SpaceCommon"),
        properties={"IsExternal": False, "HandicapAccessible": True},
    )
    ifcopenshell.api.pset.edit_pset(
        model, pset=ifcopenshell.api.pset.add_pset(model, product=s, name="ADA_Accessibility"),
        properties={
            "TurningSpaceDiameter_mm": 1525.0,
            "FloorSlope": "flat (0)",
            "FloorSurface": "firm, stable, slip-resistant",
        },
    )
    q = ifcopenshell.api.pset.add_qto(model, product=s, name="Qto_SpaceBaseQuantities")
    ifcopenshell.api.pset.edit_qto(model, qto=q, properties={"NetFloorArea": w * l, "Height": H})
    return s


s1 = space("ODA-1", "Giris / Oturma Odasi", 0.2, 0.2, 5.0, 4.8)
s2 = space("ODA-2", "Yatak Odasi / Ofis", 5.4, 0.2, 3.6, 4.8)


# Mekân sınırları: hangi kapı hangi mekânları bağlıyor (graph için anahtar bilgi)
def boundary(sp, element, internal_or_external):
    model.create_entity(
        "IfcRelSpaceBoundary",
        GlobalId=ifcopenshell.guid.new(),
        Name=f"{sp.Name}-{element.Name}",
        RelatingSpace=sp,
        RelatedBuildingElement=element,
        PhysicalOrVirtualBoundary="PHYSICAL",
        InternalOrExternalBoundary=internal_or_external,
    )


boundary(s1, d1, "EXTERNAL")
boundary(s1, d2, "INTERNAL")
boundary(s2, d2, "INTERNAL")
for sp, ws in ((s1, [w_south, w_north, w_west, w_part]), (s2, [w_south, w_north, w_east, w_part])):
    for w in ws:
        boundary(sp, w, "EXTERNAL" if w.Name.startswith("Dis") else "INTERNAL")

out = "ada_2oda_tekkat.ifc"
model.write(out)
print("Yazildi:", out)
