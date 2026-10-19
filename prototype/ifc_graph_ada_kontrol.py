"""
IFC -> Graph -> ADA erişilebilirlik kontrolü

Düğümler: IfcSpace'ler + "DIS_MEKAN" (bina dışı)
Kenarlar: İki mekânı bağlayan kapılar (IfcRelSpaceBoundary üzerinden bulunur)
Kontrol: Her kapı ADA kurallarına uyuyor mu? Uymayan kapılar grafikten çıkarılınca
         dışarıdan her odaya hâlâ ulaşılabiliyor mu?

Kullanım: python3 ifc_graph_ada_kontrol.py dosya.ifc
"""

import sys
import numpy as np
import networkx as nx
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ADA = {
    "min_clear_width_mm": 815,     # 404.2.3
    "max_threshold_mm": 13,        # 404.2.5
    "hardware_min_mm": 865,        # 404.2.7
    "hardware_max_mm": 1220,
    "turning_diameter_mm": 1525,   # 304.3.1
}

path = sys.argv[1] if len(sys.argv) > 1 else "ada_2oda_tekkat.ifc"
model = ifcopenshell.open(path)
settings = ifcopenshell.geom.settings()
settings.set("use-world-coords", True)


def bbox(product):
    shape = ifcopenshell.geom.create_shape(settings, product)
    v = np.array(shape.geometry.verts).reshape(-1, 3)
    return v.min(axis=0), v.max(axis=0)


# 1) Graph oluştur
G = nx.Graph()
G.add_node("DIS_MEKAN", label="Dis Mekan", ok=True)

space_info = {}
for s in model.by_type("IfcSpace"):
    lo, hi = bbox(s)
    w_mm, l_mm = (hi[0] - lo[0]) * 1000, (hi[1] - lo[1]) * 1000
    turning_ok = min(w_mm, l_mm) >= ADA["turning_diameter_mm"]
    space_info[s.Name] = (lo, hi)
    G.add_node(s.Name, label=s.LongName or s.Name, width_mm=w_mm, length_mm=l_mm, ok=turning_ok)

door_spaces = {}
for rel in model.by_type("IfcRelSpaceBoundary"):
    el = rel.RelatedBuildingElement
    if el and el.is_a("IfcDoor"):
        door_spaces.setdefault(el, set()).add(rel.RelatingSpace.Name)

report = []
for d, spaces in door_spaces.items():
    spaces = sorted(spaces)
    if len(spaces) == 1:          # tek mekâna bağlı kapı = dış kapı
        spaces.append("DIS_MEKAN")
    ps = uel.get_psets(d)
    ada = ps.get("ADA_Accessibility", {})
    problems = []
    cw = ada.get("ClearWidth_mm")
    if cw is None or cw < ADA["min_clear_width_mm"]:
        problems.append(f"net genislik {cw} mm < {ADA['min_clear_width_mm']}")
    th = ada.get("ThresholdHeight_mm")
    if th is None or th > ADA["max_threshold_mm"]:
        problems.append(f"esik {th} mm > {ADA['max_threshold_mm']}")
    hh = ada.get("HardwareHeight_mm")
    if hh is None or not (ADA["hardware_min_mm"] <= hh <= ADA["hardware_max_mm"]):
        problems.append(f"kol yuksekligi {hh} mm aralik disi")
    G.add_edge(*spaces, door=d.Name, ok=not problems, problems=problems)
    report.append((d.Name, spaces, problems))

# 2) Erişilebilirlik: uymayan kapıları çıkar, dışarıdan ulaşılabilen odaları bul
A = nx.Graph()
A.add_nodes_from(G.nodes)
A.add_edges_from((u, v) for u, v, e in G.edges(data=True) if e["ok"])
reachable = nx.node_connected_component(A, "DIS_MEKAN")

print(f"\nDosya: {path}")
print(f"Graph: {G.number_of_nodes()} dugum, {G.number_of_edges()} kenar\n")
print("KAPILAR (kenarlar):")
for name, sp, prob in report:
    print(f"  {name:22s} {sp[0]} <-> {sp[1]:10s} {'UYGUN' if not prob else 'UYGUN DEGIL: ' + '; '.join(prob)}")
print("\nODALAR (dugumler):")
all_ok = True
for n, data in G.nodes(data=True):
    if n == "DIS_MEKAN":
        continue
    r = n in reachable
    t = data["ok"]
    all_ok &= r and t
    print(f"  {n:6s} {data['label']:24s} {data['width_mm']:.0f}x{data['length_mm']:.0f} mm  "
          f"donus alani: {'OK' if t else 'YETERSIZ'}  disaridan erisim: {'VAR' if r else 'YOK'}")
    if r:
        print(f"         rota: {' -> '.join(nx.shortest_path(A, 'DIS_MEKAN', n))}")
print(f"\nSONUC: Bina ADA kontrollerine gore {'ERISILEBILIR' if all_ok else 'ERISILEBILIR DEGIL'}\n")

# 3) Görsel: plan + graph
fig, ax = plt.subplots(figsize=(9, 6))
for w in model.by_type("IfcWall"):
    lo, hi = bbox(w)
    ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], color="#444"))
for name, (lo, hi) in space_info.items():
    ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], color="#dbe9f6"))
    c = (lo[:2] + hi[:2]) / 2
    ax.add_patch(Circle(c, ADA["turning_diameter_mm"] / 2000, fill=False, ls="--", color="#2a7"))
pos = {n: (lo[:2] + hi[:2]) / 2 for n, (lo, hi) in space_info.items()}
for d in door_spaces:
    lo, hi = bbox(d)
    ax.add_patch(Rectangle(lo[:2], *np.maximum((hi - lo)[:2], 0.05), color="#e8a33d"))
    if d.Name.startswith("D1"):
        pos["DIS_MEKAN"] = np.array([(lo[0] + hi[0]) / 2, -1.0])
for u, v, e in G.edges(data=True):
    ax.plot(*zip(pos[u], pos[v]), color="#2a7" if e["ok"] else "#d33", lw=2.5, zorder=3)
    mid = (pos[u] + pos[v]) / 2
    ax.text(*mid, e["door"].split()[0], fontsize=9, ha="center", va="bottom", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1))
for n, p in pos.items():
    ax.scatter(*p, s=320, color="#1f4e79", zorder=4)
    ax.text(p[0], p[1] - 0.45, G.nodes[n]["label"], ha="center", fontsize=9, zorder=5)
ax.set_xlim(-0.8, 10)
ax.set_ylim(-1.8, 5.8)
ax.set_aspect("equal")
ax.set_title("Plan + erisilebilirlik graph'i (yesil = ADA uygun, kesikli = 1525 mm donus alani)", fontsize=10)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.tight_layout()
plt.savefig("ada_plan_graph.png", dpi=150)
print("Gorsel: ada_plan_graph.png")
