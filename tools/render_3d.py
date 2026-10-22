"""
Render a cut-away 3D picture of an IFC building (for the web UI, the report and
slides). Failing doors are red, others green.

Usage:
    python tools/render_3d.py data/demo/demo.ifc docs/figures/demo_3d.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom as G
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ada.ifc_reader import _storey, read_ifc  # noqa: E402
from src.ada.rules import evaluate  # noqa: E402

CUT = 1.25  # cut walls at this height (m) so the inside is visible
MAX_LABELS = 16  # more rooms + doors than this: only the failing doors are labelled
LIGHT = np.array([0.4, -0.7, 0.9]) / np.linalg.norm([0.4, -0.7, 0.9])


def render(ifc_path: str, out_png: str, elev: float = 42, azim: float = -60,
           failing: set[str] | None = None, building: dict | None = None) -> str:
    """failing = ids of the doors to draw red (ids as read_ifc gives them). None -> ask the rule
    engine (only for the report / slides; the web UI passes the main agent's failing elements).
    building = what read_ifc returned: only its rooms and doors (the checked storey) and the walls
    of that storey are drawn, labelled with its ids (repeated IFC Names get #1, #2 there)."""
    model = ifcopenshell.open(ifc_path)
    s = G.settings()
    s.set("use-world-coords", True)
    building = building or read_ifc(ifc_path)
    if failing is None:
        result = evaluate(building)
        failing = {d for d, f in result["door_failures"].items() if f}
    ids = {e["global_id"]: e["id"] for e in building["rooms"] + building["doors"] if e.get("global_id")}
    spaces = [e for e in model.by_type("IfcSpace") if e.GlobalId in ids]
    doors = [e for e in model.by_type("IfcDoor") if e.GlobalId in ids]
    walls = model.by_type("IfcWall")
    if building.get("storey", {}).get("storeys_in_file", 1) > 1:
        on = {_storey(e) for e in spaces + doors}
        walls = [w for w in walls if _storey(w) in on]

    def raw(e):
        g = G.create_shape(s, e).geometry
        return np.array(g.verts).reshape(-1, 3), np.array(g.faces).reshape(-1, 3)

    floor = min((raw(e)[0][:, 2].min() for e in spaces), default=0.0)

    def mesh(e):  # cut walls CUT above the storey's floor so the inside is visible
        v, f = raw(e)
        v[:, 2] = np.minimum(v[:, 2], floor + CUT)
        return v, f

    fig = plt.figure(figsize=(7, 5.6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    allv = []

    ax.computed_zorder = False  # draw floors, then all walls (depth-sorted together), then doors on top
    groups = {"floor": [], "wall": [], "door": []}

    def add(v, f, color, group):
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        k = 0.62 + 0.38 * np.abs(n @ LIGHT)
        cols = np.clip(np.array(mc.to_rgb(color))[None, :] * k[:, None], 0, 1)
        groups[group].append((tri, cols))
        allv.append(v)

    for e in spaces:
        v, f = mesh(e)
        z0 = v[:, 2].min()
        keep = np.array([t for t in f if np.allclose(v[t][:, 2], z0)])
        v2 = v.copy()
        v2[:, 2] = z0 + 0.01
        if len(keep):
            add(v2, keep, "#E3EEF8", "floor")
    for e in walls:
        add(*mesh(e), "#D5DBE1", "wall")
    for e in doors:
        add(*mesh(e), "#C0392B" if ids[e.GlobalId] in failing else "#1B7447", "door")
    for z, group in enumerate(("floor", "wall", "door")):
        if groups[group]:
            tri = np.concatenate([t for t, _ in groups[group]])
            cols = np.concatenate([c for _, c in groups[group]])
            pc = Poly3DCollection(tri, facecolors=cols, edgecolors=cols, linewidths=0.3)
            pc.set_zorder(z + 1)
            ax.add_collection3d(pc)

    V = np.vstack(allv)
    lo, hi = V.min(0), V.max(0)
    span = (hi - lo).max()
    c = (lo + hi) / 2
    ax.set_xlim(c[0] - span / 2, c[0] + span / 2)
    ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
    ax.set_zlim(floor - span * 0.3, floor + span * 0.7)  # storeys above sea level (hitos: 20 m) too
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.canvas.draw()

    def label(x, y, z, text, col, size):
        X, Y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.annotate(text, xy=(X, Y), fontsize=size, fontweight="bold", color=col, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.2), annotation_clip=False,
                    zorder=1000)

    few = len(spaces) + len(doors) <= MAX_LABELS  # a large real storey: only the failing doors get a label

    def short(name):  # Revit: "Family:Type:ElementId" -> the ElementId (it is in the report's full name)
        return name.rsplit(":", 1)[-1] if ":" in name and len(name) > 20 else name

    for e in spaces if few else []:
        v, _ = mesh(e)
        cc = v.mean(0)
        label(cc[0], cc[1], floor + 0.1, short(ids[e.GlobalId]), "#15202B", 15)
    for e in doors:
        v, _ = mesh(e)
        cc = v.mean(0)
        name = ids[e.GlobalId]
        if few or name in failing:
            label(cc[0], cc[1], v[:, 2].max() + 0.5, short(name), "#C0392B" if name in failing else "#1B7447",
                  13 if few else 9)
    plt.subplots_adjust(0, 0, 1, 1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, facecolor="white")
    plt.close(fig)
    _crop(out_png)
    return out_png


def _crop(png: str, pad: int = 40) -> None:
    """Remove the white margin around the drawing."""
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box:
        im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                 min(box[2] + pad, im.width), min(box[3] + pad, im.height))).save(png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc")
    ap.add_argument("png")
    a = ap.parse_args()
    print("wrote", render(a.ifc, a.png))
