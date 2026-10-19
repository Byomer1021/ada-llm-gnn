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
from src.ada.ifc_reader import read_ifc  # noqa: E402
from src.ada.rules import evaluate  # noqa: E402

CUT = 1.25  # cut walls at this height (m) so the inside is visible
LIGHT = np.array([0.4, -0.7, 0.9]) / np.linalg.norm([0.4, -0.7, 0.9])


def render(ifc_path: str, out_png: str, elev: float = 42, azim: float = -60,
           failing: set[str] | None = None) -> str:
    """failing = ids of the doors to draw red. None -> ask the rule engine (only for the
    report / slides; the web UI passes the main agent's failing elements)."""
    model = ifcopenshell.open(ifc_path)
    s = G.settings()
    s.set("use-world-coords", True)
    if failing is None:
        result = evaluate(read_ifc(ifc_path))
        failing = {d for d, f in result["door_failures"].items() if f}

    def mesh(e):
        g = G.create_shape(s, e).geometry
        v = np.array(g.verts).reshape(-1, 3)
        f = np.array(g.faces).reshape(-1, 3)
        v[:, 2] = np.minimum(v[:, 2], CUT)
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

    for e in model.by_type("IfcSpace"):
        v, f = mesh(e)
        z0 = v[:, 2].min()
        keep = np.array([t for t in f if np.allclose(v[t][:, 2], z0)])
        v2 = v.copy()
        v2[:, 2] = z0 + 0.01
        if len(keep):
            add(v2, keep, "#E3EEF8", "floor")
    for e in model.by_type("IfcWall"):
        add(*mesh(e), "#D5DBE1", "wall")
    for e in model.by_type("IfcDoor"):
        add(*mesh(e), "#C0392B" if (e.Name or e.GlobalId) in failing else "#1B7447", "door")
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
    ax.set_zlim(-span * 0.3, span * 0.7)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.canvas.draw()

    def label(x, y, z, text, col, size):
        X, Y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.annotate(text, xy=(X, Y), fontsize=size, fontweight="bold", color=col, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.2), annotation_clip=False,
                    zorder=1000)

    for e in model.by_type("IfcSpace"):
        v, _ = mesh(e)
        cc = v.mean(0)
        label(cc[0], cc[1], 0.1, e.Name or "?", "#15202B", 15)
    for e in model.by_type("IfcDoor"):
        v, _ = mesh(e)
        cc = v.mean(0)
        name = e.Name or "?"
        label(cc[0], cc[1], v[:, 2].max() + 0.5, name, "#C0392B" if name in failing else "#1B7447", 13)
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
