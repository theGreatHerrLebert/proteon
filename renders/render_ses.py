#!/usr/bin/env python3
"""Render an OBJ SES mesh to a shaded PNG with matplotlib (headless)."""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # f a//na b//nb c//nc  -> take the vertex index (1-based)
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(idx)
    return np.asarray(verts, float), np.asarray(faces, int)


def render(obj_path, png_path, title, elev=22, azim=-60, color=(0.32, 0.55, 0.85)):
    V, F = load_obj(obj_path)
    tris = V[F]  # (nf, 3, 3)

    # Face normals (outward, mesh is consistently oriented).
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)

    # Camera direction (scene -> camera) for matplotlib (elev, azim).
    er, ar = np.radians(elev), np.radians(azim)
    cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])

    # Back-face cull: keep only triangles whose outward normal faces the camera.
    front = (n @ cam) > 0
    tris, n = tris[front], n[front]

    # Painter's order: draw far triangles first (depth = centroid along cam).
    depth = (tris.mean(axis=1) @ cam)
    order = np.argsort(depth)
    tris, n = tris[order], n[order]

    # Lambertian shade from a light near the camera.
    light = cam / np.linalg.norm(cam)
    shade = np.clip(n @ light, 0.0, 1.0)
    shade = 0.30 + 0.70 * shade  # ambient + diffuse
    base = np.array(color)
    facecolors = np.clip(shade[:, None] * base, 0, 1)
    facecolors = np.concatenate([facecolors, np.ones((len(facecolors), 1))], axis=1)

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(tris, facecolors=facecolors, edgecolors="none", linewidths=0)
    pc.set_zsort("max")
    ax.add_collection3d(pc)

    lo, hi = V.min(0), V.max(0)
    ctr = (lo + hi) / 2
    rng = (hi - lo).max() / 2 * 0.9
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, color="#222")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{png_path}: {len(V)} verts, {len(F)} tris")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
