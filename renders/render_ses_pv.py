#!/usr/bin/env python3
"""High-quality headless render of an SES mesh (PLY/OBJ) via PyVista/VTK.

Real z-buffer (no painter-algorithm bleed-through), smooth per-vertex normals,
specular highlights, anti-aliasing. Run under a virtual display, e.g.:
    xvfb-run -a python render_ses_pv.py mesh.ply out.png "Title"
"""
import sys
import pyvista as pv

pv.OFF_SCREEN = True


def render(mesh_path, png_path, title="", color="#7fb2e6", azim=25, elev=18):
    mesh = pv.read(mesh_path)
    mesh = mesh.clean()  # merge coincident points so smooth normals are seamless
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True)

    pl = pv.Plotter(off_screen=True, window_size=(1400, 1400), lighting="none")
    pl.set_background("#0f1116", top="#222a38")  # dark so the lit surface pops

    pl.add_mesh(
        mesh,
        color=color,
        smooth_shading=True,
        specular=0.35,
        specular_power=20,
        ambient=0.12,  # low ambient → strong light/shadow gradient = readable form
        diffuse=0.95,
        show_edges=False,
    )

    # Warm key (upper-left) + cool fill (lower-right) + back rim: the warm/cool
    # split + low ambient makes curvature read as 3D.
    pl.add_light(pv.Light(position=(-0.7, 1.0, 0.8), color="#fff2dd", intensity=1.05,
                          light_type="scene light"))
    pl.add_light(pv.Light(position=(1.0, -0.3, 0.4), color="#cfe0ff", intensity=0.55,
                          light_type="scene light"))
    pl.add_light(pv.Light(position=(0.1, 0.2, -1.0), color="#ffffff", intensity=0.45,
                          light_type="scene light"))

    # Ambient occlusion: darken the crevices/pockets between atoms so the SES
    # "golf-ball" texture and surface pockets are visible. Radius in world (Å).
    diag = float(((mesh.bounds[1] - mesh.bounds[0]) ** 2
                  + (mesh.bounds[3] - mesh.bounds[2]) ** 2
                  + (mesh.bounds[5] - mesh.bounds[4]) ** 2) ** 0.5)
    try:
        pl.enable_ssao(radius=max(1.5, diag * 0.03), bias=0.01, blur=True)
    except Exception as e:
        print("ssao unavailable:", e)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = "iso"
    pl.camera.azimuth = azim
    pl.camera.elevation = elev
    pl.camera.zoom(1.35)
    if title:
        pl.add_text(title, position="upper_edge", font_size=16, color="#e8edf5")

    pl.screenshot(png_path, transparent_background=False)
    pl.close()
    print(f"{png_path}: {mesh.n_points} pts, {mesh.n_cells} cells")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
