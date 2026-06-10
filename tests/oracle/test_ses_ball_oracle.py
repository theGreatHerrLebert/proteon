"""Oracle test: proteon's analytic SES mesh area vs BALL's analytic SES.

Backs the `proteon-ses-vs-ball-ci` claim in
`evident/claims/surface_ses_ball.yaml`. proteon meshes the analytic
solvent-excluded (Connolly) surface (proteon-core surface pipeline,
exposed via the connector's `py_surface.ses_mesh_py`) and reports the
triangulated mesh area/volume; BALL computes the closed-form analytic
area/volume via `calculateSESArea`/`calculateSESVolume`
(`ball.ses_area`). Both are fed BYTE-IDENTICAL spheres (same centres,
same per-element vdW radii), so a disagreement is the meshing/area
algorithm, not radius assignment.

BALL's analytic path is the reference Connolly implementation proteon's
mesher ports from — and it is robust to the near-tangency degeneracies
that defeat its own triangulation (`ball.ses_mesh` raises DivisionByZero
on e.g. 1ijp), so the analytic `ball.ses_area` is the right oracle.

Skips silently if `ball` is not installed or predates the `ses_area`
binding (shipped in ball-py 0.1.0a6; the CI pin was 0.1.0a4), or if the
installed proteon connector predates `py_surface` (the surface branch).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import proteon

pytestmark = pytest.mark.oracle("ball")

ball = pytest.importorskip(
    "ball",
    reason=(
        "ball-py not installed; install from PyPI once published "
        "(see github.com/theGreatHerrLebert/ball release flow)"
    ),
)

# ses_area shipped in ball-py 0.1.0a6 (the SES bindings — commits
# 8d8a9d27/73c21e88 on theGreatHerrLebert/ball). Earlier wheels lack it;
# guard with hasattr so the suite stays green on older pins.
if not hasattr(ball, "ses_area"):
    pytest.skip(
        "ball.ses_area not present; install ball-py >= 0.1.0a6",
        allow_module_level=True,
    )

# proteon's SES is exposed by the connector's py_surface module, which
# ships on the surface (feat/mesh-export) branch. Skip if the installed
# connector predates it rather than erroring at collection.
try:
    from proteon_connector import py_surface as _py_surface
except Exception:  # noqa: BLE001 — any import shape failure ⇒ skip
    pytest.skip(
        "proteon connector lacks py_surface (SES); rebuild the connector "
        "from the surface branch",
        allow_module_level=True,
    )

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE = 1.4

# A diverse fixture set spanning fold type and scale (327 -> 3804 atoms), all in
# the repo's test-pdbs/ so they ship to CI. 1aaj is a former degenerate-crash
# case (perturbation-recovered) — a useful regression guard. All mesh on the
# analytic path; measured Delta-area vs BALL is +0.04% (1crn) to +0.08% (1ake).
FIXTURES = ["1crn", "1bpi", "1aaj", "1ubq", "1ake"]

# Fixed per-element vdW radii (A). Used for BOTH meshers so the comparison
# isolates the surface algorithm, not radius assignment (that is the SASA
# oracle's job). Standard Bondi-ish heavy-atom set + H.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}
DEFAULT_RADIUS = 1.70


def _spheres(name: str):
    path = os.path.join(REPO, "test-pdbs", f"{name}.pdb")
    if not os.path.isfile(path):
        pytest.skip(f"fixture not found at {path}")
    s = proteon.load(path)
    coords = np.ascontiguousarray(s.coords, dtype=np.float64)
    radii = np.array(
        [VDW_RADII.get(e, DEFAULT_RADIUS) for e in s.elements], dtype=np.float64
    )
    return coords, radii


class TestSesBallOracle:
    """proteon analytic SES mesh vs BALL analytic SES on identical spheres."""

    @pytest.mark.parametrize("name", FIXTURES)
    def test_ses_area_and_volume_agree(self, name):
        coords, radii = _spheres(name)

        # proteon: analytic SES mesh (probe 1.4 A).
        mesh = _py_surface.ses_mesh_py(coords, radii=radii, probe=PROBE)

        # The area comparison is only meaningful if the mesh is closed and
        # produced by the analytic path (not the resolution-limited grid
        # fallback) — otherwise we would be comparing a different surface.
        assert mesh["watertight"], f"{name}: proteon SES mesh is not watertight"
        assert mesh.get("method") == "analytic", (
            f"{name}: expected analytic SES, got method={mesh.get('method')!r} "
            "(grid fallback => resolution-limited area, not an algorithm comparison)"
        )

        # BALL: closed-form analytic SES on the SAME spheres.
        spheres = [
            [float(c[0]), float(c[1]), float(c[2]), float(r)]
            for c, r in zip(coords, radii)
        ]
        bres = ball.ses_area(spheres, PROBE)

        rel_area = abs(mesh["area"] - bres["area"]) / bres["area"]
        rel_vol = abs(mesh["volume"] - bres["volume"]) / bres["volume"]

        assert rel_area < 0.005, (
            f"{name}: SES area |proteon {mesh['area']:.2f} - BALL {bres['area']:.2f}| "
            f"/ BALL = {rel_area:.4%} >= 0.5%"
        )
        assert rel_vol < 0.01, (
            f"{name}: SES volume |proteon {mesh['volume']:.2f} - "
            f"BALL {bres['volume']:.2f}| / BALL = {rel_vol:.4%} >= 1.0%"
        )

    def test_self_consistency_floor(self):
        """BALL's analytic area is deterministic — same spheres twice agree
        exactly. A sanity floor that the oracle call itself is stable."""
        coords, radii = _spheres("1crn")
        spheres = [
            [float(c[0]), float(c[1]), float(c[2]), float(r)]
            for c, r in zip(coords, radii)
        ]
        a = ball.ses_area(spheres, PROBE)
        b = ball.ses_area(spheres, PROBE)
        assert a["area"] == b["area"]
        assert a["volume"] == b["volume"]
