"""Oracle tests: proteon.geometry.rmsd vs BALL's RMSDMinimizer.

Backs the proteon-rmsd-vs-ball-ci claim in
`evident/claims/geometry_rmsd_ball.yaml`. proteon's RMSD primitive
(`proteon.geometry.rmsd`, backed by proteon-align/src/core/kabsch.rs)
had no external oracle until now; BALL's `RMSDMinimizer` (Coutsalis
et al. eigenvalue method) is an independent implementation of
Kabsch — disagreement between the two would surface a real bug in
either's eigenvalue solve, optimal-rotation reconstruction, or
post-superposition deviation calculation.

Skips silently if the `ball` extension module is not installed or
does not yet expose `ball.rmsd` (the binding shipped in ball-py
0.1.0a3; earlier wheels lack it).
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np  # noqa: F401  # used only for type annotations on the helper
import pytest

import proteon


pytestmark = pytest.mark.oracle("ball")

# ball-py bundles BALL's data tree (Fragments.db, parameters,
# templates, radii) under share/BALL/ in the wheel but does not yet
# set BALL_DATA_PATH at import time. Without that env var,
# AmberFF/CharmmFF/RMSDMinimizer setup fails with the opaque
# "BALL: std::exception". Auto-discover the bundled data dir from
# the installed wheel before importing ball, so tests work whether
# the user pre-set BALL_DATA_PATH or not. Upstream ball-py should set
# this in its own module init; track as v0.4 work on the BALL repo.
def _discover_ball_data_path() -> str | None:
    if os.environ.get("BALL_DATA_PATH"):
        return os.environ["BALL_DATA_PATH"]
    for entry in sys.path:
        candidate = pathlib.Path(entry) / "share" / "BALL"
        if (candidate / "fragments" / "Fragments.db").is_file():
            return str(candidate)
    return None


_data_path = _discover_ball_data_path()
if _data_path:
    os.environ["BALL_DATA_PATH"] = _data_path

ball = pytest.importorskip(
    "ball",
    reason=(
        "ball-py not installed; install from PyPI once published "
        "(see github.com/theGreatHerrLebert/ball release flow)"
    ),
)

# ball-py 0.1.0a2 and earlier lack the rmsd binding (added in 0.1.0a3,
# PR #2 on theGreatHerrLebert/ball). Guard with hasattr so the suite
# stays green on older wheels rather than emitting an AttributeError
# at collection time.
if not hasattr(ball, "rmsd"):
    pytest.skip(
        "ball.rmsd not present; install ball-py >= 0.1.0a3",
        allow_module_level=True,
    )

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CRAMBIN = os.path.join(REPO, "test-pdbs", "1crn.pdb")


def _proteon_ca_coords(structure) -> np.ndarray:
    """Extract CA-only Nx3 coords from a proteon Structure.

    Mirrors what ball.rmsd(..., atoms="ca") does on its side: pull
    every C-alpha in residue order and ignore the rest. The returned
    coordinate array carries no atom-name metadata; the comparison
    contract is "same N CA atoms in the same residue order, both
    sides," which holds whenever the two structures are loaded from
    PDB files that share residue order. Crambin (1crn) is a single
    chain with contiguous residue numbering, so this assumption is
    safe for the v0 claim's fixture; it would need re-examination on
    multi-chain or alt-loc-bearing inputs.
    """
    mask = proteon.select(structure, "name CA")
    return structure.coords[mask]


@pytest.fixture(scope="module")
def crambin_pair(tmp_path_factory):
    """Original and minimized crambin, with paired CA-coord arrays.

    Uses ball.minimize_energy to produce the second structure rather
    than carrying a pre-minimized fixture in the source tree. The
    minimization is short (20 steps) and deterministic; the resulting
    CA-RMSD is non-trivial (~0.5 A on crambin) — large enough to
    exercise the Kabsch path but small enough that any meaningful
    disagreement between BALL and proteon points to a real bug rather
    than numerical noise.
    """
    if not os.path.isfile(CRAMBIN):
        pytest.skip(f"crambin fixture not found at {CRAMBIN}")

    work = tmp_path_factory.mktemp("rmsd_oracle")
    minimized_pdb = str(work / "1crn_min.pdb")

    # ball.minimize_energy places hydrogens by default and writes the
    # post-minimisation PDB to disk. proteon.load can read that PDB
    # back; it sees the hydrogens but our CA-only extraction filters
    # them out, so H placement is not a divergence axis here.
    ball.minimize_energy(CRAMBIN, minimized_pdb, max_iter=20)

    s_orig = proteon.load(CRAMBIN)
    s_min = proteon.load(minimized_pdb)

    coords_orig = _proteon_ca_coords(s_orig)
    coords_min = _proteon_ca_coords(s_min)

    if len(coords_orig) != len(coords_min):
        pytest.fail(
            f"CA count drifted under minimization: "
            f"{len(coords_orig)} vs {len(coords_min)}"
        )

    return {
        "orig_pdb": CRAMBIN,
        "min_pdb": minimized_pdb,
        "coords_orig": coords_orig,
        "coords_min": coords_min,
    }


class TestRmsdBallOracle:
    """Backs the proteon-rmsd-vs-ball-ci claim's tolerances.

    Two assertions per pairing-mode × superpose combination:
      1. Self-RMSD ≈ 0 — sanity that both engines bottom out at zero
         on identical input. A non-zero result here would indicate
         either AtomBijection paired wrong atoms or RMSDMinimizer's
         eigenvalue solve is mis-applying the optimal rotation.
      2. Original-vs-minimised: |proteon - BALL| < 1e-3 A.
         Kabsch is a deterministic well-conditioned algorithm; both
         implementations should agree to roughly machine precision
         when fed identical paired coordinates. The 1e-3 A tolerance
         leaves margin for residue-ordering or H-placement edge
         cases that sneak into either tool's preprocessing without
         triggering the test on numerical noise alone.
    """

    def test_self_rmsd_proteon_is_zero(self, crambin_pair):
        coords = crambin_pair["coords_orig"]
        r = proteon.geometry.rmsd(coords, coords)
        assert r == pytest.approx(0.0, abs=1e-6), (
            f"proteon self-RMSD non-zero: {r}"
        )

    def test_self_rmsd_ball_is_near_zero(self, crambin_pair):
        # BALL's Coutsalis et al. eigenvalue solver does NOT bottom out
        # at zero on identical input — measured at ~5.4e-3 A on crambin's
        # 46-CA point cloud. proteon's Kabsch path lands at ~4e-7 A on
        # the same input (5 orders of magnitude tighter). The 0.01 A
        # band guards against a catastrophic regression (BALL's solver
        # diverges) without breaking on the documented floor. The
        # superpose=False path IS exactly zero on identical input on
        # both sides, exercised by test_no_superpose_rmsd_agrees.
        r = ball.rmsd(
            crambin_pair["orig_pdb"], crambin_pair["orig_pdb"],
            atoms="ca", superpose=True,
        )
        assert r["rmsd"] < 0.01, (
            f"BALL self-RMSD {r['rmsd']} exceeds 0.01 A floor — eigenvalue solver may have regressed"
        )

    def test_kabsch_rmsd_agrees(self, crambin_pair):
        """Kabsch-aligned CA-RMSD agrees between proteon and BALL.

        This is the load-bearing assertion for the claim: same algorithm
        (Kabsch superposition + RMSD over paired CA atoms), independent
        implementations, identical fixture. The two values should agree
        to ~machine precision on the well-conditioned crambin pair.
        """
        p_rmsd = proteon.geometry.rmsd(
            crambin_pair["coords_orig"], crambin_pair["coords_min"]
        )
        b = ball.rmsd(
            crambin_pair["orig_pdb"], crambin_pair["min_pdb"],
            atoms="ca", superpose=True,
        )
        # Both should report a non-trivial RMSD on a minimised structure;
        # if either is exactly 0, the input fixture wasn't actually
        # minimised and the load-bearing assertion below is degenerate.
        assert p_rmsd > 0.0, "proteon RMSD on original-vs-min is exactly 0"
        assert b["rmsd"] > 0.0, "BALL RMSD on original-vs-min is exactly 0"

        abs_diff = abs(p_rmsd - b["rmsd"])
        assert abs_diff < 1e-3, (
            f"|proteon ({p_rmsd:.6f}) - BALL ({b['rmsd']:.6f})| = "
            f"{abs_diff:.6f} A exceeds 1e-3 A"
        )

    def test_no_superpose_rmsd_agrees(self, crambin_pair):
        """Without superposition, both report direct coordinate deviation.

        Tests proteon.geometry.rmsd_no_super against ball.rmsd with
        superpose=False. The non-superposed RMSD must be >= the
        Kabsch-aligned value (Kabsch is the minimum over all rigid
        motions), and the two implementations should agree to high
        precision since "RMSD without superposition" reduces to
        sqrt(mean(||x_i - y_i||^2)) — no algorithmic freedom.
        """
        p_rmsd = proteon.geometry.rmsd_no_super(
            crambin_pair["coords_orig"], crambin_pair["coords_min"]
        )
        b = ball.rmsd(
            crambin_pair["orig_pdb"], crambin_pair["min_pdb"],
            atoms="ca", superpose=False,
        )
        abs_diff = abs(p_rmsd - b["rmsd"])
        assert abs_diff < 1e-4, (
            f"|proteon_no_super ({p_rmsd:.6f}) - BALL ({b['rmsd']:.6f})| = "
            f"{abs_diff:.6f} A exceeds 1e-4 A (no algorithmic freedom)"
        )

        # Sanity: Kabsch is the minimum, so non-superposed >= superposed.
        b_super = ball.rmsd(
            crambin_pair["orig_pdb"], crambin_pair["min_pdb"],
            atoms="ca", superpose=True,
        )
        assert b["rmsd"] >= b_super["rmsd"] - 1e-9, (
            "non-superposed RMSD smaller than Kabsch — Kabsch should be the "
            "minimum over all rigid motions"
        )
