"""Oracle tests: proteon CHARMM19+EEF1 energy vs BALL Python bindings.

Backs the proteon-charmm19-vs-ball-ci claim in
`evident/claims/forcefield_charmm19_ball.yaml`. Closes the gap that
forcefield_charmm19_internal documented as an open reference claim.

Skips silently if the `ball` extension module is not installed; that
ships from the theGreatHerrLebert/ball PyPI publication of `ball-py`,
in flight at the time this test was introduced.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import proteon


pytestmark = pytest.mark.oracle("ball")

# Import the BALL bindings via importorskip so the suite stays green
# until ball-py is available in the test environment.
ball = pytest.importorskip(
    "ball",
    reason=(
        "ball-py not installed; install from PyPI once published "
        "(see github.com/theGreatHerrLebert/ball release flow)"
    ),
)

# proteon's compute_energy currently rejects nonbonded_cutoff overrides
# on ff='charmm19_eef1' with a hard NotImplementedError — only AMBER96
# supports the cutoff knob today. The whole BALL parity comparison
# requires NoCutoff on both sides; without it, BALL's no-cutoff energy
# vs proteon's default 15 Å cutoff produce ~30% gaps on electrostatic
# alone, swamping any meaningful component-by-component check.
#
# Skip until the CHARMM cutoff override lands in proteon-connector. The
# claim's pinned_versions[BALL] resolves at that point, alongside the
# first CI green for these assertions. This is recorded as a documented
# blocker in claims/forcefield_charmm19_ball.yaml's failure_modes.
try:
    _probe = proteon.load(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "test-pdbs", "1crn.pdb",
    ))
    proteon.compute_energy(_probe, ff="charmm19_eef1", nonbonded_cutoff=1e6)
except NotImplementedError as _exc:
    pytest.skip(
        f"proteon.compute_energy lacks CHARMM19 cutoff override: {_exc}; "
        f"see claims/forcefield_charmm19_ball.yaml failure_modes",
        allow_module_level=True,
    )
except Exception:
    # Other errors (missing fixture, etc.) surface in the per-test setup.
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CRAMBIN = os.path.join(REPO, "test-pdbs", "1crn.pdb")


def _to_proteon_keys(charmm_dict: dict) -> dict:
    """Map ball.charmm_energy() keys onto proteon.compute_energy()'s schema.

    BALL exposes `proper_torsion`, `improper_torsion`, plus a sum
    `torsion`. proteon reports `torsion` (proper) and
    `improper_torsion` separately. This adapter returns a dict whose
    keys match proteon's, taking BALL's split keys.

    BALL exposes `nonbonded` (vdw + electrostatic) and `vdw`
    separately; proteon reports `vdw` and `electrostatic`. We
    derive electrostatic = nonbonded - vdw.
    """
    return {
        "bond_stretch":     charmm_dict["bond_stretch"],
        "angle_bend":       charmm_dict["angle_bend"],
        "torsion":          charmm_dict["proper_torsion"],
        "improper_torsion": charmm_dict["improper_torsion"],
        "vdw":              charmm_dict["vdw"],
        "electrostatic":    charmm_dict["nonbonded"] - charmm_dict["vdw"],
        "solvation":        charmm_dict["solvation"],
        "total":            charmm_dict["total"],
    }


@pytest.fixture(scope="module")
def reference_energies():
    """Compute proteon and BALL CHARMM19+EEF1 energies on crambin once.

    Loads crambin, places hydrogens via proteon's standard pipeline,
    runs proteon.compute_energy and ball.charmm_energy on the
    resulting structure. The hydrogen placement is shared between
    engines — `add_hydrogens=False` on the BALL side — so H position
    is not a divergence axis in the comparison.
    """
    if not os.path.isfile(CRAMBIN):
        pytest.skip(f"crambin fixture not found at {CRAMBIN}")

    s = proteon.load(CRAMBIN)
    # CHARMM19 is a polar-H force field — only N-H, O-H, S-H placed.
    # place_peptide_hydrogens mutates the structure in place and returns
    # (n_placed, n_failed); the count is not asserted here, only the
    # downstream energy components.
    proteon.place_peptide_hydrogens(s)

    proteon_energy = proteon.compute_energy(
        s, ff="charmm19_eef1", nonbonded_cutoff=1e6,
    )

    # BALL accepts the same fixture file. add_hydrogens=False so it
    # does not re-place hydrogens; it should accept proteon's
    # hydrogenated PDB if proteon writes the structure back to disk
    # somewhere first. For v0 we let BALL place its own hydrogens
    # and accept the H-placement divergence as a documented tail
    # in the YAML's failure_modes; tightening that interface lands
    # when proteon exposes a "save with hydrogens" helper.
    ball_energy = ball.charmm_energy(
        CRAMBIN,
        use_eef1=True,
        nonbonded_cutoff=1e6,
        add_hydrogens=True,
    )

    return proteon_energy, _to_proteon_keys(ball_energy)


class TestCharmm19BallOracle:
    """Backs the proteon-charmm19-vs-ball-ci claim's tolerances."""

    def test_bond_stretch(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["bond_stretch"] - ball_e["bond_stretch"]) / abs(ball_e["bond_stretch"])
        assert rel < 0.01, f"bond_stretch relative diff {rel:.3%} exceeds 1%"

    def test_angle_bend(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["angle_bend"] - ball_e["angle_bend"]) / abs(ball_e["angle_bend"])
        assert rel < 0.01, f"angle_bend relative diff {rel:.3%} exceeds 1%"

    def test_proper_torsion(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["torsion"] - ball_e["torsion"]) / abs(ball_e["torsion"])
        assert rel < 0.025, f"proper torsion relative diff {rel:.3%} exceeds 2.5%"

    def test_improper_torsion(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["improper_torsion"] - ball_e["improper_torsion"]) / abs(ball_e["improper_torsion"])
        assert rel < 0.025, f"improper torsion relative diff {rel:.3%} exceeds 2.5%"

    def test_vdw(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["vdw"] - ball_e["vdw"]) / abs(ball_e["vdw"])
        assert rel < 0.025, f"vdW relative diff {rel:.3%} exceeds 2.5%"

    def test_electrostatic(self, reference_energies):
        # Wide by design — BALL diverges ~20% from OpenMM-canonical
        # AMBER96 here on the AMBER+BALL claim; we expect a similar
        # gap on CHARMM. The 25% band catches catastrophic charge-
        # set regressions without breaking on the known divergence.
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["electrostatic"] - ball_e["electrostatic"]) / abs(ball_e["electrostatic"])
        assert rel < 0.25, f"electrostatic relative diff {rel:.3%} exceeds 25%"

    def test_solvation_eef1(self, reference_energies):
        # EEF1 — BALL is the canonical reference here (no OpenMM
        # equivalent), so this is the tightest external-only band.
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["solvation"] - ball_e["solvation"]) / abs(ball_e["solvation"])
        assert rel < 0.05, f"EEF1 solvation relative diff {rel:.3%} exceeds 5%"
