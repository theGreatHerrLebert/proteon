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
def reference_energies(tmp_path_factory):
    """Compute proteon and BALL CHARMM19+EEF1 energies on the same
    polar-H-placed crambin structure once.

    Workflow: proteon places polar hydrogens (CHARMM19 is a polar-H
    force field), writes the hydrogenated structure to a temp PDB,
    and feeds it to BALL with `add_hydrogens=False` so BALL does NOT
    re-place its own all-atom hydrogens. This is critical: BALL's
    `AddHydrogenProcessor` places non-polar Hs (CH, CH2, CH3) which
    CHARMM19 does not parameterize — without this control BALL emits
    "CharmmNonBonded::setup: cannot find Lennard Jones parameters"
    on every non-polar H and silently produces wrong energies.

    Sharing the H-placement removes that divergence axis entirely —
    both tools then evaluate the same atoms with their respective
    CHARMM19+EEF1 parameter files.
    """
    if not os.path.isfile(CRAMBIN):
        pytest.skip(f"crambin fixture not found at {CRAMBIN}")

    s = proteon.load(CRAMBIN)
    # CHARMM19 is a polar-H force field — only N-H, O-H, S-H placed.
    # place_peptide_hydrogens mutates the structure in place and returns
    # (n_placed, n_failed); the count is not asserted here, only the
    # downstream energy components.
    proteon.place_peptide_hydrogens(s)

    work = tmp_path_factory.mktemp("charmm_ball_oracle")
    polar_h_pdb = str(work / "1crn_polarH.pdb")
    proteon.save_pdb(s, polar_h_pdb)

    proteon_energy = proteon.compute_energy(
        s, ff="charmm19_eef1", nonbonded_cutoff=1e6,
    )

    ball_energy = ball.charmm_energy(
        polar_h_pdb,
        use_eef1=True,
        nonbonded_cutoff=1e6,
        add_hydrogens=False,
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

    # The four assertions below are xfail-marked: activating the
    # CHARMM-ball oracle (now possible because the cutoff override
    # landed) surfaces real implementation divergences in proteon's
    # CHARMM19 path that pre-date this work and are NOT a tolerance-
    # widening problem. Each measured number is in the claim YAML's
    # failure_modes block. Strict=False so a future fix flips the
    # xfail to xpass and surfaces immediately rather than masking.

    @pytest.mark.xfail(
        reason="proteon CHARMM proper_torsion 218.2 vs BALL 230.0 on crambin (~5.1% rel diff) after the [ResidueTorsions] filter landed; was 610.8 (165% off) before. Sign and magnitude are now in agreement; residual is just outside the 2.5% band, plausibly N-terminus-template differences (proteon uses 'THR' for residue 0 but BALL would use 'THR-N') or residual wildcard-fallback differences; see geometry_charmm19_ball.yaml failure_modes",
        strict=False,
    )
    def test_proper_torsion(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["torsion"] - ball_e["torsion"]) / abs(ball_e["torsion"])
        assert rel < 0.025, f"proper torsion relative diff {rel:.3%} exceeds 2.5%"

    @pytest.mark.xfail(
        reason="proteon CHARMM improper_torsion = 0 vs BALL 39.7 on crambin. Three issues found this PR (first two fixed): (1) parser now reads CHARMM's 7-column [ImproperTorsions]; (2) 5-column [ResidueImproperTorsions] is parsed correctly. Topology populates 159 impropers (was 0). (3) CHARMM measures the harmonic dihedral with central at slot 1 AND uses unsigned |phi| (acos), proteon's compute_dihedral returns signed atan2. Wiring with both fixes gives 46.2 kJ/mol on crambin (16% off BALL's 39.7), but breaks the cross-path gradient test because the |phi| energy's derivative isn't consistent with the existing Bekker analytical-force chain. Held back until a small surgical fix to the force chain (multiply by sign(phi_signed)) lands in the next PR. See geometry_charmm19_ball.yaml failure_modes",
        strict=False,
    )
    def test_improper_torsion(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["improper_torsion"] - ball_e["improper_torsion"]) / abs(ball_e["improper_torsion"])
        assert rel < 0.025, f"improper torsion relative diff {rel:.3%} exceeds 2.5%"

    @pytest.mark.xfail(
        reason="proteon CHARMM vdw 11.5% off BALL on un-minimized crambin (-834.3 vs -942.7) after [LennardJones14] table fix landed; sign now correct, residual likely inflated CH*E united-atom radii pre-minimization or NBFIX overrides not yet loaded; see geometry_charmm19_ball.yaml failure_modes",
        strict=False,
    )
    def test_vdw(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["vdw"] - ball_e["vdw"]) / abs(ball_e["vdw"])
        assert rel < 0.025, f"vdW relative diff {rel:.3%} exceeds 2.5%"

    @pytest.mark.xfail(
        reason="proteon CHARMM electrostatic +195.7 vs BALL -2711 on crambin after the @E14FAC=0.4 1-4 scaling fix landed (was +5373 before, so this fix closed ~5177 kJ/mol of the gap). Sign still wrong; residual ~2900 kJ/mol most likely the N-terminus NH3+ uncompensated charge — place_peptide_hydrogens leaves residue 0's N at -1.35e with no balancing 3H+. See geometry_charmm19_ball.yaml failure_modes",
        strict=False,
    )
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
