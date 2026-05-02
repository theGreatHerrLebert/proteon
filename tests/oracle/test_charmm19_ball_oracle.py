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

    BALL's `nonbonded` is the FULL nonbonded component energy as
    reported by `CharmmFF::getNonbondedEnergy()` — i.e. the sum of
    `vdw_energy_ + electrostatic_energy_ + solvation_energy_`
    (`charmmNonBonded.C:1217`). To extract the pure Coulomb term we
    subtract BOTH vdW and EEF1 solvation. (An earlier version of this
    adapter just did `nonbonded - vdw` which silently double-counted
    solvation in the "electrostatic" field, hiding ~1.4 kJ/mol of
    real signal under thousands of kJ/mol of EEF1 noise.)
    """
    return {
        "bond_stretch":     charmm_dict["bond_stretch"],
        "angle_bend":       charmm_dict["angle_bend"],
        "torsion":          charmm_dict["proper_torsion"],
        "improper_torsion": charmm_dict["improper_torsion"],
        "vdw":              charmm_dict["vdw"],
        "electrostatic":    charmm_dict["nonbonded"] - charmm_dict["vdw"] - charmm_dict["solvation"],
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
    # CHARMM19 is a polar-H force field — backbone amide H + sidechain
    # polar Hs (NH, OH, SH, guanidinium, imidazole, indole, NH3+ at the
    # N-terminus). C-terminus stays deprotonated (COO-, no HXT).
    #
    # place_peptide_hydrogens alone is NOT enough: it places only
    # backbone amide H, leaving the N-terminal NH3+ unprotonated
    # (residue 0's N at -1.35e with no balancing H+) and the LYS/ARG/
    # HIS sidechains unprotonated. With those polar H atoms missing,
    # the system carries ~-10e of fictitious net charge and the
    # electrostatic energy is sign-flipped vs BALL.
    #
    # place_all_hydrogens(polar_only=True) does the full polar-H job:
    # backbone + sidechain + N-terminal NH3+, C-terminal stays COO-.
    proteon.place_all_hydrogens(s, polar_only=True)

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
        reason="proteon CHARMM proper_torsion 238 vs BALL 994 on crambin (~76% off) under the corrected polar-H fixture (place_all_hydrogens(polar_only=True)). Earlier readings of '218 vs 230 / 5.1%' came from the half-prepared fixture (place_peptide_hydrogens-only) which left N-terminal and sidechain polar H's unplaced — once those H atoms are present, BALL discovers many new H-containing 4-atom torsion paths (e.g. H1-N-CA-C at the N-terminus) that proteon's [ResidueTorsions] filter does not yet enumerate. The fix is in proteon's residue_prefix logic for terminal-variant entries (THR-N, ASN-C) and in the H-atom alt-name machinery for HD21/HD22 vs 1HD2/2HD2 at the [ResidueTorsions] lookup site. See geometry_charmm19_ball.yaml failure_modes",
        strict=False,
    )
    def test_proper_torsion(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["torsion"] - ball_e["torsion"]) / abs(ball_e["torsion"])
        assert rel < 0.025, f"proper torsion relative diff {rel:.3%} exceeds 2.5%"

    @pytest.mark.xfail(
        reason="proteon CHARMM improper_torsion = 0 vs BALL 264.94 on crambin under the corrected polar-H fixture (was BALL 39.7 under the half-prepared fixture; the higher number reflects new sidechain N-H impropers like ASN:ND2-HD21-HD22-CG that are visible only with full polar-H placement). Parser + topology scaffolding from PR #22 is in place (CHARMM's 7-column [ImproperTorsions], 5-column [ResidueImproperTorsions]); the harmonic-improper energy compute is dead-code-gated until the unsigned-dihedral chain rule is reconciled with the existing Bekker analytical-force chain. See geometry_charmm19_ball.yaml failure_modes",
        strict=False,
    )
    def test_improper_torsion(self, reference_energies):
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["improper_torsion"] - ball_e["improper_torsion"]) / abs(ball_e["improper_torsion"])
        assert rel < 0.025, f"improper torsion relative diff {rel:.3%} exceeds 2.5%"

    def test_vdw(self, reference_energies):
        # PHE/TYR para-diagonal LJ exclusion (CHARMM convention from
        # charmmNonBonded.C:547-565) closed an ~11.5% / ~110 kJ/mol gap
        # vs BALL on crambin. proteon used to count CG-CZ, CD1-CE2,
        # CD2-CE1 as ordinary 1-4 LJ pairs at ~2.8 Å where the wall is
        # very repulsive (~12-25 kJ/mol per pair). Topology now mirrors
        # BALL and zeroes the LJ contribution for these pairs while
        # leaving Coulomb intact. Residual on crambin is < 0.01 kJ/mol.
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["vdw"] - ball_e["vdw"]) / abs(ball_e["vdw"])
        assert rel < 0.025, f"vdW relative diff {rel:.3%} exceeds 2.5%"

    def test_electrostatic(self, reference_energies):
        # CHARMM19+EEF1 distance-dependent dielectric (ε ∝ r → 1/r²) is
        # the canonical convention from Lazaridis & Karplus 1999 — it's
        # the implicit screening the partial charges expect. proteon now
        # consults `ForceField::uses_distance_dependent_dielectric()`
        # (true for CharmmParams, false for AmberParams) and the
        # electrostatic matches BALL within 0.02% on crambin (-2465.79
        # vs -2466.29 kJ/mol). Before this fix proteon was 2.81x too
        # negative because constant ε attracts charges much more than
        # ε=r — visible only via cross-tool oracle, not via internal
        # cross-path tests (NBL/forces all agreed in the same wrong way).
        # The 1% band is wide enough to absorb floating-point and atom-
        # ordering noise without hiding a parameter regression.
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["electrostatic"] - ball_e["electrostatic"]) / abs(ball_e["electrostatic"])
        assert rel < 0.01, f"electrostatic relative diff {rel:.3%} exceeds 1%"

    def test_solvation_eef1(self, reference_energies):
        # EEF1 — BALL is the canonical reference here (no OpenMM
        # equivalent), so this is the tightest external-only band.
        proteon_e, ball_e = reference_energies
        rel = abs(proteon_e["solvation"] - ball_e["solvation"]) / abs(ball_e["solvation"])
        assert rel < 0.05, f"EEF1 solvation relative diff {rel:.3%} exceeds 5%"
