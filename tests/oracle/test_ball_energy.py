"""Oracle test: proteon AMBER96 energy vs BALL Julia (BiochemicalAlgorithms.jl).

Compares energy components on crambin (1crn) without reconstruction,
validating the energy computation, topology building, and charge assignment.
Both proteon and BALL output in kJ/mol.

The **authoritative** AMBER96 reference for proteon is OpenMM (see
`validation/amber96_oracle.py` — 1000-PDB benchmark, <0.5% parity on all
components at NoCutoff). BALL is a second, independent AMBER96
implementation whose behavior has known, documented deviations from the
OpenMM reference; this test pins the pieces where BALL and OpenMM agree
(so a regression in proteon would show up against either), and
documents-but-tolerates the pieces where BALL intrinsically diverges.

Tolerances are component-specific and reflect measured BALL-vs-proteon
agreement on 1crn at NoCutoff, not an aspirational 0.1%. The number
next to each assertion is the band inside which proteon must stay; a
violation means either a proteon regression or a newer BALL that moved
the reference (re-run `ball_energy_raw.jl` to distinguish).

Known BALL-specific divergences, asserted loosely or one-sided:

- **Impropers** — BALL's matcher supports only single-wildcard patterns;
  the AMBER spec requires double-wildcard (e.g. `* * N H` for the amide
  plane). BALL misses ~100 amide-plane impropers on crambin. Proteon
  matches OpenMM within 0.5%, so proteon is the AMBER-canonical value.
  One-sided: proteon >= BALL, within a small absolute magnitude.

- **Electrostatic** — BALL diverges from proteon by ~20% at NoCutoff on
  crambin (77998 vs 93874 kJ/mol, 2026-04-18). Most likely cause is a
  different partial-charge dictionary or 1-4 scaling convention in
  BALL's AmberFF build vs the OpenMM-canonical AMBER96 values proteon
  uses. Since the OpenMM oracle already pins proteon's charges, we
  assert only a wide 25% band here — enough to catch a catastrophic
  charge-set regression, loose enough to not break on the known gap.

- **Proper torsion and vdw** — 1.5-2% residual at NoCutoff, plausibly
  another small convention difference (BALL parameter-file vintage,
  1-4 vdw scaling). Matches the pattern: narrow, non-regressing gaps.
"""

import os
import pytest
import proteon

pytestmark = pytest.mark.oracle("ball")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CRAMBIN = os.path.join(REPO, "test-pdbs", "1crn.pdb")

# BALL Julia reference values (kJ/mol) on raw 1crn.pdb (no reconstruction).
# Regenerate with:
#     julia --project=path/to/BiochemicalAlgorithms.jl \
#           tests/oracle/ball_energy_raw.jl test-pdbs/1crn.pdb
# Last regenerated: 2026-04-18 with BALL Julia commit-current.
BALL_CRAMBIN_RAW = {
    "bond_stretch": 353.349,
    "angle_bend": 275.770,
    "torsion": 704.441,
    "improper_torsion": 2.077,
    "vdw": -422.993,
    "electrostatic": 77998.26,
}


class TestBallEnergyOracle:
    """Compare proteon AMBER96 energy against BALL Julia reference."""

    @pytest.fixture
    def crambin_energy(self):
        s = proteon.load(CRAMBIN)
        # NoCutoff matches the test's intent: pin the energy math against
        # BALL's own no-cutoff reference, independent of proteon's
        # default 15 Å performance-vs-accuracy policy.
        return proteon.compute_energy(s, ff="amber96", nonbonded_cutoff=1e6)

    def test_bond_stretch_matches(self, crambin_energy):
        """Tolerance 1.0%: measured gap ~0.7% on 1crn; 1% buys margin for
        BALL parameter-file drift without letting a regression slip."""
        fe = crambin_energy["bond_stretch"]
        ball = BALL_CRAMBIN_RAW["bond_stretch"]
        pct = abs(fe - ball) / ball * 100
        assert pct < 1.0, f"Bond stretch: {fe:.2f} vs BALL {ball:.2f} ({pct:.4f}%)"

    def test_angle_bend_matches(self, crambin_energy):
        """Tolerance 1.0%: measured gap ~0.5% on 1crn."""
        fe = crambin_energy["angle_bend"]
        ball = BALL_CRAMBIN_RAW["angle_bend"]
        pct = abs(fe - ball) / ball * 100
        assert pct < 1.0, f"Angle bend: {fe:.2f} vs BALL {ball:.2f} ({pct:.4f}%)"

    def test_torsion_matches(self, crambin_energy):
        """Tolerance 2.5%: measured gap ~1.6% on 1crn. Looser than
        bonded because proper torsion is sensitive to small parameter-
        file differences (phase angles, multiplicities)."""
        fe = crambin_energy["torsion"]
        ball = BALL_CRAMBIN_RAW["torsion"]
        pct = abs(fe - ball) / ball * 100
        assert pct < 2.5, f"Torsion: {fe:.2f} vs BALL {ball:.2f} ({pct:.4f}%)"

    def test_vdw_matches(self, crambin_energy):
        """Tolerance 2.5%: measured gap ~1.2% on 1crn at NoCutoff."""
        fe = crambin_energy["vdw"]
        ball = BALL_CRAMBIN_RAW["vdw"]
        pct = abs(fe - ball) / abs(ball) * 100
        assert pct < 2.5, f"VdW: {fe:.2f} vs BALL {ball:.2f} ({pct:.4f}%)"

    def test_electrostatic_in_reasonable_range(self, crambin_energy):
        """Tolerance 25%: measured gap ~20% on 1crn. See module docstring
        — BALL diverges from the OpenMM-canonical AMBER96 electrostatic,
        and proteon matches OpenMM (not BALL). The 25% band catches a
        catastrophic charge-set regression without breaking on the
        known BALL convention gap."""
        fe = crambin_energy["electrostatic"]
        ball = BALL_CRAMBIN_RAW["electrostatic"]
        pct = abs(fe - ball) / abs(ball) * 100
        assert pct < 25.0, f"Electrostatic: {fe:.2f} vs BALL {ball:.2f} ({pct:.4f}%)"

    def test_improper_on_same_order_as_ball(self, crambin_energy):
        """Improper is a small term; proteon's number is expected to exceed
        BALL's because proteon enforces AMBER's full double-wildcard
        improper lookup (see params.rs::get_improper_torsion) and thus
        catches the ~100 amide-plane impropers (`* * N H`) that BALL's
        single-wildcard implementation misses on every peptide bond.

        Before the 2026-04-13 double-wildcard fix: proteon had 15
        impropers = 2.2 kJ/mol, close to BALL's 2.08 kJ/mol.
        After the fix: 125 impropers = 8.1 kJ/mol, which is the AMBER-
        canonical value (OpenMM's amber96 agrees within 0.5%).

        We still assert the absolute magnitude is small so this term
        doesn't quietly explode.
        """
        fe = crambin_energy["improper_torsion"]
        ball = BALL_CRAMBIN_RAW["improper_torsion"]
        # Proteon expected > BALL but still small in magnitude.
        assert 0 < fe < 20, f"Improper proteon: {fe:.2f} kJ/mol (expected 2-15)"
        # Historical expectation: BALL was ~2 kJ/mol. We should be above that.
        assert fe >= ball, (
            f"Improper regressed below BALL: {fe:.2f} < BALL {ball:.2f}. "
            "Double-wildcard improper matching should catch more amide-plane "
            "terms than BALL's single-wildcard implementation."
        )

    def test_all_components_finite(self, crambin_energy):
        import numpy as np
        for key in ["bond_stretch", "angle_bend", "torsion", "improper_torsion",
                     "vdw", "electrostatic", "total"]:
            assert np.isfinite(crambin_energy[key]), f"{key} is not finite"

    def test_total_is_sum(self, crambin_energy):
        expected = sum(crambin_energy[k] for k in [
            "bond_stretch", "angle_bend", "torsion", "improper_torsion",
            "vdw", "electrostatic", "solvation",
        ])
        assert abs(crambin_energy["total"] - expected) < 0.1  # kJ/mol tolerance
