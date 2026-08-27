"""Oracle test: proteon AMBER96 (+OBC GB) energy vs Molly.jl.

Molly.jl (https://github.com/JuliaMolSim/Molly.jl) is a third independent
implementation of the same force field, and — crucially — it parses the very
same OpenMM `amber96.xml` that `validation/amber96_oracle.py` feeds to OpenMM.
That makes it a *tie-breaker*: proteon's AMBER96 numbers were previously
anchored to OpenMM alone, and one oracle cannot distinguish "proteon is
correct" from "proteon and OpenMM share a convention".

What the third opinion bought (measured 2026-08-27, see the table below):

- **bond stretch, angle bend, proper torsion, vdW** — all three engines agree
  to 0.000%. These components are now triangulated, not merely cross-checked.
- **electrostatic** — proteon vs Molly 0.019%; OpenMM vs Molly 0.000% on the
  vdW+electrostatic sum. Proteon's ~1 kJ/mol offset on a -5483 kJ/mol term is
  well inside any defensible band.
- **improper torsion** — all three engines disagree, and the size of the
  disagreement moves with the geometry (see `test_improper_torsion_*` below).
  This is the finding that needed a second oracle to see.
- **GB solvation** — Molly is the outlier, for a documented reason: it does
  *not* read `GBSAOBCForce` from the OpenMM XML (it warns and skips it) and
  sources OBC radii and screen factors from its own element table instead.
  So this comparison measures a parameter-set difference, not GB math, and
  Molly cannot currently serve as a tie-breaker for proteon's OBC work.
  OpenMM remains the authoritative GB oracle (proteon within ≤5%, Phase B).

Reference structure. Values are frozen against a *committed* prepared PDB,
`tests/oracle/data/1crn_prepped_amber96.pdb`, not a fresh PDBFixer run.
PDBFixer's `addMissingHydrogens` is **not deterministic**: three successive
preparations of 1crn.pdb produced three different hydrogen coordinate sets,
moving single-point energies by tens of kJ/mol. Any test asserting stored
numbers has to pin the structure.

Regenerate the frozen values with:

    JULIA=/path/to/julia \\
    .venv/bin/python validation/amber96_molly_triangulate.py --prepped \\
        tests/oracle/data/1crn_prepped_amber96.pdb

or call the Julia oracle directly (see docs/ORACLE_SETUP.md § Molly).

This test itself needs no Julia — it compares live proteon against the frozen
Molly reference, the same shape as `test_ball_energy.py`.
"""

import os

import pytest

import proteon

pytestmark = pytest.mark.oracle("molly")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREPPED_CRAMBIN = os.path.join(
    REPO, "tests", "oracle", "data", "1crn_prepped_amber96.pdb"
)

# Molly.jl reference values (kJ/mol) on tests/oracle/data/1crn_prepped_amber96.pdb.
# Molly v0.23.3, Julia 1.11.5, amber96.xml from OpenMM 8.5.2, NoCutoff
# (nonbonded_method=:none with dist_cutoff widened past the structure extent —
# leaving dist_cutoff at Molly's 1.0 nm default silently truncates long-range
# Coulomb and is worth ~2600 kJ/mol on crambin).
# Last regenerated: 2026-08-27.
MOLLY_CRAMBIN_PREPPED = {
    "bond_stretch": 8827.026205343744,
    "angle_bend": 3525.6440701983206,
    "torsion": 2159.1447733123887,
    "improper_torsion": 116.82717865201845,
    "vdw": -536.4173166683873,
    "electrostatic": -4947.178980807389,
    "total_vacuum": 9145.045930030772,
    "solvation_obc1": -1658.2404609474063,
}

# Topology counts Molly derives from the same XML. A count mismatch is a
# sharper signal than an energy delta — the BALL improper gap surfaced as
# 10-vs-125 impropers long before anyone explained the kJ/mol.
MOLLY_CRAMBIN_COUNTS = {
    "n_atoms": 642,
    "n_bonds": 652,
    "n_angles": 1183,
    "n_torsions": 1747,
    "n_impropers": 125,
}


@pytest.fixture(scope="module")
def vacuum_energy():
    """proteon AMBER96 in vacuum, NoCutoff, exact O(N²) nonbonded path."""
    s = proteon.load(PREPPED_CRAMBIN)
    return proteon.compute_energy(
        s, ff="amber96", units="kJ/mol", nbl_threshold=10**9, nonbonded_cutoff=1e6
    )


@pytest.fixture(scope="module")
def obc_energy():
    """proteon AMBER96+OBC1, same settings."""
    s = proteon.load(PREPPED_CRAMBIN)
    return proteon.compute_energy(
        s, ff="amber96_obc", units="kJ/mol", nbl_threshold=10**9, nonbonded_cutoff=1e6
    )


def _pct(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0) * 100


class TestMollyTopology:
    """Topology must match before any energy comparison is meaningful."""

    def test_counts_match(self, vacuum_energy):
        for key, expected in MOLLY_CRAMBIN_COUNTS.items():
            if key == "n_atoms":
                continue  # proteon reports this as n_topo_atoms
            assert int(vacuum_energy[key]) == expected, (
                f"{key}: proteon {vacuum_energy[key]} vs Molly {expected}. "
                "A count mismatch means the two engines built different "
                "topologies from the same XML — fix that before reading energies."
            )

    def test_atom_count_matches(self, vacuum_energy):
        assert int(vacuum_energy["n_topo_atoms"]) == MOLLY_CRAMBIN_COUNTS["n_atoms"]


class TestMollyExactAgreement:
    """Components where proteon and Molly agree bit-for-bit at print precision.

    Tolerance 0.001% throughout: the measured gap is 0.000%, so anything that
    moves these at all is a regression, not parameter drift. These are the
    components that are now genuinely triangulated (proteon == OpenMM == Molly).
    """

    @pytest.mark.parametrize(
        "component", ["bond_stretch", "angle_bend", "torsion", "vdw"]
    )
    def test_component_matches_exactly(self, vacuum_energy, component):
        got = vacuum_energy[component]
        ref = MOLLY_CRAMBIN_PREPPED[component]
        pct = _pct(got, ref)
        assert pct < 0.001, f"{component}: {got:.6f} vs Molly {ref:.6f} ({pct:.6f}%)"


class TestMollyNonbonded:
    def test_electrostatic_matches(self, vacuum_energy):
        """Tolerance 0.1%: measured 0.019% (~0.95 kJ/mol on -4947 kJ/mol).

        OpenMM and Molly agree to 0.000% on the vdW+electrostatic sum, so this
        small residual is proteon's, not the references' — but it is two orders
        of magnitude inside the 1% total-energy contract and has been stable.
        """
        got = vacuum_energy["electrostatic"]
        ref = MOLLY_CRAMBIN_PREPPED["electrostatic"]
        pct = _pct(got, ref)
        assert pct < 0.1, f"Electrostatic: {got:.4f} vs Molly {ref:.4f} ({pct:.4f}%)"

    def test_total_vacuum_matches(self, vacuum_energy):
        """Tolerance 0.5%: measured 0.143%, almost all of it the improper term."""
        got = vacuum_energy["total"]
        ref = MOLLY_CRAMBIN_PREPPED["total_vacuum"]
        pct = _pct(got, ref)
        assert pct < 0.5, f"Total (vacuum): {got:.4f} vs Molly {ref:.4f} ({pct:.4f}%)"


class TestMollyImproperTorsion:
    """The one force-field component where all three engines disagree.

    Measured on the committed fixture (kJ/mol):

        proteon  128.939
        OpenMM   124.495     (proteon vs OpenMM:  3.57%)
        Molly    116.827     (proteon vs Molly:  10.37%,  OpenMM vs Molly: 6.56%)

    All three build 125 impropers from the same `amber96.xml`, so this is not
    a matching gap of the kind BALL has (BALL finds only 10 — single-wildcard
    patterns — which is a genuine spec violation). Here every engine finds the
    same set and evaluates it differently.

    The spread is geometry-dependent: on a differently-hydrogenated crambin the
    same three engines gave 81.08 / 75.72 / 75.61, where OpenMM and Molly agreed
    to 0.144%. A constant parameter offset cannot produce that; differing
    atom-ordering conventions within the improper 4-tuple can, since the AMBER
    improper dihedral is order-sensitive and its value depends on the local
    geometry.

    So this is asserted as a *documented band*, not equality, per the
    devdocs/ORACLE.md decision tree — with a one-sided guard, since proteon has
    been the high engine in every measurement so far. Narrowing this band is
    real work: it needs the improper 4-tuple orderings compared atom-by-atom
    across the three engines. Tracked as a known convention gap, not a pass.
    """

    def test_improper_within_documented_band(self, vacuum_energy):
        got = vacuum_energy["improper_torsion"]
        ref = MOLLY_CRAMBIN_PREPPED["improper_torsion"]
        pct = _pct(got, ref)
        assert pct < 15.0, (
            f"Improper torsion: {got:.4f} vs Molly {ref:.4f} ({pct:.4f}%). "
            "Expected gap is ~10.4% — see class docstring. A larger gap means "
            "proteon's improper handling moved."
        )

    def test_improper_is_the_high_engine(self, vacuum_energy):
        """One-sided regression guard on the direction of the known gap.

        Mirrors the `proteon >= BALL` guard in test_ball_energy.py: if proteon
        ever drops below Molly, the convention gap changed character and the
        band above needs re-deriving rather than re-widening.
        """
        got = vacuum_energy["improper_torsion"]
        ref = MOLLY_CRAMBIN_PREPPED["improper_torsion"]
        assert got >= ref, (
            f"Improper torsion: proteon {got:.4f} fell below Molly {ref:.4f}. "
            "proteon has been the high engine in every measurement; investigate "
            "rather than adjusting the band."
        )


class TestMollyImplicitSolvent:
    """GB is NOT triangulated by Molly — this test pins the reason.

    Molly ignores `GBSAOBCForce` when parsing the OpenMM XML (it emits
    "GBSAOBCForce not currently supported, ignoring") and builds
    `ImplicitSolventOBC` from its own element-keyed radii and screen tables.
    The comparison below therefore measures how far two *different OBC
    parameterisations* land apart, which is useful context but is not evidence
    about proteon's GB math.

    Measured: proteon -1360.611, OpenMM -1301.782, Molly -1658.240 kJ/mol.
    proteon vs OpenMM 4.5% (inside the ≤5% Phase B contract); proteon vs Molly
    17.9%; OpenMM vs Molly 21.5%. OpenMM stays the authoritative GB oracle.
    """

    def test_solvation_within_parameter_gap_band(self, obc_energy):
        """Tolerance 25%: measured 17.9%, driven by the radii-table difference.

        Wide on purpose. This band catches a catastrophic GB regression (a sign
        flip, a dropped term) while not pretending to validate GB math that
        Molly's parameters cannot speak to.
        """
        got = obc_energy["solvation"]
        ref = MOLLY_CRAMBIN_PREPPED["solvation_obc1"]
        pct = _pct(got, ref)
        assert pct < 25.0, f"Solvation: {got:.4f} vs Molly {ref:.4f} ({pct:.4f}%)"

    def test_solvation_is_stabilising(self, obc_energy):
        """GB solvation of a folded, charged protein must be negative.

        Cheap invariant, but it is the one that would have caught the 2026-04-11
        EEF1 sign bug immediately.
        """
        assert obc_energy["solvation"] < 0
