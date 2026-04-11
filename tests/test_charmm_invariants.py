"""Internal invariant tests for CHARMM19+EEF1.

Why this file exists
--------------------
CHARMM19+EEF1 has been the production default for `batch_prepare` since
commit 73248f7, but has NO external oracle in the repo. The AMBER96
oracle (tests/oracle/test_ball_energy.py) validates AMBER96 against the
BALL Julia implementation to 0.02% on heavy-only 1crn — but AMBER96 is
not what users actually run. The well-validated FF is the one nobody
uses; the FF that runs on every batch_prepare call is checked only by
"the number is not zero".

This file fills the gap by enforcing INVARIANTS — properties that must
hold regardless of force-field parameter values. Any violation indicates
a real bug. These tests catch:

  - NaN / Inf in any energy component (numerical blowup)
  - Missing components (CHARMM19-EEF1 should produce all seven)
  - Σ(components) ≠ total (component-accounting bug)
  - Negative bond/angle energy (sign error in a harmonic potential)
  - Zero solvation (EEF1 silently fell back to vacuum CHARMM)
  - Non-determinism (same input → different output)
  - Energy increasing under minimization (minimizer broken)
  - CHARMM and AMBER giving the same answer (one is silently falling
    back to the other)

What these tests do NOT do
--------------------------
They do NOT validate parameter correctness — wrong K_b values for a
specific bond type, wrong charges on a specific atom, etc. — because
that requires an external oracle. The "external CHARMM19-EEF1 oracle"
is a known gap (Tier-3 work in the SOTA validation roadmap). These
internal invariants are a strict subset of what an oracle would catch.
"""

import math
import os

import pytest

import ferritin

from conftest import (
    ENERGY_COMPONENTS as CHARMM_COMPONENTS,
    PATHS,
    STRUCTURES,
    TEST_PDBS_DIR,
)


# Force-field name used by ferritin's `compute_energy(ff=...)` API.
CHARMM = "charmm19_eef1"


# Generate the (structure × path) matrix from the shared registry.
# Adding a new structure or a new path in conftest.py automatically
# grows this fixture's parameter set without touching this file.
# Each param is a single (structure_spec, path_tuple) value — pytest
# fixture params want one object per entry, unlike parametrize.
_STRUCT_PATH_PARAMS = [
    pytest.param((s, path), id=f"{s.name}-{path[0]}")
    for s in STRUCTURES
    for path in PATHS
]


@pytest.fixture(params=_STRUCT_PATH_PARAMS)
def charmm_energy(request):
    """(name, energy_dict) tuple, parametrized over (STRUCTURE × PATH).

    Uses `compute_energy` (not `batch_prepare`) so the input geometry is
    the raw PDB — no H placement, no minimization. This is the cleanest
    "is the energy kernel correct on a known input" measurement.

    The code-path dimension ensures every invariant runs on BOTH the
    O(N²) exact path and the neighbor-list path. Without this, the NBL
    implementation could silently diverge from the exact one as long as
    no default-sized test PDB happened to cross NBL_AUTO_THRESHOLD —
    which is exactly how the 2026-04-11 EEF1-in-NBL-path bug stayed
    hidden until the Tier-2 oracle surfaced it.
    """
    structure_spec, (_path_id, nbl_threshold) = request.param
    s = ferritin.load(structure_spec.absolute_path)
    e = ferritin.compute_energy(
        s, ff=CHARMM, units="kJ/mol", nbl_threshold=nbl_threshold
    )
    return structure_spec.name, e


# =========================================================================
# Numerical sanity
# =========================================================================


class TestNoNanInf:
    """Every energy component is finite (no NaN, no Inf)."""

    def test_total_is_finite(self, charmm_energy):
        name, e = charmm_energy
        assert math.isfinite(e["total"]), f"{name}: total is {e['total']}"

    def test_components_are_finite(self, charmm_energy):
        name, e = charmm_energy
        for comp in CHARMM_COMPONENTS:
            v = e.get(comp)
            assert v is not None, f"{name}: component {comp} is missing"
            assert math.isfinite(v), f"{name}: {comp} is {v}"


class TestComponentsPresent:
    """All seven CHARMM components plus `total` are present in the dict."""

    def test_all_keys_present(self, charmm_energy):
        name, e = charmm_energy
        for comp in CHARMM_COMPONENTS:
            assert comp in e, f"{name}: missing key {comp!r}"
        assert "total" in e, f"{name}: missing key 'total'"


class TestSumsMatchTotal:
    """Σ(components) ≈ total — catches accounting bugs like dropping a
    component from the sum.

    Tolerance: max(1e-3 * |total|, 1e-3) kJ/mol. Relative for large
    energies (a 10000 kJ/mol total can drift ~10 kJ/mol from float
    accumulation), absolute floor for small energies.
    """

    def test_components_sum_to_total(self, charmm_energy):
        name, e = charmm_energy
        component_sum = sum(e[c] for c in CHARMM_COMPONENTS)
        diff = abs(component_sum - e["total"])
        tol = max(1e-3 * abs(e["total"]), 1e-3)
        assert diff < tol, (
            f"{name}: Σ(components) {component_sum:.6f} != total "
            f"{e['total']:.6f} (diff {diff:.4e}, tol {tol:.4e})"
        )


# =========================================================================
# Physical sanity
# =========================================================================


class TestSigns:
    """Harmonic potentials are sums of squares — they MUST be ≥ 0.
    A negative value would indicate a sign error in the energy kernel.
    """

    def test_bond_stretch_nonnegative(self, charmm_energy):
        name, e = charmm_energy
        assert e["bond_stretch"] >= 0, (
            f"{name}: bond_stretch {e['bond_stretch']:.6f} is negative — "
            "harmonic potential should always be ≥ 0"
        )

    def test_angle_bend_nonnegative(self, charmm_energy):
        name, e = charmm_energy
        assert e["angle_bend"] >= 0, (
            f"{name}: angle_bend {e['angle_bend']:.6f} is negative — "
            "harmonic potential should always be ≥ 0"
        )


class TestSolvationNegative:
    """EEF1 self-solvation MUST be negative for any folded protein.

    Why this exists (regression test for the 2026-04-10 sign bug):
    The Tier-2 weak oracle (commit 8f979f4) caught that ferritin's
    CHARMM19+EEF1 returns POSITIVE solvation on every v1 PDB while
    OpenMM's CHARMM36+OBC2 returns NEGATIVE. Canonical EEF1 self-
    solvation for 1crn computed independently from the .ini parameter
    file is -2748 kJ/mol; ferritin reports +537 kJ/mol. The bug was
    invisible because the existing test_solvation_nontrivial only
    checks |solvation| > 1.0 — never the sign.

    This is a SOFT assertion: it depends on the convention that EEF1's
    "solvation" reports the canonical Lazaridis-Karplus ΔG_solv, where
    polar atoms (peptide N, carbonyl O, OH) contribute large negative
    terms (favorable solvation) and hydrophobic CH3 contribute small
    positives. For a real folded protein the polar terms dominate,
    so the total Σ dg_ref is always negative under this convention.

    If a future ferritin maintainer deliberately changes the sign
    convention (e.g., reports the negation, or some "lost solvation"
    quantity), update this test AND the SOTA aggregator's
    `solvation_sign_agree` check in compare_energy_weak together —
    they encode the same assumption.
    """

    def test_solvation_negative(self, charmm_energy):
        name, e = charmm_energy
        assert e["solvation"] < 0, (
            f"{name}: solvation = {e['solvation']:.6f} kJ/mol is "
            f"non-negative. EEF1 should produce a NEGATIVE total "
            f"solvation free energy for folded proteins (canonical "
            f"Lazaridis-Karplus convention). For 1crn, the canonical "
            f"value computed directly from charmm19_eef1.ini is "
            f"≈ -2748 kJ/mol. See "
            f"validation/sota_comparison/diagnose_charmm_eef1.py for "
            f"the regression that caught this in the first place."
        )


class TestSolvationActive:
    """EEF1 implicit solvation must contribute non-trivially.

    If solvation == 0 across a real protein, EEF1 silently fell back to
    vacuum CHARMM and the production default is broken. The threshold
    of 1.0 kJ/mol is far below what any real protein produces — buried
    hydrophobics and exposed polars give tens to hundreds of kJ/mol —
    so it only catches the "EEF1 isn't running at all" failure mode,
    not "EEF1 has slightly wrong parameters".
    """

    def test_solvation_nontrivial(self, charmm_energy):
        name, e = charmm_energy
        assert abs(e["solvation"]) > 1.0, (
            f"{name}: solvation = {e['solvation']:.6f} kJ/mol — EEF1 "
            "should produce a non-trivial implicit-solvation contribution"
        )


# =========================================================================
# Determinism
# =========================================================================


class TestDeterminism:
    """Same input → identical output across two compute_energy calls.

    Catches non-deterministic accumulation order, uninitialized memory,
    and parallelism with race conditions in the energy kernel. The
    tolerance is exact equality (abs=0) because compute_energy on the
    same input should be bit-identical, not just numerically close.

    Historical note: until 2026-04-12 this test was marked xfail for
    a 1 ULP drift on ``bond_stretch``. The root cause was
    ``build_topology`` iterating a ``HashMap<usize, Vec<usize>>`` of
    residue-to-atom-indices — HashMap iteration order in Rust is
    non-deterministic across instances, so each compute_energy call
    pushed bonds into ``topo.bonds`` in a different order, and the
    serial sum saw a different FP accumulation order. Fixed by
    switching to BTreeMap (deterministic by type). If this ever goes
    xfail again, investigate whether some other HashMap in topology
    construction has slipped in.
    """

    @pytest.mark.parametrize(
        "spec",
        STRUCTURES,
        ids=[s.name for s in STRUCTURES],
    )
    def test_two_calls_identical(self, spec):
        s = ferritin.load(spec.absolute_path)
        e1 = ferritin.compute_energy(s, ff=CHARMM)
        e2 = ferritin.compute_energy(s, ff=CHARMM)
        for key in CHARMM_COMPONENTS + ("total",):
            assert e1[key] == e2[key], (
                f"{spec.name}: {key} not deterministic — "
                f"call1={e1[key]!r}, call2={e2[key]!r}"
            )


# =========================================================================
# Minimization
# =========================================================================


class TestMinimizationDecreases:
    """LBFGS minimization must end at energy ≤ start.

    If final > initial, the minimizer is broken (wrong gradient sign,
    bad step acceptance, etc.). LBFGS line search can briefly land
    above the start during a search step, but the final accepted state
    should always be at or below the initial energy.
    """

    @pytest.mark.parametrize(
        "spec",
        STRUCTURES,
        ids=[s.name for s in STRUCTURES],
    )
    def test_final_not_above_initial(self, spec):
        s = ferritin.load(spec.absolute_path)
        reports = ferritin.batch_prepare(
            [s],
            reconstruct=False,
            hydrogens="all",
            minimize=True,
            minimize_method="lbfgs",
            # 200 steps is enough to reach a stable minimum on a small
            # protein and keeps the test suite fast.
            minimize_steps=200,
            gradient_tolerance=0.1,
            strip_hydrogens=False,
            ff=CHARMM,
        )
        r = reports[0]
        # Allow a tiny numerical overshoot: 0.1 kJ/mol absolute or
        # 0.01% relative, whichever is larger.
        tol = max(0.1, 1e-4 * abs(r.initial_energy))
        assert r.final_energy <= r.initial_energy + tol, (
            f"{spec.name}: final_energy {r.final_energy:.4f} > "
            f"initial_energy {r.initial_energy:.4f} (tol {tol:.4f}) — "
            "LBFGS minimization is not decreasing the energy"
        )


class TestMinimizedTotalIsNegative:
    """CHARMM19+EEF1 minimized total energy MUST be negative on any
    folded protein (canonical polar-H united-atom convention).

    Pre-fix rationale: the 2026-04-12 batch of fixes (EEF1 bugs, water,
    1bpi, polar-H, heavy-atom freeze) incrementally moved totals from
    spurious-positive to physically-correct negative. Before the final
    heavy-atom-freeze fix, CHARMM19 minimized totals were
    wrong-signed on 3 of 4 v1 PDBs because the minimizer was freezing
    heavy atoms (a sensible default for AMBER96's explicit-H model but
    wrong for CHARMM19's inflated united-atom radii). This test would
    have caught that regression immediately — every step along the
    way a single wrong-signed PDB turns the suite red.

    The negative-total invariant is a STRONGER check than the
    TestSolvationNegative invariant: solvation can be negative while
    the total remains positive if the bonded/nonbonded terms
    overwhelm it. For a minimized small protein under CHARMM19+EEF1
    the total should always settle negative.
    """

    @pytest.mark.parametrize(
        "spec",
        STRUCTURES,
        ids=[s.name for s in STRUCTURES],
    )
    def test_final_total_negative(self, spec):
        s = ferritin.load(spec.absolute_path)
        reports = ferritin.batch_prepare(
            [s],
            reconstruct=False,
            hydrogens="all",
            minimize=True,
            minimize_method="lbfgs",
            # 500 steps: enough for v1 PDBs to reach sign-correct
            # territory. Full convergence (gradient < 0.1) typically
            # takes 1000-2000 steps; we don't require convergence here,
            # only that the minimizer reaches the sign-correct regime.
            minimize_steps=500,
            gradient_tolerance=0.1,
            strip_hydrogens=False,
            ff=CHARMM,
        )
        r = reports[0]
        assert r.final_energy < 0, (
            f"{spec.name}: minimized CHARMM19+EEF1 total = "
            f"{r.final_energy:+.2f} kJ/mol, expected negative. "
            "A folded protein under a polar-H united-atom force field "
            "with EEF1 implicit solvation should always settle to a "
            "negative total. If this regression fires, the most likely "
            "culprit is the batch_prepare `constrain_heavy` default "
            "having flipped back to True for CHARMM19 — see the "
            "2026-04-12 commit for context."
        )


# =========================================================================
# CHARMM is actually running (not silently falling back to AMBER)
# =========================================================================


class TestCharmmDistinctFromAmber:
    """If CHARMM and AMBER return the same number, one of them is
    silently falling back to the other. Sanity check that we're
    actually evaluating CHARMM19-EEF1 when we ask for it.
    """

    def test_charmm_total_differs_from_amber(self):
        s = ferritin.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))
        e_charmm = ferritin.compute_energy(s, ff=CHARMM)
        e_amber = ferritin.compute_energy(s, ff="amber96")
        assert e_charmm["total"] != e_amber["total"], (
            "CHARMM and AMBER returned identical totals on 1crn — "
            "one force field is silently falling back to the other"
        )

    def test_charmm_has_solvation_amber_does_not(self):
        """The most distinctive signal: AMBER96 in vacuum has no
        solvation term, CHARMM19+EEF1 always does. If both have the
        same solvation value, EEF1 is not active.
        """
        s = ferritin.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))
        e_charmm = ferritin.compute_energy(s, ff=CHARMM)
        e_amber = ferritin.compute_energy(s, ff="amber96")
        assert e_charmm["solvation"] != e_amber["solvation"], (
            f"CHARMM solvation ({e_charmm['solvation']}) == AMBER "
            f"solvation ({e_amber['solvation']}) — EEF1 is not running"
        )
