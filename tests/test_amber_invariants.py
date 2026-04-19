"""Internal invariant tests for AMBER96.

Why this file exists
--------------------
AMBER96 is proteon's well-oracled force field (test_ball_energy.py
validates to ~0.02% against BALL Julia on heavy-atom crambin), but
it was never run through the cross-path × invariant suite that
test_charmm_invariants.py applies to CHARMM19+EEF1. That's the
same coverage asymmetry that let the 2026-04-11 CHARMM bugs ship:
one force field gets a rigorous oracle, the other gets invariants,
and neither gets BOTH.

This file closes the asymmetry from the other side. Every invariant
that makes sense for a vacuum force field (no implicit solvation
term) runs on AMBER96 across the same (structure × code path)
matrix that CHARMM uses. The solvation-specific invariants are
replaced with AMBER-specific ones ("solvation must be exactly 0",
because AMBER96 has no implicit solvent).

What these tests do NOT do
--------------------------
They do NOT validate parameter correctness — that's what the
BALL Julia oracle does. These internal invariants are a cheap
companion to the oracle: they run on every pytest invocation
(the oracle needs Julia installed) and they cover every PDB in
the registry rather than just crambin.
"""

from __future__ import annotations

import math

import pytest

import proteon

from conftest import (
    ENERGY_COMPONENTS,
    PATHS,
    STRUCTURES,
)


AMBER = "amber96"


# Vacuum AMBER96 has no implicit solvation, so the solvation slot
# must be exactly 0.0. Every other component is expected to be
# non-trivial on a real protein.
AMBER_COMPONENTS_NONZERO = (
    "bond_stretch",
    "angle_bend",
    "torsion",
    # improper_torsion may be zero for some structures — don't require non-zero
    "vdw",
    "electrostatic",
)


_STRUCT_PATH_PARAMS = [
    pytest.param((s, path), id=f"{s.name}-{path[0]}")
    for s in STRUCTURES
    for path in PATHS
]


@pytest.fixture(params=_STRUCT_PATH_PARAMS)
def amber_energy(request):
    """(name, energy_dict) parametrized over (STRUCTURE × PATH), AMBER96."""
    structure_spec, (_path_id, nbl_threshold) = request.param
    s = proteon.load(structure_spec.absolute_path)
    e = proteon.compute_energy(
        s, ff=AMBER, units="kJ/mol", nbl_threshold=nbl_threshold
    )
    return structure_spec.name, e


# =========================================================================
# Numerical sanity
# =========================================================================


class TestNoNanInf:
    """Every energy component is finite (no NaN, no Inf)."""

    def test_total_is_finite(self, amber_energy):
        name, e = amber_energy
        assert math.isfinite(e["total"]), f"{name}: total is {e['total']}"

    def test_components_are_finite(self, amber_energy):
        name, e = amber_energy
        for comp in ENERGY_COMPONENTS:
            v = e.get(comp)
            assert v is not None, f"{name}: component {comp} is missing"
            assert math.isfinite(v), f"{name}: {comp} is {v}"


class TestComponentsPresent:
    """All seven component keys plus `total` are present in the dict —
    identical to the CHARMM test, because the schema is shared.
    """

    def test_all_keys_present(self, amber_energy):
        name, e = amber_energy
        for comp in ENERGY_COMPONENTS:
            assert comp in e, f"{name}: missing key {comp!r}"
        assert "total" in e, f"{name}: missing key 'total'"


class TestSumsMatchTotal:
    """Σ(components) ≈ total. Catches accounting bugs like the
    2026-04-11 compute_energy_and_forces_nbl regression where
    solvation was silently omitted from the total sum. On AMBER96
    the solvation is always 0 so omitting it wouldn't diverge, but
    this test would still catch any other missing component.
    """

    def test_components_sum_to_total(self, amber_energy):
        name, e = amber_energy
        component_sum = sum(e[c] for c in ENERGY_COMPONENTS)
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
    """Harmonic potentials are sums of squares — they MUST be ≥ 0."""

    def test_bond_stretch_nonnegative(self, amber_energy):
        name, e = amber_energy
        assert e["bond_stretch"] >= 0, (
            f"{name}: bond_stretch {e['bond_stretch']:.6f} is negative — "
            "harmonic potential should always be ≥ 0"
        )

    def test_angle_bend_nonnegative(self, amber_energy):
        name, e = amber_energy
        assert e["angle_bend"] >= 0, (
            f"{name}: angle_bend {e['angle_bend']:.6f} is negative — "
            "harmonic potential should always be ≥ 0"
        )


class TestSolvationIsZero:
    """AMBER96 is a vacuum force field — the solvation slot MUST be
    exactly 0. Any non-zero value means EEF1 (or some other implicit
    solvation) has leaked into the AMBER path.
    """

    def test_solvation_exactly_zero(self, amber_energy):
        name, e = amber_energy
        assert e["solvation"] == 0.0, (
            f"{name}: AMBER96 solvation = {e['solvation']:.6f}, expected "
            "exactly 0.0 (AMBER96 is vacuum — no implicit solvation). "
            "A non-zero value means implicit solvent has leaked into "
            "the AMBER code path."
        )


class TestBondedTermsActive:
    """Every real protein must produce non-trivial bonded-term energies.
    If any of bond/angle/torsion/vdw/electrostatic is exactly zero,
    the corresponding kernel silently fell through — the topology
    builder found no bonds, or the parameter lookup returned all
    None, etc. This would not be caught by TestNoNanInf (0 is finite)
    or TestSigns (0 ≥ 0).
    """

    def test_bonded_terms_nontrivial(self, amber_energy):
        name, e = amber_energy
        for comp in AMBER_COMPONENTS_NONZERO:
            assert abs(e[comp]) > 1e-6, (
                f"{name}: AMBER96 {comp} is {e[comp]:.9f}, expected "
                f"non-trivial magnitude. Zero suggests the kernel "
                f"silently fell through."
            )


# =========================================================================
# AMBER vs CHARMM distinctness (regression for silent fallback)
# =========================================================================


class TestAmberDistinctFromCharmm:
    """Mirror of test_charmm_invariants.TestCharmmDistinctFromAmber.
    If AMBER and CHARMM return the same number on the same structure,
    one is silently falling back to the other. Distinct from the
    CHARMM version because we also want to check from AMBER's side,
    and on every structure, not just 1crn.
    """

    @pytest.mark.parametrize(
        "spec",
        STRUCTURES,
        ids=[s.name for s in STRUCTURES],
    )
    def test_amber_total_differs_from_charmm(self, spec):
        s = proteon.load(spec.absolute_path)
        e_amber = proteon.compute_energy(s, ff=AMBER)
        e_charmm = proteon.compute_energy(s, ff="charmm19_eef1")
        assert e_amber["total"] != e_charmm["total"], (
            f"{spec.name}: AMBER and CHARMM returned identical totals — "
            "one force field is silently falling back to the other"
        )
