"""Cross-path parity: the NBL and exact code paths must agree to 1e-6
kcal/mol on every energy component, for every force field, on every
reference structure.

This is the highest-leverage regression guard in the proteon test
suite. Both 2026-04-11 CHARMM+EEF1 bugs would have failed this test
on day one:

  Bug #1: eef1_energy() missing 1-2/1-3 exclusions. Both paths have
      the same wrong formula, so this one doesn't diverge — but the
      invariant suite's TestSolvationNegative would still catch it
      via the sign check, so defence in depth holds.

  Bug #2: compute_energy_and_forces_nbl() never called EEF1 at all,
      leaving solvation = 0 while compute_energy_impl returned the
      real (buggy pre-fix, correct post-fix) value. The divergence
      is O(10³) kcal/mol — unmissable.

The Rust side has the same parity test in
proteon-connector/src/forcefield/energy.rs gradient_tests module,
but the Python version exercises the full compute_energy binding
path including the kJ/mol unit conversion and dict assembly — so
both sides close the loop from API down to kernel.
"""

from __future__ import annotations

import math

import pytest

import proteon

from conftest import STRUCT_FF_PARAMS, ENERGY_COMPONENTS


# Parity tolerance. 1e-6 kcal/mol is strict enough to catch any real
# bug (NBL vs exact drift is either zero or O(10⁰+) kcal/mol — there
# is no physical process producing drift in between), and loose enough
# to tolerate FP reassociation across the two code paths.
TOL_KJ_MOL = 1e-6 * 4.184  # 1e-6 kcal/mol in kJ/mol


@pytest.mark.parametrize("structure_spec,ff", STRUCT_FF_PARAMS)
def test_nbl_matches_exact_all_components(structure_spec, ff, loaded_structures):
    """compute_energy with nbl_threshold=0 must match nbl_threshold=huge
    to TOL_KJ_MOL on every component.

    The default code path will follow one or the other depending on
    atom count vs NBL_AUTO_THRESHOLD, so testing both forced regimes
    pins the parity independent of the heuristic.
    """
    structure = loaded_structures[structure_spec.name]

    e_exact = proteon.compute_energy(
        structure, ff=ff, units="kJ/mol", nbl_threshold=10_000_000
    )
    e_nbl = proteon.compute_energy(
        structure, ff=ff, units="kJ/mol", nbl_threshold=0
    )

    mismatches = []
    for comp in ENERGY_COMPONENTS + ("total",):
        exact = e_exact[comp]
        nbl = e_nbl[comp]
        # Finite guard — non-finite on either side is its own failure
        # category, handled by the invariant tests. Here we only check
        # parity. NaN != NaN would spuriously trip this assertion.
        if not (math.isfinite(exact) and math.isfinite(nbl)):
            continue
        diff = abs(exact - nbl)
        # Relative tolerance on the absolute magnitude, plus TOL_KJ_MOL
        # absolute floor. Large components (electrostatic on clashy raw
        # PDBs is O(10⁵) kJ/mol) get a proportionally looser bar.
        tol = max(TOL_KJ_MOL, 1e-9 * abs(exact))
        if diff > tol:
            mismatches.append((comp, exact, nbl, diff, tol))

    if mismatches:
        lines = [
            f"{structure_spec.name} / {ff}: NBL vs exact path divergence:",
        ]
        for comp, exact, nbl, diff, tol in mismatches:
            lines.append(
                f"  {comp:<18} exact={exact:+.9f} nbl={nbl:+.9f} "
                f"diff={diff:.3e} tol={tol:.3e}"
            )
        pytest.fail("\n".join(lines))


@pytest.mark.parametrize("structure_spec,ff", STRUCT_FF_PARAMS)
def test_default_path_matches_one_of_the_forced_paths(
    structure_spec, ff, loaded_structures
):
    """The library's default NBL threshold (2000 atoms) must pick
    exactly one of the two forced paths — whichever is appropriate
    for the structure size. This test documents the heuristic and
    makes it a test-enforced contract.

    Any structure with ≤ 2000 atoms must match ``forbid_nbl`` exactly;
    any structure with > 2000 atoms must match ``force_nbl`` exactly.
    """
    structure = loaded_structures[structure_spec.name]
    e_default = proteon.compute_energy(structure, ff=ff, units="kJ/mol")
    e_exact = proteon.compute_energy(
        structure, ff=ff, units="kJ/mol", nbl_threshold=10_000_000
    )
    e_nbl = proteon.compute_energy(
        structure, ff=ff, units="kJ/mol", nbl_threshold=0
    )

    # Which forced path should default match?
    # Library constant: NBL_AUTO_THRESHOLD = 2000.
    expected = e_nbl if structure_spec.atoms > 2000 else e_exact
    expected_name = "force_nbl" if structure_spec.atoms > 2000 else "forbid_nbl"

    # Tolerance rather than strict equality: there is a pre-existing 1 ULP
    # FP reordering across separate compute_energy calls (TestDeterminism
    # failures in test_charmm_invariants.py, unrelated to path selection).
    # A real path divergence would be O(10⁰+) kJ/mol, far above the ULP
    # noise floor — so this tolerance still catches every meaningful
    # regression without being flaky on the ULP noise.
    for comp in ENERGY_COMPONENTS + ("total",):
        if not (math.isfinite(e_default[comp]) and math.isfinite(expected[comp])):
            continue
        diff = abs(e_default[comp] - expected[comp])
        tol = max(TOL_KJ_MOL, 1e-9 * abs(expected[comp]))
        assert diff < tol, (
            f"{structure_spec.name} / {ff}: default path diverges from "
            f"{expected_name} on {comp}: default={e_default[comp]:+.9f}, "
            f"{expected_name}={expected[comp]:+.9f}, diff={diff:.3e}, "
            f"tol={tol:.3e}. Either NBL_AUTO_THRESHOLD changed or the "
            "path selection logic regressed."
        )
