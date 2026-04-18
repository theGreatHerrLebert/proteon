"""Shared pytest fixtures + registry for the ferritin test suite.

This file is the single source of truth for:
  * which force fields we test      → FORCE_FIELDS
  * which execution paths we test   → PATHS
  * which reference structures      → STRUCTURES

Any new test file should consume these constants (or the ready-made
pytest parameter lists derived from them) rather than hardcoding its
own lists. Adding a new force field, a new reference PDB, or a new
code path is then a single-line edit here — every parametrized test
automatically grows its coverage matrix.

Why this file exists
--------------------
The 2026-04-11 CHARMM+EEF1 bugs demonstrated that ad-hoc test files
with hardcoded V1 lists can't keep up with growing coverage needs.
test_charmm_invariants.py had its own `V1_PDBS = [...]` constant;
test_ball_energy.py had its own crambin path. Neither file tested the
force field × code path cross product. Bug #2 in compute_energy_and_forces_nbl
(missing EEF1 call) hid for months because only 1ake (3317 atoms) of
the 4 V1 PDBs crossed NBL_AUTO_THRESHOLD, and the single invariant
that could have caught it, TestSolvationActive, was checked through
the default code path only — so 11 of 12 invariant cases on CHARMM
never touched the NBL path at all.

The fix is this registry: every test parametrizes over the full
matrix, so the moment we add a new force field or a new reference
structure, every invariant, parity check, and gradient test auto-
expands. Adding charmm36 later will mean appending one line below,
nothing else.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest

import ferritin


# ---------------------------------------------------------------------------
# Registry: force fields, code paths, reference structures
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PDBS_DIR = os.path.join(REPO_ROOT, "test-pdbs")


#: All force fields to run parametrized invariants against. Extending
#: this list is how new FFs are pulled into the test matrix.
FORCE_FIELDS: List[str] = ["amber96", "charmm19_eef1"]


#: Code paths to exercise. The `nbl_threshold` value controls which
#: path compute_energy_auto picks:
#:   - None  → library default (NBL_AUTO_THRESHOLD = 2000 atoms)
#:   - 0     → always NBL (every structure, regardless of size)
#:   - 10**7 → never NBL (always the O(N²) exact path)
#:
#: Parametrizing over all three catches bugs where one path drifts
#: from the other on a specific input regime. The 2026-04-11 bug in
#: compute_energy_and_forces_nbl would have failed every "force_nbl"
#: case on CHARMM.
PATHS: List[Tuple[str, Optional[int]]] = [
    ("default", None),
    ("force_nbl", 0),
    ("forbid_nbl", 10_000_000),
]


@dataclass(frozen=True)
class StructureSpec:
    """Metadata for a reference PDB used in parametrized tests."""

    name: str
    path: str
    atoms: int  # approx heavy + H count; used for size-regime assertions
    description: str  # human context, also shown in test IDs

    @property
    def absolute_path(self) -> str:
        return os.path.join(TEST_PDBS_DIR, os.path.basename(self.path))


#: Reference structures for the v1 invariant suite. Chosen to span
#: the NBL_AUTO_THRESHOLD (2000 atoms) so both code paths are
#: exercised by the default parametrization. Adding a new PDB here
#: expands every parametrized test to cover it.
STRUCTURES: List[StructureSpec] = [
    StructureSpec(
        name="1crn",
        path="1crn.pdb",
        atoms=327,
        description="crambin, 46 residues, smallest canonical test protein",
    ),
    StructureSpec(
        name="1ubq",
        path="1ubq.pdb",
        atoms=602,
        description="ubiquitin, 76 residues, small globular benchmark",
    ),
    StructureSpec(
        name="1bpi",
        path="1bpi.pdb",
        atoms=454,
        description="BPTI, 58 residues, disulfide-rich",
    ),
    StructureSpec(
        name="1ake",
        path="1ake.pdb",
        atoms=3317,
        description="adenylate kinase, 214 residues, LARGE — crosses NBL_AUTO_THRESHOLD",
    ),
]


# ---------------------------------------------------------------------------
# Pre-built parameter lists for pytest.mark.parametrize / indirect fixtures.
# Prefer these over re-computing the product inline in each test file.
# ---------------------------------------------------------------------------

#: (structure, force_field) — for tests that don't care about the
#: code path (invariants that should hold regardless).
STRUCT_FF_PARAMS = [
    pytest.param(s, ff, id=f"{s.name}-{ff}")
    for s, ff in itertools.product(STRUCTURES, FORCE_FIELDS)
]

#: (structure, force_field, path_tuple) — full matrix. Use when the
#: code path is part of what's being tested (parity, coverage of both
#: compute_energy_impl and compute_energy_nbl).
STRUCT_FF_PATH_PARAMS = [
    pytest.param(s, ff, path, id=f"{s.name}-{ff}-{path[0]}")
    for s, ff, path in itertools.product(STRUCTURES, FORCE_FIELDS, PATHS)
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures: load each PDB once per test run, not once
# per test. Saves measurable time on the full suite.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def loaded_structures():
    """Dict mapping StructureSpec.name → loaded ferritin.Structure.

    Loaded once per pytest session. Tests that mutate coordinates
    should do a fresh ``ferritin.load(...)`` instead of using this
    fixture, to avoid cross-test contamination.
    """
    return {s.name: ferritin.load(s.absolute_path) for s in STRUCTURES}


@pytest.fixture(scope="session")
def v1_energies(loaded_structures):
    """Dict {(structure_name, ff, path_id): energy_dict} covering the
    full STRUCT_FF_PATH cross product, computed once per session.

    Energies are in kJ/mol, matching the existing CHARMM invariant
    suite convention. Tests that want kcal/mol or a different unit
    should call compute_energy themselves rather than use this
    fixture.
    """
    out = {}
    for s in STRUCTURES:
        structure = loaded_structures[s.name]
        for ff in FORCE_FIELDS:
            for path_id, nbl_threshold in PATHS:
                out[(s.name, ff, path_id)] = ferritin.compute_energy(
                    structure,
                    ff=ff,
                    units="kJ/mol",
                    nbl_threshold=nbl_threshold,
                )
    return out


# ---------------------------------------------------------------------------
# Helpers used by multiple test modules
# ---------------------------------------------------------------------------


#: All energy component keys returned by compute_energy (for CHARMM
#: and AMBER both — AMBER has zero solvation, but the key is present).
ENERGY_COMPONENTS = (
    "bond_stretch",
    "angle_bend",
    "torsion",
    "improper_torsion",
    "vdw",
    "electrostatic",
    "solvation",
)


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------
#
# `oracle(tool)`: every externally-anchored test ("does ferritin agree with
# OpenMM / BALL / Biopython / MMseqs2 / …?") is tagged with
# @pytest.mark.oracle("<tool>"), so you can run just the oracle layer with:
#
#     pytest -m oracle                       # all oracles
#     pytest -m oracle -k usalign            # tool filter via test-id keyword
#
# and inspect the coverage with:
#
#     pytest --collect-only -m oracle -q
#
# `slow`: tests that take >10s wall-clock (corpus downloads, full-suite
# parity sweeps, large PDB pipelines). Skip them in the dev inner loop with:
#
#     pytest -m "not slow and not oracle"    # fast feedback cycle
#     pytest -m slow                         # run only the slow ones
#
# See `tests/oracle/README.md` for oracle conventions and `devdocs/ORACLE.md`
# for the full philosophy.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "oracle(tool): test compares ferritin against an independent "
        "external implementation (BALL, OpenMM, Biopython, Gemmi, "
        "FreeSASA, DSSP, MMseqs2, USAlign, …). Slow and may require "
        "installing the oracle. Filter with `pytest -m oracle`.",
    )
    config.addinivalue_line(
        "markers",
        "slow: test takes >10s wall-clock (corpus downloads, full-suite "
        "parity sweeps, large PDB pipelines). Skip in dev loop with "
        "`pytest -m 'not slow'`.",
    )
