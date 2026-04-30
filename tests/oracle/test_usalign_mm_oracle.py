"""Oracle tests: multi-chain TM-score parity vs C++ USAlign -mm 1.

Companion to the single-chain test_tmscore_oracle.py. Exercises the
mm_align_pair entry on the proteon-connector side and compares against
USAlign's `-mm 1 -outfmt 2` tabular output. Backed by the
proteon-usalign-mm-vs-cpp-ci claim in
`evident/claims/usalign_mm.yaml`.

Skips silently when the C++ USAlign binary is not on PATH (or at
$USALIGN_BIN), matching the existing single-chain test pattern.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import NamedTuple, Optional

import pytest

from proteon_connector import py_align_funcs, py_io

pytestmark = pytest.mark.oracle("usalign")


USALIGN_BIN = os.environ.get("USALIGN_BIN", "USalign")


# Fixture pool root. The proteon repo's test-pdbs/ has the small
# heavy-atom monomers; the /scratch/TMAlign/test-pdbs/ host directory
# (the staging-parent layout on the dev box) carries the multi-chain
# ones we need here. Tests skip rather than fail when the host path
# is not present — CI environments without that mount drop these
# tests, the in-tree ones still run.
HOST_TEST_PDBS = "/scratch/TMAlign/test-pdbs"


class CppMMAlignResult(NamedTuple):
    tm_complex: float
    rmsd: float
    n_aligned: int


def _resolve_usalign() -> Optional[str]:
    """Return path to USalign or None when not installed."""
    if os.path.exists(USALIGN_BIN):
        return USALIGN_BIN
    found = shutil.which(USALIGN_BIN)
    return found


def _run_cpp_usalign_mm(path1: str, path2: str) -> Optional[CppMMAlignResult]:
    """Run `USalign -mm 1 -outfmt 2` and parse the tabular row."""
    bin_path = _resolve_usalign()
    if bin_path is None:
        return None
    try:
        result = subprocess.run(
            [bin_path, "-mm", "1", path1, path2, "-outfmt", "2"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    for line in result.stdout.strip().split("\n"):
        if line.startswith("#") or not line:
            continue
        parts = line.split("\t")
        # Tabular schema (USalign -outfmt 2): #PDBchain1 PDBchain2
        # TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali. Index 2 is TM
        # normalised by chain1 length, which on `-mm 1` is the
        # whole-complex TM normalised by query-complex length.
        if len(parts) >= 11:
            try:
                return CppMMAlignResult(
                    tm_complex=float(parts[2]),
                    rmsd=float(parts[4]),
                    n_aligned=int(parts[10]),
                )
            except ValueError:
                continue
    return None


def _pdb_path(name: str) -> str:
    """Resolve a PDB name to an absolute path on the host fixture pool."""
    return os.path.join(HOST_TEST_PDBS, f"{name}.pdb")


def _skip_if_fixture_missing(*names: str) -> None:
    for n in names:
        p = _pdb_path(n)
        if not os.path.isfile(p):
            pytest.skip(
                f"Multi-chain fixture {p} not available; tests run on hosts "
                f"with /scratch/TMAlign/test-pdbs/ mounted"
            )


# ---- Fixture pairs ---------------------------------------------------------

# (id, pdb1, pdb2, expected_tm_floor)
# expected_tm_floor is a soft sanity check: TM should be at least this
# value on the corresponding pair. Self-alignments are 1.0; the cross
# pair is documented as ~0.3 in the case writeup.
PAIRS = [
    ("1ake_self",      "1ake", "1ake", 0.99),
    ("1cse_self",      "1cse", "1cse", 0.99),
    ("1ake_vs_1cse",   "1ake", "1cse", 0.20),
]


@pytest.fixture(params=PAIRS, ids=lambda p: p[0])
def pair(request):
    """Yields (name, pdb1_path, pdb2_path, tm_floor) per parametrize cell."""
    name, n1, n2, floor = request.param
    _skip_if_fixture_missing(n1, n2)
    return name, _pdb_path(n1), _pdb_path(n2), floor


def _skip_if_no_usalign() -> None:
    if _resolve_usalign() is None:
        pytest.skip("C++ USAlign binary not available (set USALIGN_BIN)")


class TestMMAlignVsCpp:
    """Backs the proteon-usalign-mm-vs-cpp-ci claim's tolerances."""

    def test_total_tm_score_matches(self, pair):
        _skip_if_no_usalign()
        name, p1, p2, floor = pair
        cpp = _run_cpp_usalign_mm(p1, p2)
        assert cpp is not None, f"USalign failed to produce output on {name}"
        assert cpp.tm_complex >= floor, (
            f"sanity floor breached: cpp.tm_complex={cpp.tm_complex} "
            f"on {name} but expected >= {floor}"
        )

        pdb1 = py_io.load(p1)
        pdb2 = py_io.load(p2)
        proteon_result = py_align_funcs.mm_align_pair(pdb1, pdb2)

        # Tolerance: 5e-4 absolute. Matches the single-chain claim's
        # ffast-math-floor on TM-score parity.
        assert abs(proteon_result.total_score - cpp.tm_complex) < 5e-4, (
            f"{name}: proteon.total_score={proteon_result.total_score} "
            f"vs cpp.tm_complex={cpp.tm_complex} differ beyond 5e-4"
        )

    def test_chain_assignments_match_as_set(self, pair):
        _skip_if_no_usalign()
        name, p1, p2, _floor = pair
        # Per-pair RMSD/n_aligned and chain-assignment-set assertions
        # need parsing the per-chain rows from -outfmt 2; that lands
        # in a follow-up commit once the parser is generalised. For
        # v0 the claim asserts on the whole-complex TM (above) and
        # documents the per-chain checks in the YAML — see
        # failure_modes for the gap.
        pdb1 = py_io.load(p1)
        pdb2 = py_io.load(p2)
        proteon_result = py_align_funcs.mm_align_pair(pdb1, pdb2)

        # On a self-alignment the assignment must be identity.
        if p1 == p2:
            n_chains = len(proteon_result.chain_assignments)
            assert proteon_result.chain_assignments == [
                (i, i) for i in range(n_chains)
            ], (
                f"{name} self-alignment: expected identity chain assignment "
                f"got {proteon_result.chain_assignments}"
            )

        # Cross pairs: just assert the assignment is a valid bijection
        # over a non-empty set of chains. Set-equality vs USAlign's
        # reported pairs lands in the per-chain follow-up.
        assert len(proteon_result.chain_assignments) > 0
        qs = {q for (q, _t) in proteon_result.chain_assignments}
        ts = {t for (_q, t) in proteon_result.chain_assignments}
        assert len(qs) == len(proteon_result.chain_assignments)
        assert len(ts) == len(proteon_result.chain_assignments)
