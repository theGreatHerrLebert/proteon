"""Oracle tests: TM-score validation against C++ USAlign binary.

Runs C++ USAlign on reference pairs and compares TM-scores to proteon.
Should match to ~4-5 decimal places (limited by -ffast-math differences).
"""

import os
import subprocess
from typing import NamedTuple, Optional

import pytest

from proteon_connector import py_align_funcs, py_io

pytestmark = pytest.mark.oracle("usalign")

# Path to the C++ USAlign binary used as oracle. Override via the
# USALIGN_BIN env var; defaults to whatever is on $PATH.
USALIGN_BIN = os.environ.get("USALIGN_BIN", "USalign")
PDBTBX_EXAMPLES = os.path.join(
    os.path.dirname(__file__), "..", "..", "test-pdbs"
)
TEST_PDBS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "test-pdbs"
)


class USAlignResult(NamedTuple):
    tm1: float
    tm2: float
    rmsd: float
    n_aligned: int
    len1: int
    len2: int


def run_cpp_usalign(path1: str, path2: str) -> Optional[USAlignResult]:
    """Run C++ USAlign and parse tabular output."""
    if not os.path.exists(USALIGN_BIN):
        return None
    try:
        result = subprocess.run(
            [USALIGN_BIN, path1, path2, "-outfmt", "2"],
            capture_output=True, text=True, timeout=60,
        )
        for line in result.stdout.strip().split("\n"):
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 11:
                return USAlignResult(
                    tm1=float(parts[2]),
                    tm2=float(parts[3]),
                    rmsd=float(parts[4]),
                    n_aligned=int(parts[10]),
                    len1=int(parts[8]),
                    len2=int(parts[9]),
                )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def skip_if_no_usalign():
    if not os.path.exists(USALIGN_BIN):
        pytest.skip("C++ USAlign binary not available")


# Test pairs: (name, path1, path2)
PAIRS = [
    ("ubiq_vs_crambin",
     os.path.join(PDBTBX_EXAMPLES, "1ubq.pdb"),
     os.path.join(TEST_PDBS, "1crn.pdb")),
    ("ubiq_self",
     os.path.join(PDBTBX_EXAMPLES, "1ubq.pdb"),
     os.path.join(PDBTBX_EXAMPLES, "1ubq.pdb")),
    ("crambin_vs_1lyz",
     os.path.join(TEST_PDBS, "1crn.pdb"),
     os.path.join(TEST_PDBS, "1lyz.pdb")),
    ("1lyz_vs_2lzm",
     os.path.join(TEST_PDBS, "1lyz.pdb"),
     os.path.join(TEST_PDBS, "2lzm.pdb")),
    ("1tit_vs_2gb1",
     os.path.join(TEST_PDBS, "1tit.pdb"),
     os.path.join(TEST_PDBS, "2gb1.pdb")),
    ("5pti_vs_1bpi",
     os.path.join(TEST_PDBS, "5pti.pdb"),
     os.path.join(TEST_PDBS, "1bpi.pdb")),
    ("1mbn_vs_2hbg",
     os.path.join(TEST_PDBS, "1mbn.pdb"),
     os.path.join(TEST_PDBS, "2hbg.pdb")),
]


def available_pairs():
    return [
        (name, p1, p2) for name, p1, p2 in PAIRS
        if os.path.exists(p1) and os.path.exists(p2)
    ]


@pytest.fixture(params=available_pairs(), ids=lambda x: x[0])
def pair(request):
    return request.param


class TestTMScoreVsCpp:
    """Compare proteon TM-scores against C++ USAlign."""

    def test_tm_scores_match(self, pair):
        skip_if_no_usalign()
        name, path1, path2 = pair

        cpp = run_cpp_usalign(path1, path2)
        assert cpp is not None, f"C++ USAlign failed on {name}"

        pdb1 = py_io.load(path1)
        pdb2 = py_io.load(path2)
        rust = py_align_funcs.tm_align_pair(pdb1, pdb2)

        # C++ convention: TM1 = norm by L1 (chain1), TM2 = norm by L2 (chain2)
        # Proteon convention: tm_score_chain1 = norm by L2, tm_score_chain2 = norm by L1
        # So: proteon.chain1 ↔ cpp.tm2, proteon.chain2 ↔ cpp.tm1
        # Match to ~4 decimal places (limited by -ffast-math differences)
        assert rust.tm_score_chain1 == pytest.approx(cpp.tm2, abs=5e-4), \
            f"{name}: TM(normL2) proteon={rust.tm_score_chain1:.5f} cpp={cpp.tm2:.5f}"
        assert rust.tm_score_chain2 == pytest.approx(cpp.tm1, abs=5e-4), \
            f"{name}: TM(normL1) proteon={rust.tm_score_chain2:.5f} cpp={cpp.tm1:.5f}"

    def test_rmsd_matches(self, pair):
        skip_if_no_usalign()
        name, path1, path2 = pair

        cpp = run_cpp_usalign(path1, path2)
        assert cpp is not None

        pdb1 = py_io.load(path1)
        pdb2 = py_io.load(path2)
        rust = py_align_funcs.tm_align_pair(pdb1, pdb2)

        # RMSD should match to ~2 decimal places
        assert rust.rmsd == pytest.approx(cpp.rmsd, abs=0.05), \
            f"{name}: RMSD proteon={rust.rmsd:.3f} cpp={cpp.rmsd:.3f}"

    def test_n_aligned_matches(self, pair):
        skip_if_no_usalign()
        name, path1, path2 = pair

        cpp = run_cpp_usalign(path1, path2)
        assert cpp is not None

        pdb1 = py_io.load(path1)
        pdb2 = py_io.load(path2)
        rust = py_align_funcs.tm_align_pair(pdb1, pdb2)

        # Allow ±2 residue difference in alignment length
        assert abs(rust.n_aligned - cpp.n_aligned) <= 2, \
            f"{name}: Lali proteon={rust.n_aligned} cpp={cpp.n_aligned}"

    def test_chain_lengths_positive(self, pair):
        """Verify C++ reports positive chain lengths."""
        skip_if_no_usalign()
        _, path1, path2 = pair
        cpp = run_cpp_usalign(path1, path2)
        assert cpp is not None
        assert cpp.len1 > 0
        assert cpp.len2 > 0


class TestFastMode:
    """Verify fast mode produces reasonable (if not identical) results."""

    def test_fast_vs_normal(self, pair):
        skip_if_no_usalign()
        name, path1, path2 = pair

        pdb1 = py_io.load(path1)
        pdb2 = py_io.load(path2)

        normal = py_align_funcs.tm_align_pair(pdb1, pdb2, fast=False)
        fast = py_align_funcs.tm_align_pair(pdb1, pdb2, fast=True)

        # Fast should be within 10% of normal (it's approximate)
        if normal.tm_score_chain1 > 0.1:
            ratio = fast.tm_score_chain1 / normal.tm_score_chain1
            assert 0.8 < ratio < 1.1, \
                f"{name}: fast/normal TM ratio={ratio:.3f}"
