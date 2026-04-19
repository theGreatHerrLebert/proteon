"""Oracle test: proteon DSSP secondary structure vs pydssp.

Proteon ships its own port of the Kabsch-Sander 1983 DSSP algorithm.
This test pins its per-residue H/E assignments against `pydssp`, an
independent NumPy/PyTorch reimplementation of the same algorithm by a
different research group (Shintaro Minami, AIST —
https://github.com/ShintaroMinami/PyDSSP).

Two independent implementations of the same paper are the right oracle
shape: a divergence is either a proteon bug or a convention gap, and
the cause is almost always readable from the failure diff.

Convention gap (why we compare in 3-class, not 8-class)
-------------------------------------------------------
pydssp emits a 3-class alphabet: H (helix) / E (strand) / - (loop).
Proteon emits the full 8-class DSSP alphabet (H, G, I, E, B, T, S, C).
We collapse proteon's output before comparing:

    H, G, I  ->  H      (all helix flavors)
    E, B     ->  E      (extended strand + isolated beta bridge)
    T, S, C  ->  -      (turn + bend + coil)

The collapse is lossy on helix flavor (alpha vs 3_10 vs pi) and on
isolated-vs-ladder beta, but it exactly spans the ground truth pydssp
can produce. Any surviving disagreement is therefore a real difference
in H-bond detection or boundary extension, not an alphabet artifact.

Tolerance
---------
95% per-structure agreement. Empirically the two implementations agree
97-100% on the canonical test set; the residual gap sits on helix/strand
boundary residues where the H-bond energy crosses the -0.5 kcal/mol
threshold and tiny coordinate differences flip the call. 95% buys
comfortable room for that boundary wobble without letting a systematic
regression slip through.
"""

import os

import numpy as np
import pytest

import proteon

pydssp = pytest.importorskip("pydssp")

pytestmark = pytest.mark.oracle("pydssp")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROTEON_PDBS = os.path.normpath(os.path.join(_HERE, "..", "..", "test-pdbs"))
SHARED_PDBS = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "test-pdbs"))


def _pdb(name: str, base: str) -> str:
    return os.path.join(base, f"{name}.pdb")


# (label, path) — mix of single-chain and multi-chain, helix/sheet/mixed folds.
STRUCTURES = [
    ("1crn", _pdb("1crn", PROTEON_PDBS)),       # 46 res,  mixed
    ("1ubq", _pdb("1ubq", PROTEON_PDBS)),       # 76 res,  mixed
    ("1enh", _pdb("1enh", SHARED_PDBS)),         # 54 res,  3-helix bundle
    ("1ake", _pdb("1ake", SHARED_PDBS)),         # 428 res, alpha/beta
    ("4hhb", _pdb("4hhb", SHARED_PDBS)),         # 574 res, 4-chain hemoglobin
]

AGREEMENT_TOLERANCE = 0.95


def _collapse_8class_to_3class(ss8: str) -> np.ndarray:
    """Collapse proteon's H/G/I/E/B/T/S/C to pydssp's H/E/-."""
    out = np.empty(len(ss8), dtype="<U1")
    for i, c in enumerate(ss8):
        if c in "HGI":
            out[i] = "H"
        elif c in "EB":
            out[i] = "E"
        else:
            out[i] = "-"
    return out


def _pydssp_3class(pdb_path: str) -> np.ndarray:
    with open(pdb_path) as f:
        pdb_text = f.read()
    coord = pydssp.read_pdbtext(pdb_text)
    return np.asarray(pydssp.assign(coord, out_type="c3"))


def _diff_positions(a: np.ndarray, b: np.ndarray, ss8: str, limit: int = 10):
    """Return first `limit` disagreement sites as (index, proteon-8class, proteon-3class, pydssp-3class)."""
    idx = np.where(a != b)[0]
    return [(int(i), ss8[i], str(a[i]), str(b[i])) for i in idx[:limit]]


class TestDsspOracle:
    """Compare proteon DSSP against pydssp per-residue."""

    @pytest.fixture(params=STRUCTURES, ids=[s[0] for s in STRUCTURES])
    def case(self, request):
        name, path = request.param
        if not os.path.exists(path):
            pytest.skip(f"test structure missing: {path}")
        s = proteon.load(path)
        ss_proteon_8 = proteon.dssp(s)
        ss_pydssp_3 = _pydssp_3class(path)
        return name, ss_proteon_8, ss_pydssp_3

    def test_length_matches(self, case):
        """Both tools must count the same residues. A length mismatch means
        one filtered differently (e.g. HETATM handling) — that's a fixture
        problem, not a scoring tolerance, so we fail hard."""
        name, ss_f8, ss_p3 = case
        assert len(ss_f8) == len(ss_p3), (
            f"{name}: residue count diverges — proteon={len(ss_f8)} "
            f"pydssp={len(ss_p3)}. Check HETATM / chain filtering."
        )

    def test_3class_agreement(self, case):
        name, ss_f8, ss_p3 = case
        ss_f3 = _collapse_8class_to_3class(ss_f8)
        agree = float((ss_f3 == ss_p3).mean())
        diffs = _diff_positions(ss_f3, ss_p3, ss_f8, limit=10)
        assert agree >= AGREEMENT_TOLERANCE, (
            f"{name}: 3-class agreement {agree * 100:.2f}% < "
            f"{AGREEMENT_TOLERANCE * 100:.0f}%. "
            f"First disagreements (idx, proteon-8, proteon-3, pydssp-3): {diffs}"
        )

    def test_helix_fraction_parity(self, case):
        """Gross helix fraction parity — catches systematic H-bond drift
        that 3-class agreement might mask on structures with very few
        helices."""
        name, ss_f8, ss_p3 = case
        ss_f3 = _collapse_8class_to_3class(ss_f8)
        h_f = float((ss_f3 == "H").mean())
        h_p = float((ss_p3 == "H").mean())
        assert abs(h_f - h_p) < 0.10, (
            f"{name}: helix fraction diverges — "
            f"proteon={h_f * 100:.1f}% pydssp={h_p * 100:.1f}%"
        )

    def test_strand_fraction_parity(self, case):
        name, ss_f8, ss_p3 = case
        ss_f3 = _collapse_8class_to_3class(ss_f8)
        e_f = float((ss_f3 == "E").mean())
        e_p = float((ss_p3 == "E").mean())
        assert abs(e_f - e_p) < 0.10, (
            f"{name}: strand fraction diverges — "
            f"proteon={e_f * 100:.1f}% pydssp={e_p * 100:.1f}%"
        )
