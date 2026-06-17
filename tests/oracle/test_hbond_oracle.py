"""Oracle test: proteon backbone H-bonds vs canonical DSSP (mkdssp).

`proteon.backbone_hbonds` detects backbone hydrogen bonds with the **same
Kabsch–Sander electrostatic criterion DSSP uses** (interaction energy < −0.5
kcal/mol between backbone C=O and N–H). DSSP *is* a backbone-H-bond detector, so
canonical mkdssp is the natural oracle. `tests/test_hbond.py` has unit tests but
no external oracle; this adds one, reusing the DSSP-oracle's mkdssp machinery
(#169/#170).

mkdssp-only
-----------
Unlike the secondary-structure oracle, there is no `gmx dssp` fallback: gmx emits
SS but not per-residue H-bond energies. So this test runs only where mkdssp is
installed (CI via `apt-get install dssp`) and skips locally. A mkdssp that is
present but *broken* fails loudly rather than skipping (it can't quietly green CI).

Comparison: unordered residue-ID pairs
--------------------------------------
proteon's `[res_a, res_b]` columns are NOT a stable donor/acceptor convention
relative to DSSP's NH→O direction (measured on 1crn: 22 of 26 common bonds are
opposite-order, 4 same-order; proteon also lists a few bonds reciprocally). An
H-bond between residues i and j is the same physical bond regardless of column,
so we compare **unordered residue-ID pairs** — robust to both the column
convention and any residue-ordering/filtering difference (codex: align by ID,
not positional index). Measured vs mkdssp 4.2.2: 93.6–100% precision, 91.1–100%
recall, aggregate 94.8% / 97.6%; matched-pair energy median ≤ 0.42, p90 ≤ 0.70
kcal/mol.

mkdssp 4.x gotcha: treats a file not starting with `HEADER` as mmCIF and fails;
we prepend a synthetic HEADER to a temp copy.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

import proteon

pytestmark = pytest.mark.oracle("hbond")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROTEON_PDBS = os.path.normpath(os.path.join(_HERE, "..", "..", "test-pdbs"))

STRUCTURES = ["1crn", "1ubq", "1enh", "1ake", "4hhb"]
ENERGY_CUTOFF = -0.5

PRECISION_TOLERANCE = 0.90
RECALL_TOLERANCE = 0.88
COUNT_PARITY_TOLERANCE = 0.08
ENERGY_MEDIAN_TOLERANCE = 0.5  # kcal/mol
ENERGY_P90_TOLERANCE = 1.0


class _BackendError(Exception):
    """mkdssp is present but failed — must be loud, not skipped (codex)."""


def _resid(chain, resseq, icode) -> str:
    return f"{chain}:{resseq}:{(icode or ' ').strip() or ' '}"


def _find_mkdssp() -> "str | None":
    if os.environ.get("PROTEON_MKDSSP"):
        return os.environ["PROTEON_MKDSSP"]
    for n in ("mkdssp", "dssp"):
        p = shutil.which(n)
        if p:
            return p
    return None


def _dssp_hbonds(path: str) -> "dict | None":
    """DSSP backbone H-bonds as ``{frozenset({id_a, id_b}): energy}``.

    None when mkdssp / Biopython is ABSENT; raises [`_BackendError`] when mkdssp
    is present but fails. Built from each residue's NH→O partners (donor's N–H to
    the acceptor at ``row + relidx`` in Biopython's DSSP-row order) with energy
    below the cutoff.
    """
    mkdssp = _find_mkdssp()
    if mkdssp is None:
        return None
    try:
        from Bio.PDB import DSSP, PDBParser
    except ImportError:
        return None

    with open(path) as fh:
        text = fh.read()
    cleanup = None
    use = path
    if not text.startswith("HEADER"):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
        tmp.write("HEADER    GENERATED FOR DSSP                      01-JAN-00   XXXX\n")
        tmp.write(text)
        tmp.close()
        use = tmp.name
        cleanup = tmp.name
    try:
        structure = PDBParser(QUIET=True).get_structure("x", use)
        d = DSSP(structure[0], use, dssp=mkdssp)
    except Exception as e:  # noqa: BLE001
        raise _BackendError(f"mkdssp ({mkdssp}) failed on {os.path.basename(path)}: {e}") from e
    finally:
        if cleanup:
            os.unlink(cleanup)

    keys = list(d.keys())  # Biopython DSSP-row order
    rids = [_resid(k[0], k[1][1], k[1][2]) for k in keys]
    rows = [d[k] for k in keys]
    bonds: "dict" = {}
    # Per-residue tuple: ... 6 NH_O_1_relidx, 7 NH_O_1_energy, 10 NH_O_2_relidx,
    # 11 NH_O_2_energy. NH→O: this residue donates N-H to the acceptor's C=O.
    for p, r in enumerate(rows):
        for ridx, eidx in ((6, 7), (10, 11)):
            relidx, energy = r[ridx], r[eidx]
            if energy < ENERGY_CUTOFF and relidx != 0:
                acc = p + relidx
                if 0 <= acc < len(rows):
                    key = frozenset((rids[p], rids[acc]))
                    # reciprocal/duplicate listings: keep the strongest (min) energy
                    bonds[key] = min(energy, bonds.get(key, energy))
    return bonds


def _proteon_hbonds(path: str) -> "dict":
    """proteon backbone H-bonds as ``{frozenset({id_a, id_b}): energy}``."""
    s = proteon.load(path)
    rids = [
        _resid(ch.id, r.serial_number, r.insertion_code)
        for ch in s.chains
        for r in ch.residues
        if r.is_amino_acid
    ]
    out: "dict" = {}
    for row in proteon.backbone_hbonds(s, energy_cutoff=ENERGY_CUTOFF):
        a, b, energy = int(row[0]), int(row[1]), float(row[2])
        if 0 <= a < len(rids) and 0 <= b < len(rids) and a != b:
            key = frozenset((rids[a], rids[b]))
            out[key] = min(energy, out.get(key, energy))
    return out


def _compare(name: str):
    """Return a dict of comparison metrics for one structure, skipping cleanly
    if no mkdssp backend is available."""
    path = os.path.join(PROTEON_PDBS, f"{name}.pdb")
    if not os.path.exists(path):
        pytest.skip(f"missing structure: {path}")
    try:
        dssp = _dssp_hbonds(path)
    except _BackendError as e:
        pytest.fail(str(e))  # present-but-broken, no fallback ⇒ loud
    if dssp is None:
        pytest.skip("mkdssp not available (this oracle has no gmx fallback)")
    pro = _proteon_hbonds(path)
    common = set(pro) & set(dssp)
    de = [abs(pro[k] - dssp[k]) for k in common]
    return {
        "name": name,
        "pro": pro,
        "dssp": dssp,
        "common": common,
        "precision": len(common) / len(pro) if pro else 0.0,
        "recall": len(common) / len(dssp) if dssp else 0.0,
        "count_parity": abs(len(pro) - len(dssp)) / len(dssp) if dssp else 1.0,
        "energy_median": float(np.median(de)) if de else 0.0,
        "energy_p90": float(np.percentile(de, 90)) if de else 0.0,
    }


def _symdiff(m, limit=8):
    pro_only = [sorted(k) for k in (set(m["pro"]) - set(m["dssp"]))][:limit]
    dssp_only = [sorted(k) for k in (set(m["dssp"]) - set(m["pro"]))][:limit]
    return f"proteon-only={pro_only} dssp-only={dssp_only}"


class TestHbondOracle:
    """proteon backbone H-bonds vs canonical mkdssp, per structure."""

    @pytest.fixture(params=STRUCTURES, ids=STRUCTURES)
    def case(self, request):
        return _compare(request.param)

    def test_precision(self, case):
        assert case["precision"] >= PRECISION_TOLERANCE, (
            f"{case['name']}: H-bond precision {case['precision'] * 100:.1f}% < "
            f"{PRECISION_TOLERANCE * 100:.0f}%. {_symdiff(case)}"
        )

    def test_recall(self, case):
        assert case["recall"] >= RECALL_TOLERANCE, (
            f"{case['name']}: H-bond recall {case['recall'] * 100:.1f}% < "
            f"{RECALL_TOLERANCE * 100:.0f}%. {_symdiff(case)}"
        )

    def test_count_parity(self, case):
        assert case["count_parity"] <= COUNT_PARITY_TOLERANCE, (
            f"{case['name']}: H-bond count drift {case['count_parity'] * 100:.1f}% > "
            f"{COUNT_PARITY_TOLERANCE * 100:.0f}% "
            f"(proteon={len(case['pro'])}, dssp={len(case['dssp'])})"
        )

    def test_energy_agreement(self, case):
        """Matched-pair Kabsch–Sander energies must agree (both implement the
        same formula). Median AND p90 — median alone hides bad tails (codex)."""
        assert case["energy_median"] < ENERGY_MEDIAN_TOLERANCE, (
            f"{case['name']}: median |ΔE| {case['energy_median']:.3f} ≥ "
            f"{ENERGY_MEDIAN_TOLERANCE} kcal/mol"
        )
        assert case["energy_p90"] < ENERGY_P90_TOLERANCE, (
            f"{case['name']}: p90 |ΔE| {case['energy_p90']:.3f} ≥ "
            f"{ENERGY_P90_TOLERANCE} kcal/mol"
        )


def test_aggregate_precision_recall():
    """A broad mild degradation across all structures should not pass when every
    one sits just above its per-structure floor (codex)."""
    metrics = [_compare(n) for n in STRUCTURES]
    tot_common = sum(len(m["common"]) for m in metrics)
    tot_pro = sum(len(m["pro"]) for m in metrics)
    tot_dssp = sum(len(m["dssp"]) for m in metrics)
    agg_prec = tot_common / tot_pro
    agg_recall = tot_common / tot_dssp
    assert agg_prec >= 0.90, f"aggregate precision {agg_prec * 100:.1f}% < 90%"
    assert agg_recall >= 0.90, f"aggregate recall {agg_recall * 100:.1f}% < 90%"
