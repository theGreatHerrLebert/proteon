"""Oracle test: proteon DSSP 8-class secondary structure vs *canonical* DSSP.

The sibling `test_dssp_oracle.py` pins proteon against pydssp, but pydssp only
emits 3 classes (H/E/loop). proteon emits the full 8-class DSSP alphabet
(H, G, I, E, B, T, S, C), so the helix-flavor (G/I), beta-bridge (B), and
turn/bend (T/S) classes are never validated there.

This test compares proteon's full 8-class assignment against the *canonical*
Kabsch–Sander DSSP, from one of two interchangeable backends:

- **mkdssp** (the reference DSSP binary, DSSP 4.x), via Biopython's
  `Bio.PDB.DSSP`. Keyed by residue id ``(chain, resseq, icode)`` ⇒ robust
  id-based alignment, so HETATM / chain filtering differences never misalign
  the comparison. This is the CI backend (`apt-get install dssp`).
- **`gmx dssp`** (GROMACS' DSSP), positional. A local convenience backend for
  developers who have GROMACS but not mkdssp; used only when its residue count
  matches proteon's (single-chain structures).

Both are canonical implementations of the same paper — two independent DSSPs is
the right oracle shape, exactly like the pydssp test.

Normalisation
-------------
proteon does not model the polyproline-II (`P`) class that DSSP 4.x emits, so we
fold mkdssp's `P` into loop, alongside the loop sentinels (`-`/`~`/space) → `C`.
The comparison alphabet is then exactly proteon's H/G/I/E/B/T/S/C.

Thresholds (measured against mkdssp 4.2.2 on the five structures below):
8-class 93.0–100%, 3-class 93.5–100%, helix↔strand confusion 0 everywhere.
We require 8-class ≥ 0.90, 3-class ≥ 0.92, and *zero* helix↔strand swaps — the
residual disagreements are within-category boundary calls (alpha vs pi helix,
turn vs bend vs 3-10 helix), not gross errors.

mkdssp 4.x gotcha: it treats any file not starting with `HEADER` as mmCIF and
fails; we prepend a synthetic HEADER to a temp copy before handing it over.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import proteon

pytestmark = pytest.mark.oracle("dssp8")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROTEON_PDBS = os.path.normpath(os.path.join(_HERE, "..", "..", "test-pdbs"))
SHARED_PDBS = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "test-pdbs"))

# (label, path) — single-chain (gmx- and mkdssp-comparable) + multi-chain
# (mkdssp id-aligned only).
STRUCTURES = [
    ("1crn", os.path.join(PROTEON_PDBS, "1crn.pdb")),
    ("1ubq", os.path.join(PROTEON_PDBS, "1ubq.pdb")),
    ("1enh", os.path.join(SHARED_PDBS, "1enh.pdb")),
    ("1ake", os.path.join(SHARED_PDBS, "1ake.pdb")),
    ("4hhb", os.path.join(SHARED_PDBS, "4hhb.pdb")),
]

EIGHT_CLASS_TOLERANCE = 0.90
THREE_CLASS_TOLERANCE = 0.92
COVERAGE_TOLERANCE = 0.98


def _norm(c: str) -> str:
    """Canonicalise a DSSP code to proteon's H/G/I/E/B/T/S/C alphabet.

    Loop sentinels (mkdssp `-`, gmx `~`, blank) and the polyproline-II `P`
    class proteon does not model all fold to coil `C`.
    """
    return "C" if c in "-~P C" else c


def _group(c: str) -> str:
    """3-class grouping: helix / strand / loop."""
    if c in "HGI":
        return "H"
    if c in "EB":
        return "E"
    return "L"


def _proteon_id_ss(path: str) -> "dict[tuple, str]":
    """proteon's per-residue 8-class SS, keyed by ``(chain, resseq, icode)``.

    `proteon.dssp` covers amino-acid residues in chain order, so we zip it with
    the amino-acid residues in the same order to recover residue identity.
    """
    s = proteon.load(path)
    ss = proteon.dssp(s)
    ids = [
        (ch.id, r.serial_number, (r.insertion_code or " ").strip() or " ")
        for ch in s.chains
        for r in ch.residues
        if r.is_amino_acid
    ]
    assert len(ids) == len(ss), (
        f"{path}: proteon AA-residue count {len(ids)} != dssp length {len(ss)}"
    )
    return {k: _norm(c) for k, c in zip(ids, ss)}


def _find_binary(env: str, *names: str) -> "str | None":
    if os.environ.get(env):
        return os.environ[env]
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


class _BackendError(Exception):
    """A canonical-DSSP backend is PRESENT but failed to run — distinct from
    a backend simply being absent. The caller must not silently skip on this
    (it would hide a broken oracle in CI); it falls back or fails."""


def _mkdssp_reference(path: str) -> "dict[tuple, str] | None":
    """Canonical 8-class SS from mkdssp, keyed by residue id.

    Returns None only when the mkdssp binary or Biopython is ABSENT (the oracle
    is then optional). Raises [`_BackendError`] when mkdssp is present but fails
    — a present-but-broken canonical backend must be loud, not skipped (codex).
    """
    mkdssp = _find_binary("PROTEON_MKDSSP", "mkdssp", "dssp")
    if mkdssp is None:
        return None
    try:
        from Bio.PDB import DSSP, PDBParser
    except ImportError:
        return None

    # mkdssp 4.x assumes mmCIF unless the file starts with HEADER; ensure one.
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
        raise _BackendError(
            f"mkdssp ({mkdssp}) present but failed on {os.path.basename(path)}: {e}"
        ) from e
    finally:
        if cleanup:
            os.unlink(cleanup)
    out = {}
    for chain, (_, resseq, icode) in d.keys():
        ss = d[(chain, (" ", resseq, icode))][2]
        out[(chain, resseq, (icode or " ").strip() or " ")] = _norm(ss)
    return out


def _gmx_reference(path: str, proteon_ss: "dict[tuple, str]") -> "dict[tuple, str] | None":
    """Canonical 8-class SS from `gmx dssp` (positional). None if unavailable
    or if its residue count does not match proteon's (multi-chain)."""
    gmx = _find_binary(
        "PROTEON_GMX",
        "gmx",
        os.path.join(SHARED_PDBS, "..", "gromacs-2026.1", "build", "bin", "gmx"),
    )
    if gmx is None or not (os.path.isabs(gmx) and os.path.exists(gmx) or shutil.which(gmx)):
        return None
    out_dat = tempfile.NamedTemporaryFile(suffix=".dat", delete=False).name
    try:
        subprocess.run(
            [gmx, "dssp", "-s", path, "-o", out_dat, "-hmode", "dssp", "-polypro", "no"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [ln for ln in open(out_dat).read().splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return None
    finally:
        if os.path.exists(out_dat):
            os.unlink(out_dat)
    if not lines:
        return None
    ref_str = lines[0]
    # Positional: only usable when the residue counts line up (single chain).
    keys = list(proteon_ss.keys())
    if len(ref_str) != len(keys):
        return None
    return {k: _norm(c) for k, c in zip(keys, ref_str)}


def _reference(path: str, proteon_ss: "dict[tuple, str]") -> "tuple[dict, str]":
    """Return (reference dict, backend label).

    Skip ONLY when no backend is installed at all. If mkdssp is present but
    broken, fall back to gmx if available, else FAIL — a configured canonical
    backend that cannot run must not be silently skipped (codex)."""
    try:
        ref = _mkdssp_reference(path)
    except _BackendError as e:
        gmx_ref = _gmx_reference(path, proteon_ss)
        if gmx_ref is not None:
            return gmx_ref, "gmx (mkdssp present but broken)"
        pytest.fail(str(e))  # present-but-broken + no fallback ⇒ loud, not skipped
    if ref is not None:
        return ref, "mkdssp"
    ref = _gmx_reference(path, proteon_ss)
    if ref is not None:
        return ref, "gmx"
    pytest.skip("no canonical DSSP backend available (need mkdssp or gmx)")


class TestDssp8ClassOracle:
    """Compare proteon's full 8-class DSSP against canonical mkdssp / gmx dssp."""

    @pytest.fixture(params=STRUCTURES, ids=[s[0] for s in STRUCTURES])
    def case(self, request):
        name, path = request.param
        if not os.path.exists(path):
            pytest.skip(f"test structure missing: {path}")
        proteon_ss = _proteon_id_ss(path)
        ref, backend = _reference(path, proteon_ss)
        matched = [k for k in proteon_ss if k in ref]
        return name, backend, proteon_ss, ref, matched

    def test_coverage(self, case):
        """Almost every proteon residue must be found in the reference; a low
        match rate means a residue-set divergence (filtering / id mapping), the
        very bug this oracle should catch — not a scoring wobble."""
        name, backend, proteon_ss, ref, matched = case
        coverage = len(matched) / len(proteon_ss)
        missing = [k for k in proteon_ss if k not in ref][:10]
        assert coverage >= COVERAGE_TOLERANCE, (
            f"{name} [{backend}]: coverage {coverage * 100:.1f}% < "
            f"{COVERAGE_TOLERANCE * 100:.0f}%. proteon residues absent from "
            f"reference (first 10): {missing}"
        )

    def test_8class_agreement(self, case):
        name, backend, proteon_ss, ref, matched = case
        import collections

        agree = float(np.mean([proteon_ss[k] == ref[k] for k in matched]))
        diffs = collections.Counter(
            (proteon_ss[k], ref[k]) for k in matched if proteon_ss[k] != ref[k]
        )
        assert agree >= EIGHT_CLASS_TOLERANCE, (
            f"{name} [{backend}]: 8-class agreement {agree * 100:.2f}% < "
            f"{EIGHT_CLASS_TOLERANCE * 100:.0f}%. "
            f"Confusions (proteon, ref): {dict(diffs.most_common(8))}"
        )

    def test_3class_agreement(self, case):
        name, backend, proteon_ss, ref, matched = case
        agree = float(np.mean([_group(proteon_ss[k]) == _group(ref[k]) for k in matched]))
        assert agree >= THREE_CLASS_TOLERANCE, (
            f"{name} [{backend}]: 3-class agreement {agree * 100:.2f}% < "
            f"{THREE_CLASS_TOLERANCE * 100:.0f}%"
        )

    def test_no_helix_strand_confusion(self, case):
        """A helix residue called strand (or vice versa) by canonical DSSP is a
        gross error, not a boundary wobble — there must be none."""
        name, backend, proteon_ss, ref, matched = case
        swaps = [
            (k, proteon_ss[k], ref[k])
            for k in matched
            if {_group(proteon_ss[k]), _group(ref[k])} == {"H", "E"}
        ]
        assert not swaps, (
            f"{name} [{backend}]: {len(swaps)} helix↔strand disagreements "
            f"(residue, proteon, ref): {swaps[:10]}"
        )
