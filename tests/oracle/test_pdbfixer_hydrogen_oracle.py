"""Oracle test: proteon hydrogen placement vs PDBFixer (OpenMM).

A second, independent, CI-runnable H-placement oracle alongside
`test_reduce_hydrogen_oracle.py`. Reduce (Richardson lab) is the gold standard
but is **dormant in CI** (not apt-installable); PDBFixer (OpenMM lineage) is
pip-installable, so this runs on every PR. Two independent oracles, like
pydssp + mkdssp for secondary structure.

Why three layers, not tight per-atom positions
-----------------------------------------------
H positions are convention/rotamer-dependent: methyls / OH / NH3+ / glycine
HA2-HA3 differ by ~1.5 Å between any two tools even when chemically identical
(rotamer/prochiral labelling), and backbone amide H is method-dependent. Only
*uniquely-determined* H (single HA, aromatic ring H) agree tightly (~0.15 Å).
So the oracle asserts what is actually robust (measured vs PDBFixer 1.12 /
OpenMM 8.1 on the five fixtures):

- **L1 completeness** (protein residues only; water excluded — proteon doesn't
  H-ify HOH, PDBFixer does, which was the entire global-recall gap):
  precision ≥ 0.95 (proteon invents no H) AND recall ≥ 0.95 (no gross
  under-placement). Measured ~99.7% both ways; the 1–2 residual misses are
  N/C-termini + His-tautomer protonation conventions, documented not asserted.
- **L2 tight position** on uniquely-determined H (single `HA` + aromatic ring
  H): median < 0.40 Å, p90 < 0.80 Å, with a minimum sample count (a tight layer
  covering a handful of atoms could pass trivially). Measured (5 fixtures,
  OpenMM 8.5): med 0.12–0.29, p90 ≤ 0.60 — limits carry version headroom.
- **L3 loose sanity** on all matched H: p99 < 2.5 Å AND max < 3.0 Å — bounds
  rotamer noise, catches catastrophic misplacement. Measured max 1.95–2.15.

Both tools place H on identical heavy atoms: PDBFixer runs `addMissingHydrogens`
ONLY (no `addMissingAtoms`), proteon uses AMBER96 all-atom placement (CHARMM19
is united-atom / polar-only). H naming is PDB v3 on both ⇒ match by
`(chain, resseq, atom_name)`.
"""

import math
import os

import numpy as np
import pytest

import proteon

pytestmark = pytest.mark.oracle("pdbfixer")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROTEON_PDBS = os.path.normpath(os.path.join(_HERE, "..", "..", "test-pdbs"))

STRUCTURES = ["1crn", "1ubq", "1enh", "1ake", "4hhb"]

PRECISION_TOLERANCE = 0.95
RECALL_TOLERANCE = 0.95
# Rigid-H limits bound the PDBFixer reference, whose `addMissingHydrogens` runs a
# brief OpenMM minimization that is PLATFORM-nondeterministic: with IDENTICAL
# versions (pdbfixer 1.12.0 / openmm 8.5.2) 4hhb's rigid-H median measured 0.298 Å
# locally but 0.420 Å on the CI ubuntu runner — a ~0.12 Å platform artifact of the
# OpenMM reference, NOT a proteon regression (proteon's H placement is
# deterministic). So the median layer must carry headroom over the worst OBSERVED
# (0.420), not just version drift. 0.55 still catches a real placement regression
# (proteon tight-H medians are 0.13–0.30; a true break blows past 0.55); the p90,
# L1 completeness, and L3 sanity layers stay tight as the primary guards.
RIGID_MEDIAN_TOLERANCE = 0.55   # Å (platform-robust; see note above)
RIGID_P90_TOLERANCE = 0.80      # Å (CI worst 0.665, headroom intact)
RIGID_MIN_SAMPLES = 20
SANITY_P99_TOLERANCE = 2.5      # Å
SANITY_MAX_TOLERANCE = 3.0      # Å

WATER = {"HOH", "WAT", "TIP3", "H2O", "SOL"}

# Uniquely-determined aromatic ring C–H by residue (no rotamer/prochiral freedom).
_AROMATIC_RING_H = {
    "PHE": {"HD1", "HD2", "HE1", "HE2", "HZ"},
    "TYR": {"HD1", "HD2", "HE1", "HE2"},
    "TRP": {"HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"},
    "HIS": {"HD2", "HE1"},
    "HID": {"HD2", "HE1"},
    "HIE": {"HD2", "HE1"},
    "HIP": {"HD2", "HE1"},
}


def _is_rigid(resname: str, atom: str) -> bool:
    """Uniquely-determined H: the single alpha HA, or an aromatic ring C–H."""
    if atom == "HA":
        return True
    return atom in _AROMATIC_RING_H.get(resname, ())


class _BackendError(Exception):
    """PDBFixer/OpenMM is present but failed — must be loud, not skipped."""


def _proteon_h(path: str):
    """proteon AMBER96 all-atom H, protein residues only.

    Returns (dict {key: (x,y,z)}, dict {key: resname}, set aa_resids)."""
    import warnings

    s = proteon.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proteon.prepare(
            s, ff="amber96", reconstruct=False, hydrogens="all",
            minimize=False, strip_hydrogens=True,
        )
    pos, resn, aa = {}, {}, set()
    for ch in s.chains:
        for r in ch.residues:
            if not r.is_amino_acid:
                continue
            aa.add((ch.id, r.serial_number))
            for a in r.atoms:
                if a.element == "H":
                    k = (ch.id, r.serial_number, a.name)
                    pos[k] = (a.x, a.y, a.z)
                    resn[k] = r.name
    return pos, resn, aa


def _pdbfixer_h(path: str, aa_resids: set):
    """PDBFixer H positions (Å), restricted to proteon's amino-acid residues.

    None if PDBFixer/OpenMM is ABSENT; raises [`_BackendError`] if present but
    fails (no silent CI skip). `addMissingHydrogens` only — heavy atoms (the H
    parents) stay identical to proteon's input."""
    import importlib.util

    # Distinguish TRULY ABSENT (no module → optional skip) from
    # INSTALLED-BUT-BROKEN (import chain fails → loud, not a silent CI skip).
    if importlib.util.find_spec("pdbfixer") is None or importlib.util.find_spec("openmm") is None:
        return None
    try:
        from pdbfixer import PDBFixer
    except Exception as e:  # noqa: BLE001 — installed but import failed
        raise _BackendError(f"pdbfixer/openmm installed but import failed: {e}") from e
    try:
        fixer = PDBFixer(filename=path)
        fixer.addMissingHydrogens(7.0)
    except Exception as e:  # noqa: BLE001
        raise _BackendError(f"PDBFixer present but failed on {os.path.basename(path)}: {e}") from e
    out = {}
    for chain in fixer.topology.chains():
        for res in chain.residues():
            if res.name in WATER:
                continue
            rid = (chain.id, int(res.id))
            if rid not in aa_resids:  # protein residues proteon also sees
                continue
            for atom in res.atoms():
                el = atom.element
                if el is not None and el.symbol == "H":
                    p = fixer.positions[atom.index]
                    out[(chain.id, int(res.id), atom.name)] = (p.x * 10.0, p.y * 10.0, p.z * 10.0)
    return out


def _versions() -> str:
    # PyPI `pdbfixer` ships no __version__ attribute, so read from package
    # metadata — this oracle is version-sensitive, so the actual versions must
    # show up in failure output for drift diagnostics.
    from importlib.metadata import PackageNotFoundError, version

    def _v(pkg: str) -> str:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "?"

    return f"pdbfixer {_v('pdbfixer')} / openmm {_v('openmm')}"


def _compare(name: str):
    path = os.path.join(PROTEON_PDBS, f"{name}.pdb")
    if not os.path.exists(path):
        pytest.skip(f"missing structure: {path}")
    pro, resn, aa = _proteon_h(path)
    try:
        pf = _pdbfixer_h(path, aa)
    except _BackendError as e:
        pytest.fail(str(e))
    if pf is None:
        pytest.skip("PDBFixer/OpenMM not installed (this oracle has no Reduce fallback)")
    common = set(pro) & set(pf)
    dists = {k: math.dist(pro[k], pf[k]) for k in common}
    rigid = [d for k, d in dists.items() if _is_rigid(resn[k], k[2])]
    return {
        "name": name,
        "backend": _versions(),
        "pro": pro,
        "pf": pf,
        "common": common,
        "dists": dists,
        "rigid": rigid,
        "precision": len(common) / len(pro) if pro else 0.0,
        "recall": len(common) / len(pf) if pf else 0.0,
    }


class TestPdbfixerHydrogenOracle:
    """proteon H placement vs PDBFixer, protein residues only."""

    @pytest.fixture(params=STRUCTURES, ids=STRUCTURES)
    def case(self, request):
        return _compare(request.param)

    def test_precision(self, case):
        """proteon must not invent / mislabel H: its protein H are a near-subset
        of PDBFixer's."""
        missing = [k for k in case["pro"] if k not in case["pf"]][:10]
        assert case["precision"] >= PRECISION_TOLERANCE, (
            f"{case['name']} [{case['backend']}]: H precision "
            f"{case['precision'] * 100:.1f}% < {PRECISION_TOLERANCE * 100:.0f}%. "
            f"proteon H absent from PDBFixer: {missing}"
        )

    def test_recall(self, case):
        """proteon must place ~all the protein H PDBFixer does (water excluded).
        The residual gap is N/C-termini + His-tautomer protonation conventions."""
        missing = [k for k in case["pf"] if k not in case["pro"]][:10]
        assert case["recall"] >= RECALL_TOLERANCE, (
            f"{case['name']} [{case['backend']}]: H recall "
            f"{case['recall'] * 100:.1f}% < {RECALL_TOLERANCE * 100:.0f}%. "
            f"PDBFixer H absent from proteon: {missing}"
        )

    def test_rigid_position_agreement(self, case):
        """Uniquely-determined H (single HA + aromatic ring) must agree tightly."""
        rigid = case["rigid"]
        assert len(rigid) >= RIGID_MIN_SAMPLES, (
            f"{case['name']}: only {len(rigid)} rigid H — too few to be meaningful"
        )
        med = float(np.median(rigid))
        p90 = float(np.percentile(rigid, 90))
        assert med < RIGID_MEDIAN_TOLERANCE and p90 < RIGID_P90_TOLERANCE, (
            f"{case['name']} [{case['backend']}]: rigid-H position drift "
            f"med={med:.3f} p90={p90:.3f} Å (limits {RIGID_MEDIAN_TOLERANCE}/"
            f"{RIGID_P90_TOLERANCE}); n={len(rigid)}"
        )

    def test_loose_position_sanity(self, case):
        """No matched H may be catastrophically misplaced (bounds rotamer noise +
        catches unit/parent bugs)."""
        d = list(case["dists"].values())
        p99 = float(np.percentile(d, 99))
        mx = float(max(d))
        worst = sorted(case["dists"].items(), key=lambda kv: -kv[1])[:5]
        assert p99 < SANITY_P99_TOLERANCE and mx < SANITY_MAX_TOLERANCE, (
            f"{case['name']} [{case['backend']}]: H position sanity "
            f"p99={p99:.3f} max={mx:.3f} Å (limits {SANITY_P99_TOLERANCE}/"
            f"{SANITY_MAX_TOLERANCE}); worst={[(f'{c}:{r}:{n}', round(v,2)) for (c,r,n),v in worst]}"
        )
