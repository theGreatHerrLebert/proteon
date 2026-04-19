"""Oracle test: proteon H placement vs reduce (Richardson lab, Duke).

Reduce (https://github.com/rlabduke/reduce) is the canonical asymmetric
hydrogen placer — the Richardson lab's reference implementation, widely
trusted, independent of any MD engine. Our current hydrogen placement
is cross-checked against BALL + GROMACS (both MD-linked); reduce is the
third voice that doesn't share force-field assumptions.

What this pins
--------------
For every standard amino-acid hydrogen whose position is **geometrically
determined** by its parent heavy atom and the adjacent backbone, reduce
and proteon agree within 0.1 Å. That covers:

    parent         H count  where it appears
    ─────────────  ───────  ──────────────────────────────────────────
    N (backbone)   1        amide N-H on every non-N-terminal residue
    Cα             1 or 2   HA everywhere; HA2/HA3 on Gly (CH2)
    aromatic C     1        PHE/TYR/TRP/HIS ring C-H
    sp3 methylene  2        CB on polar+charged AAs, CG on ARG/GLN/…
                            CG1 on ILE, CD on ARG/PRO/LYS, CE on LYS
    sp3 methine    1        CB on VAL/ILE/THR (single H)
    sp2 indole N   1        TRP NE1-H
    sp2 imidazole  1        HIS ND1-H / NE2-H (protonated tautomer)

Everything above uses `−NUClear` in reduce to match proteon's nuclear
H-X bond lengths (C-H 1.09, N-H 1.01, O-H 0.96 Å). Without that flag
reduce defaults to X-ray-shortened positions that disagree by ~0.1 Å
systematically — that's the X-ray / neutron convention, not an
algorithm difference.

What this does *not* pin
------------------------
The three known convention gaps, documented here instead of asserted:

1. **Methyl groups (ALA/VAL/LEU/ILE/THR/MET)**: both tools place a valid
   CH₃ but at different 3-fold rotamer orientations. Average residual
   after optimal matching is ~0.5 Å. A tighter oracle would need an MD-
   level rotamer search that matches reduce's H-bond scoring — out of
   scope here.

2. **sp2 amide NH₂ (ASN HD21/HD22, GLN HE21/HE22, ARG NH1/NH2)**:
   in-plane placement convention differs by ~120° between reduce and
   proteon, producing ~1.2 Å residuals even after -NOFLIP + optimal
   matching. Both geometries are valid sp2; a pair-convention test
   would need a separate fixture and is deferred.

3. **Rotatable polar H (SER/THR/TYR OH, CYS SH, LYS NH3+, N-terminal
   NH3+)**: reduce optimizes rotamer via H-bond scoring, proteon
   places at template default. 1.5-2 Å residuals are structural, not
   bugs. A separate oracle over a curated high-resolution set would
   pin this once proteon gains rotatable-H optimization.

FF parametrization
------------------
Two runs per structure:

- `polar_only=True`  → CHARMM19-style placement (N-H, O-H, S-H only).
  We assert proteon's placed subset agrees with reduce's polar subset.
- `polar_only=False` → AMBER96-style placement (full H set).
  We assert two-way agreement on the rigid parents above.

Installation
------------
Reduce has no pip/apt package. Build from source:

    git clone https://github.com/rlabduke/reduce.git
    cmake -S reduce -B reduce/build && make -C reduce/build

Then point the test at the binary + het dictionary:

    export REDUCE_BIN=/path/to/reduce/build/reduce_src/reduce
    export REDUCE_DB=/path/to/reduce/reduce_wwPDB_het_dict.txt

The test skips cleanly if REDUCE_BIN isn't set or the binary isn't
executable.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pytest

import proteon

pytestmark = pytest.mark.oracle("reduce")


# ---------------------------------------------------------------------------
# Reduce binary resolution
# ---------------------------------------------------------------------------


def _resolve_reduce() -> Tuple[str, str] | None:
    """Return (binary, het_dict) or None if reduce isn't available."""
    binary = os.environ.get("REDUCE_BIN") or shutil.which("reduce")
    if binary is None:
        return None
    if not os.access(binary, os.X_OK):
        return None
    # Het dict: env var wins; else probe the standard build-layout sibling.
    db = os.environ.get("REDUCE_DB")
    if db is None:
        guess = os.path.normpath(
            os.path.join(os.path.dirname(binary), "..", "..", "reduce_wwPDB_het_dict.txt")
        )
        if os.path.exists(guess):
            db = guess
    if db is None or not os.path.exists(db):
        return None
    return binary, db


_REDUCE = _resolve_reduce()
if _REDUCE is None:
    pytest.skip(
        "reduce not available — set REDUCE_BIN (and optionally REDUCE_DB) or "
        "install per tests/oracle/README.md",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# PDB fixtures
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
PROTEON_PDBS = os.path.normpath(os.path.join(_HERE, "..", "..", "test-pdbs"))

STRUCTURES = [
    ("1crn", os.path.join(PROTEON_PDBS, "1crn.pdb")),
    ("1ubq", os.path.join(PROTEON_PDBS, "1ubq.pdb")),
]

TOLERANCE = 0.1   # Å — rigid-group per-atom distance after optimal matching


# ---------------------------------------------------------------------------
# Parsing + grouping helpers
# ---------------------------------------------------------------------------


AtomKey = Tuple[str, int, str]  # (chain, resnum, atom_name)
AtomPos = Tuple[str, np.ndarray]  # (resname, xyz)


def _parse_pdb_atoms(pdb_bytes: bytes) -> Tuple[Dict[AtomKey, AtomPos], Dict[AtomKey, AtomPos]]:
    """Return (heavy_atoms, hydrogens) indexed by (chain, resnum, name)."""
    heavy: Dict[AtomKey, AtomPos] = {}
    hdict: Dict[AtomKey, AtomPos] = {}
    for raw in pdb_bytes.split(b"\n"):
        if not raw.startswith(b"ATOM") or len(raw) < 78:
            continue
        element = raw[76:78].strip()
        name = raw[12:16].decode("ascii").strip()
        resname = raw[17:20].decode("ascii").strip()
        chain = raw[21:22].decode("ascii").strip()
        resnum = int(raw[22:26])
        pos = np.array(
            [float(raw[30:38]), float(raw[38:46]), float(raw[46:54])], dtype=np.float64
        )
        key: AtomKey = (chain, resnum, name)
        if element == b"H":
            hdict[key] = (resname, pos)
        else:
            heavy[key] = (resname, pos)
    return heavy, hdict


def _strip_hydrogens(path: str) -> str:
    """Write a hydrogen-free copy of `path` to a tempfile and return its path.

    Both tools otherwise preserve pre-existing H atoms (e.g. in 1ubq.pdb),
    which would compare the original X-ray refinement's H positions against
    themselves instead of comparing the two tools' placement templates.
    """
    fd, out = tempfile.mkstemp(suffix=".pdb", prefix="reduce_oracle_")
    with os.fdopen(fd, "wb") as f_out, open(path, "rb") as f_in:
        for line in f_in:
            if line.startswith((b"ATOM", b"HETATM")) and len(line) >= 78:
                if line[76:78].strip() == b"H":
                    continue
            f_out.write(line)
    return out


def _extract_proteon(path: str, *, polar_only: bool) -> Tuple[Dict[AtomKey, AtomPos], Dict[AtomKey, AtomPos]]:
    """Load with proteon, place H, return (heavy, h) dicts."""
    from proteon_connector import py_add_hydrogens

    structure = proteon.load(path)
    py_add_hydrogens.place_all_hydrogens(structure.get_py_ptr(), polar_only)
    heavy: Dict[AtomKey, AtomPos] = {}
    hdict: Dict[AtomKey, AtomPos] = {}
    for atom in structure.get_py_ptr().atoms:
        key = (atom.chain_id.strip(), atom.residue_serial_number, atom.name.strip())
        pos = np.array([atom.x, atom.y, atom.z], dtype=np.float64)
        entry = (atom.residue_name.strip(), pos)
        if atom.element == "H":
            hdict[key] = entry
        else:
            heavy[key] = entry
    return heavy, hdict


def _run_reduce(path: str) -> Tuple[Dict[AtomKey, AtomPos], Dict[AtomKey, AtomPos]]:
    binary, db = _REDUCE  # type: ignore[misc]
    # reduce returns exit 1 (ABANDONED_RC) when a flip/rotamer clique is too
    # large to enumerate exhaustively. Output is still a valid PDB and we
    # use it; only an empty stdout indicates a real failure.
    proc = subprocess.run(
        [binary, "-NOFLIP", "-Quiet", "-NUClear", "-DB", db, path],
        check=False,
        capture_output=True,
    )
    if not proc.stdout:
        raise RuntimeError(
            f"reduce produced no output for {path} (exit={proc.returncode}).\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')}"
        )
    return _parse_pdb_atoms(proc.stdout)


def _group_by_parent(
    heavy: Dict[AtomKey, AtomPos], h: Dict[AtomKey, AtomPos]
) -> Dict[Tuple[str, int, str, str, str], List[Tuple[str, np.ndarray]]]:
    """Index each H by its nearest heavy atom in the same residue.

    Returns {(chain, resnum, resname, parent_name, parent_element): [(h_name, h_pos), …]}.
    parent_element is the first character of the parent atom name (C, N, O, S)
    which doubles as the PDB element for our purposes on standard AAs.
    """
    groups: Dict[Tuple[str, int, str, str, str], List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for (chain, resnum, hname), (resname, hpos) in h.items():
        best_d, best_parent = 1e9, None
        for (c2, r2, pname), (_, ppos) in heavy.items():
            if c2 == chain and r2 == resnum:
                d = float(np.linalg.norm(hpos - ppos))
                if d < best_d:
                    best_d, best_parent = d, pname
        if best_parent is not None and best_d < 1.5:  # H-X bond max ~1.1 Å
            groups[(chain, resnum, resname, best_parent, best_parent[0])].append((hname, hpos))
    return groups


# sp2 guanidinium/amide single-H positions where reduce and proteon use
# different in-plane conventions (~1 Å residual even with -NOFLIP). Classified
# alongside the N+2 sp2 amides for the same "convention gap" reason.
_SP2_CONVENTION_PARENTS = frozenset({("ARG", "NE")})


def _is_rigid_parent(
    parent_element: str, resname: str, parent_name: str, h_count: int
) -> bool:
    """Classify whether this parent's H group is geometrically determined.

    Carbon with 1 or 2 H: methine or methylene → rigid.
    Carbon with 3 H: methyl — 3-fold rotation convention differs between
        tools, excluded from the tight oracle.
    Nitrogen with 1 H: backbone amide, Trp indole, His ring → rigid.
        Exception: ARG NE (guanidinium sp2) — in-plane placement convention
        differs between tools (~1 Å), excluded.
    Nitrogen with 2 H: sp2 amide NH₂ (ASN/GLN/ARG) — convention gap.
    Nitrogen with 3 H: NH3+ (N-terminal, Lys NZ) — rotatable.
    Oxygen with 1 H: OH (Ser/Thr/Tyr) — rotatable, excluded.
    Sulfur with 1 H: SH (Cys) — rotatable, excluded.
    """
    if parent_element == "C":
        return h_count in (1, 2)
    if parent_element == "N":
        if h_count != 1:
            return False
        return (resname, parent_name) not in _SP2_CONVENTION_PARENTS
    return False


def _optimal_match(
    group_a: List[Tuple[str, np.ndarray]], group_b: List[Tuple[str, np.ndarray]]
) -> List[float] | None:
    """Return per-atom distances under the minimum-sum matching.

    Groups hold at most 3 H atoms (methyl upper bound), so brute-force
    permutations are fine — no scipy dependency needed.
    """
    if len(group_a) != len(group_b):
        return None
    n = len(group_a)
    positions_a = [p for _, p in group_a]
    positions_b = [p for _, p in group_b]
    best_perm, best_cost = None, float("inf")
    for perm in itertools.permutations(range(n)):
        cost = float(sum(float(np.linalg.norm(positions_a[i] - positions_b[perm[i]])) for i in range(n)))
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    assert best_perm is not None
    return [float(np.linalg.norm(positions_a[i] - positions_b[best_perm[i]])) for i in range(n)]


def _collect_rigid_residuals(
    heavy_r: Dict[AtomKey, AtomPos],
    h_r: Dict[AtomKey, AtomPos],
    heavy_f: Dict[AtomKey, AtomPos],
    h_f: Dict[AtomKey, AtomPos],
) -> List[Tuple[Tuple[str, int, str, str], float, str, str]]:
    """Return per-H residuals for the rigid subset shared by both tools.

    Each entry is ((chain, resnum, resname, parent), distance,
    reduce_h_name, proteon_h_name).
    """
    groups_r = _group_by_parent(heavy_r, h_r)
    groups_f = _group_by_parent(heavy_f, h_f)
    out: List[Tuple[Tuple[str, int, str, str], float, str, str]] = []
    for key in set(groups_r) & set(groups_f):
        chain, resnum, resname, parent_name, parent_elem = key
        group_r = sorted(groups_r[key])
        group_f = sorted(groups_f[key])
        if len(group_r) != len(group_f):
            # Missing H's on one side — not a rigid-group comparison.
            continue
        if not _is_rigid_parent(parent_elem, resname, parent_name, len(group_r)):
            continue
        dists = _optimal_match(group_r, group_f)
        if dists is None:
            continue
        # Recover the permutation to report names in the diff.
        perm, _ = min(
            (
                (
                    perm,
                    sum(
                        float(np.linalg.norm(group_r[i][1] - group_f[perm[i]][1]))
                        for i in range(len(group_r))
                    ),
                )
                for perm in itertools.permutations(range(len(group_f)))
            ),
            key=lambda x: x[1],
        )
        for i in range(len(group_r)):
            out.append(
                ((chain, resnum, resname, parent_name), dists[i], group_r[i][0], group_f[perm[i]][0])
            )
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReduceHydrogenOracle:
    """Per-atom parity on the rigid H subset, parametrized over FF mode."""

    @pytest.fixture(
        scope="class",
        params=STRUCTURES,
        ids=[s[0] for s in STRUCTURES],
    )
    def stripped_pdb(self, request):
        """H-stripped copy of the test PDB, used as input to both tools so
        neither preserves any pre-existing refined H positions."""
        _, path = request.param
        if not os.path.exists(path):
            pytest.skip(f"test PDB missing: {path}")
        stripped = _strip_hydrogens(path)
        yield stripped
        try:
            os.unlink(stripped)
        except OSError:
            pass

    @pytest.fixture(scope="class")
    def reduce_atoms(self, stripped_pdb):
        return _run_reduce(stripped_pdb)

    @pytest.fixture(params=[False, True], ids=["amber96_full", "charmm19_polar"])
    def polar_only(self, request):
        return request.param

    def test_rigid_parity(self, request, stripped_pdb, reduce_atoms, polar_only):
        name = request.node.callspec.id.split("-")[0]
        heavy_r, h_r = reduce_atoms
        heavy_f, h_f = _extract_proteon(stripped_pdb, polar_only=polar_only)

        residuals = _collect_rigid_residuals(heavy_r, h_r, heavy_f, h_f)
        assert residuals, (
            f"{name} ({'polar-only' if polar_only else 'full-H'}): "
            "no rigid H groups found — comparison machinery broke."
        )

        over = [entry for entry in residuals if entry[1] > TOLERANCE]
        n_total = len(residuals)
        if over:
            worst = sorted(over, key=lambda x: -x[1])[:10]
            lines = [
                f"  {r[0][2]:>3} {r[0][3]:<4} resid={r[0][1]:<3}  "
                f"reduce={r[2]:<5} proteon={r[3]:<5}  d={r[1]:.3f} Å"
                for r in worst
            ]
            raise AssertionError(
                f"{name} ({'polar-only' if polar_only else 'full-H'}): "
                f"{len(over)}/{n_total} rigid H atoms exceed {TOLERANCE} Å "
                f"after optimal matching.\nWorst offenders:\n" + "\n".join(lines)
            )

    def test_backbone_amide_coverage(self, request, stripped_pdb, reduce_atoms, polar_only):
        """Every non-proline, non-N-terminal residue should have an amide N-H
        in both outputs — basic sanity check that the placement ran."""
        name = request.node.callspec.id.split("-")[0]
        _, h_r = reduce_atoms
        _, h_f = _extract_proteon(stripped_pdb, polar_only=polar_only)

        n_r = sum(1 for key in h_r if key[2] == "H")
        n_f = sum(1 for key in h_f if key[2] == "H")
        assert abs(n_r - n_f) <= 2, (
            f"{name}: backbone amide H count diverges — reduce={n_r} "
            f"proteon={n_f}. Proline / terminal handling likely differs."
        )
