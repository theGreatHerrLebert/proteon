"""Cross-check proteon's heavy-atom clash metric against MolProbity.

MolProbity (`phenix.clashscore`, or `probe`+`reduce`) is the community-standard
steric-clash oracle. proteon's `n_heavy_clashes` is a heavy-atom-only count
(MolProbity's clashscore is all-atom incl. hydrogens, per 1000 atoms), so the
two are NOT in the same units — but they must AGREE IN RANK: a structure
MolProbity rates clashy must have more proteon clashes than one it rates clean.

Skip-if-absent: when no MolProbity backend is installed (the common CI case),
this test skips rather than failing — like the other proteon oracles.
"""

import os
import re
import shutil
import subprocess

import pytest

import proteon

pytestmark = pytest.mark.oracle("molprobity")

TEST_PDBS = os.path.join(os.path.dirname(__file__), "..", "..", "test-pdbs")
# A spread of structures from pristine (1crn, 0.5 Å) to old/low-res (4hhb, 1984).
STRUCTURES = ["1crn", "1ubq", "1enh", "1bpi", "1ake", "4hhb"]


def _molprobity_clashscore(pdb_path):
    """Return MolProbity clashscore for a PDB, or None if no backend is usable."""
    phenix = shutil.which("phenix.clashscore") or shutil.which("molprobity.clashscore")
    if phenix is None:
        return None
    try:
        out = subprocess.run(
            [phenix, pdb_path], capture_output=True, text=True, timeout=300
        )
    except (subprocess.SubprocessError, OSError):
        return None
    # phenix.clashscore prints e.g. "clashscore = 12.34"
    m = re.search(r"clashscore\s*=\s*([0-9.]+)", out.stdout)
    return float(m.group(1)) if m else None


def _spearman(xs, ys):
    """Spearman rank correlation (no scipy dependency)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) ** 0.5
        * sum((b - my) ** 2 for b in ry) ** 0.5
    )
    return num / den if den else 0.0


def test_clash_metric_agrees_with_molprobity_in_rank():
    proteon_counts = []
    molprobity_scores = []
    for name in STRUCTURES:
        path = os.path.join(TEST_PDBS, f"{name}.pdb")
        if not os.path.exists(path):
            continue
        score = _molprobity_clashscore(path)
        if score is None:
            pytest.skip("no MolProbity backend (phenix.clashscore / molprobity.clashscore)")
        report = proteon.prepare(proteon.load(path))
        proteon_counts.append(report.n_heavy_clashes)
        molprobity_scores.append(score)

    if len(proteon_counts) < 3:
        pytest.skip("need >=3 structures with MolProbity scores to correlate")

    rho = _spearman(proteon_counts, molprobity_scores)
    assert rho > 0.5, (
        f"proteon clash count should rank-agree with MolProbity clashscore, "
        f"got Spearman rho={rho:.2f}\n"
        f"proteon={proteon_counts}\nmolprobity={molprobity_scores}"
    )


def test_molprobity_clean_structures_have_few_proteon_clashes():
    # Structures MolProbity rates essentially clash-free (clashscore < 5) must
    # also be near-zero under proteon's heavier-grained heavy-atom metric.
    checked = 0
    for name in STRUCTURES:
        path = os.path.join(TEST_PDBS, f"{name}.pdb")
        if not os.path.exists(path):
            continue
        score = _molprobity_clashscore(path)
        if score is None:
            pytest.skip("no MolProbity backend available")
        if score < 5.0:
            report = proteon.prepare(proteon.load(path))
            assert report.n_heavy_clashes <= 5, (
                f"{name}: MolProbity clashscore {score:.1f} (clean) but proteon "
                f"reports {report.n_heavy_clashes} heavy clashes"
            )
            checked += 1
    if checked == 0:
        pytest.skip("no MolProbity-clean structures in the set")
