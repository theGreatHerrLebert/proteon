"""prepare_for_supervision — label-safe-by-default load+prepare for DL.

Same pipeline as batch_load_and_prepare, but reconstruct defaults to False (a
reconstructed atom is a model-derived guess, not an experimental label) and the
result exposes label_safe / label_hazards as the gate.
"""

import os

import pytest

import proteon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDBS = os.path.join(REPO, "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def _pdb(name):
    return os.path.join(PDBS, name)


class TestPrepareForSupervision:
    def test_reconstruct_off_by_default(self):
        # The key DL-safety default: missing atoms are NOT fabricated into labels.
        fixture = os.path.join(CORPUS, "missing_atoms", "missing_cb.pdb")
        r = proteon.prepare_for_supervision([fixture])[0]
        assert r.report is not None
        assert r.report.atoms_reconstructed == 0

    def test_incomplete_structure_is_not_label_safe(self):
        # With reconstruct off, a residue missing heavy atoms must surface as a
        # `missing_atoms` hazard — NOT silently pass as label_safe (codex).
        fixture = os.path.join(CORPUS, "missing_atoms", "missing_cb.pdb")
        r = proteon.prepare_for_supervision([fixture])[0]
        assert r.report.n_missing_heavy_atoms > 0
        assert r.label_safe is False
        assert "missing_atoms" in r.label_hazards

    def test_reconstruct_can_be_re_enabled(self):
        fixture = os.path.join(CORPUS, "missing_atoms", "missing_cb.pdb")
        r = proteon.prepare_for_supervision([fixture], reconstruct=True)[0]
        assert r.report.atoms_reconstructed > 0
        # ...and those fabricated atoms make it not label_safe.
        assert r.label_safe is False
        assert "reconstructed_atoms" in r.label_hazards

    def test_clean_structure_is_label_safe(self):
        r = proteon.prepare_for_supervision([_pdb("1crn.pdb")])[0]
        assert r.label_safe is True
        assert r.label_hazards == []

    def test_length_and_order_preserved(self):
        paths = [_pdb("1crn.pdb"), _pdb("4hhb.pdb"), _pdb("1ubq.pdb")]
        res = proteon.prepare_for_supervision(paths)
        assert [r.path for r in res] == [str(p) for p in paths]

    def test_load_failure_is_not_label_safe(self, tmp_path):
        junk = tmp_path / "x.pdb"
        junk.write_text("not a structure\n")
        r = proteon.prepare_for_supervision([str(junk)])[0]
        assert r.loaded is False
        assert r.label_safe is False
        assert r.label_hazards == ["load_failed"]

    def test_only_safe_filters_out_unsafe(self):
        # 4hhb (old, clashy + heme) is not label_safe; 1crn is.
        paths = [_pdb("1crn.pdb"), _pdb("4hhb.pdb")]
        safe = proteon.prepare_for_supervision(paths, only_safe=True)
        assert all(r.label_safe for r in safe)
        assert _pdb("1crn.pdb") in [r.path for r in safe]
        assert _pdb("4hhb.pdb") not in [r.path for r in safe]

    def test_profiles_available_via_report(self):
        # The per-label-type profiles are reachable for consumers that tolerate
        # specific hazards (e.g. heavy-coord labels on a heme protein).
        r = proteon.prepare_for_supervision([_pdb("4hhb.pdb")])[0]
        assert hasattr(r.report, "label_safe_heavy_coords")
        assert hasattr(r.report, "label_safe_energy")
