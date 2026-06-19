"""Per-residue completeness validity + coverage gate (phase 1 of masking)."""

import os

import numpy as np
import pytest

import proteon
from proteon.residue_mask import DEFAULT_COVERAGE, residue_completeness, structure_coverage

PDBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def _load(path):
    return proteon.load(path)


class TestStructureCoverage:
    def test_complete_structure_is_full_coverage(self):
        cov = structure_coverage(_load(os.path.join(PDBS, "1crn.pdb")))
        assert cov.n_residues == 46
        assert cov.n_valid == 46
        assert cov.coverage == 1.0
        assert cov.node_valid.dtype == bool
        assert cov.node_valid.all()

    def test_node_valid_aligned_and_sized(self):
        # node_valid is one bool per amino-acid residue, in residue_index order.
        cov = structure_coverage(_load(os.path.join(PDBS, "1crn.pdb")))
        assert cov.node_valid.shape == (cov.n_residues,)

    def test_missing_backbone_atom_masks_exactly_that_residue(self):
        # missing_backbone_c: residue index 1 is missing a backbone C.
        cov = structure_coverage(
            _load(os.path.join(CORPUS, "missing_atoms", "missing_backbone_c.pdb")),
            profile="backbone",
        )
        assert cov.node_valid.tolist() == [True, False, True]
        assert cov.coverage == pytest.approx(2 / 3)

    def test_backbone_profile_is_laxer_than_heavy(self):
        # missing_cb: backbone intact everywhere, but a side-chain atom missing.
        p = os.path.join(CORPUS, "missing_atoms", "missing_cb.pdb")
        heavy = structure_coverage(_load(p), profile="heavy_coords")
        backbone = structure_coverage(_load(p), profile="backbone")
        assert backbone.coverage >= heavy.coverage
        assert backbone.coverage == 1.0  # all backbones present

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError):
            structure_coverage(_load(os.path.join(PDBS, "1crn.pdb")), profile="nonsense")

    def test_report_makes_coverage_count_clashes_invalid(self):
        # With a report, coverage drops because clashing residues become invalid.
        s = _load(os.path.join(PDBS, "4hhb.pdb"))
        r = proteon.prepare(s, minimize=False)
        assert r.clash_residue_indices  # 4hhb clashes
        no_rep = structure_coverage(s, chain_id="A")
        with_rep = structure_coverage(s, chain_id="A", report=r)
        assert with_rep.n_valid < no_rep.n_valid
        assert with_rep.coverage < no_rep.coverage

    def test_mild_clash_does_not_reduce_coverage(self):
        # 1bpi: clashes present but NOT severe (clashscore 11). Coverage must skip
        # them (mild clashes are tolerated heavy-coordinate labels) (codex) — only
        # completeness + altloc apply, NOT clash.
        from proteon.residue_mask import residue_completeness, residue_trustworthy

        s = _load(os.path.join(PDBS, "1bpi.pdb"))
        r = proteon.prepare(s, minimize=False)
        assert r.clash_residue_indices and not r.has_severe_clashes
        res = [x for x in s.models[0].chains[0].residues if x.is_amino_acid]
        complete, altloc = residue_completeness(res), residue_trustworthy(res)
        expected = int((complete & altloc).sum())  # no clash term
        assert structure_coverage(s, report=r).n_valid == expected
        # ...and clash WOULD have reduced it, proving it was genuinely skipped.
        clash_free = proteon.residue_clash_mask(s, r.clash_residue_indices, "A")
        assert int((complete & altloc & clash_free).sum()) < expected

    def test_report_clean_structure_same_coverage(self):
        s = _load(os.path.join(PDBS, "1crn.pdb"))
        r = proteon.prepare(s, minimize=False)
        assert structure_coverage(s, report=r).coverage == structure_coverage(s).coverage == 1.0

    def test_coverage_property_zero_when_no_residues(self):
        from proteon.residue_mask import ResidueCoverage

        c = ResidueCoverage(profile="heavy_coords", n_residues=0, n_valid=0,
                            node_valid=np.zeros((0,), dtype=bool))
        assert c.coverage == 0.0


class TestResidueCompleteness:
    def test_empty_residues(self):
        assert residue_completeness([]).shape == (0,)


class TestCoverageGate:
    def _diverse(self, n):
        import glob
        corpus = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "validation", "pdbs_10k", "*.pdb")))
        corpus = [p for p in corpus if os.path.exists(p)]
        return corpus[::200][:n]

    def test_min_coverage_attaches_to_all_results(self):
        paths = [os.path.join(PDBS, "1crn.pdb"), os.path.join(PDBS, "1ubq.pdb")]
        res = proteon.prepare_for_supervision(paths, min_coverage=0.8, minimize=False)
        for r in res:
            if r.loaded:
                assert r.coverage is not None
                assert r.coverage_info is not None
                assert 0.0 <= r.coverage <= 1.0

    def test_no_coverage_by_default(self):
        # Without min_coverage, coverage stays None (no extra work, no change).
        res = proteon.prepare_for_supervision([os.path.join(PDBS, "1crn.pdb")], minimize=False)
        assert res[0].coverage is None
        assert res[0].coverage_info is None

    def test_only_safe_filters_on_coverage(self):
        # A coverage floor of 1.0 keeps only the fully-complete structures; the
        # gate is coverage, independent of label_safe (1crn has clashes? no — it's
        # pristine, coverage 1.0).
        paths = self._diverse(10)
        kept = proteon.prepare_for_supervision(
            paths, min_coverage=1.0, only_safe=True, minimize=False)
        allr = proteon.prepare_for_supervision(
            paths, min_coverage=1.0, minimize=False)
        assert all(r.coverage == 1.0 for r in kept)
        assert len(kept) <= len([r for r in allr if r.loaded])
        # a lower floor keeps at least as many
        kept_loose = proteon.prepare_for_supervision(
            paths, min_coverage=0.7, only_safe=True, minimize=False)
        assert len(kept_loose) >= len(kept)

    def test_default_coverage_constant(self):
        assert DEFAULT_COVERAGE == 0.8

    def test_coverage_gate_still_rejects_other_hazards(self):
        # 4hhb is complete but has severe clashes + 4 chains. Coverage masks only
        # missing atoms, so it must NOT pass the coverage gate (codex P1). It is
        # also multi-chain (unscored -> dropped); assert it is excluded.
        kept = proteon.prepare_for_supervision(
            [os.path.join(PDBS, "4hhb.pdb")], min_coverage=0.5,
            only_safe=True, minimize=False)
        assert kept == []

    def _keep_with(self, report, frac_valid):
        # _supervision_keep over a result whose coverage already reflects the
        # report (the clash residues counted invalid -> frac_valid).
        from proteon.prepare import _supervision_keep
        from proteon.residue_mask import ResidueCoverage

        class _R:
            coverage_info = ResidueCoverage("heavy_coords", 100, int(frac_valid * 100),
                                            np.ones((100,), dtype=bool))

            @property
            def coverage(self):
                return self.coverage_info.coverage

        r = _R()
        r.report = report
        return _supervision_keep(r, repair=None, min_coverage=0.8)

    def test_severe_clash_is_coverage_decided_not_auto_excluded(self):
        # Gate-relaxation: severe clash is localized + masked, so it no longer
        # auto-excludes — coverage (which already counts clash residues invalid)
        # decides. Localized (high coverage) kept; pervasive (low) dropped.
        from proteon import PrepReport
        rep = PrepReport(hydrogens_added=50, n_heavy_clashes=50, n_heavy_atoms=200)
        assert "severe_heavy_clashes" in rep.label_hazards
        assert self._keep_with(rep, 0.9) is True   # localized -> kept + masked
        assert self._keep_with(rep, 0.6) is False  # pervasive -> dropped

    def test_unmasked_hazard_still_excludes_at_full_coverage(self):
        # Chirality is NOT localized by coverage masking -> still drops the whole
        # structure even at coverage 1.0.
        from proteon import PrepReport
        rep = PrepReport(hydrogens_added=50, n_chirality_outliers=1)
        assert "chirality_outliers" in rep.label_hazards
        assert self._keep_with(rep, 1.0) is False

    def test_multichain_requires_chain_id(self):
        cov = structure_coverage(_load(os.path.join(PDBS, "4hhb.pdb")), chain_id="A")
        assert cov.n_residues > 0
        with pytest.raises(ValueError):
            structure_coverage(_load(os.path.join(PDBS, "4hhb.pdb")))  # 4 chains, no id

    def test_bad_coverage_profile_raises_up_front(self):
        with pytest.raises(ValueError):
            proteon.prepare_for_supervision(
                [os.path.join(PDBS, "1crn.pdb")], min_coverage=0.8,
                coverage_profile="nonsense", minimize=False)
