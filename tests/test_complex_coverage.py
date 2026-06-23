"""Multi-chain coverage + the interface-mode complex export (label-safe substrate)."""

import os

import numpy as np
import pytest

import proteon
from proteon.residue_mask import complex_coverage
from proteon.supervision import (
    ComplexSupervisionExamples,
    build_complex_supervision_examples,
)

PDBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-pdbs")


def _prep(name):
    res = proteon.prepare_for_supervision([os.path.join(PDBS, f"{name}.pdb")], minimize=False)[0]
    return res.structure, res.report


def _prep_no_incidental_chirality(name):
    # 4hhb carries ONE borderline chirality outlier (a real but incidental quirk).
    # The gate-mechanics tests below isolate the assembly/coverage logic, so clear
    # it; the unmasked-hazard test exercises the chirality block separately.
    s, r = _prep(name)
    r.n_chirality_outliers = 0
    return s, r


class TestComplexCoverage:
    def test_per_chain_coverages_present(self):
        s, r = _prep("4hhb")
        cc = complex_coverage(s, report=r)
        assert cc.n_protein_chains == 4
        assert set(cc.chains) == {"A", "B", "C", "D"}
        for cov in cc.chains.values():
            assert 0.0 <= cov.coverage <= 1.0

    def test_min_is_weakest_chain(self):
        s, r = _prep("4hhb")
        cc = complex_coverage(s, report=r)
        assert cc.min_coverage == min(c.coverage for c in cc.chains.values())
        assert cc.min_coverage <= cc.total_coverage or abs(cc.min_coverage - cc.total_coverage) < 0.2

    def test_clash_indices_partition_across_chains_exactly(self):
        # The shared global clash_residue_indices must map to per-chain residues
        # with NO loss or double-count — the alignment proof (claudex). Sum of the
        # per-chain clashing residues == the global count.
        s, r = _prep("4hhb")
        assert r.has_severe_clashes and r.clash_residue_indices
        total = 0
        for cid in cc_chain_ids(s):
            m = proteon.residue_clash_mask(s, r.clash_residue_indices, cid)
            total += int((~m).sum())
        assert total == len(r.clash_residue_indices)

    def test_clean_complex_full_coverage(self):
        # A hypothetical clean multi-chain: every chain 1.0. (4hhb is severe, so we
        # check the no-report path = completeness only, which 4hhb passes fully.)
        s, _ = _prep("4hhb")
        cc = complex_coverage(s)  # no report -> completeness only
        assert cc.min_coverage == 1.0


def cc_chain_ids(structure):
    return [
        ch.id for ch in structure.models[0].chains if any(x.is_amino_acid for x in ch.residues)
    ]


class TestInterfaceGate:
    def test_verified_tetramer_kept(self):
        s, r = _prep_no_incidental_chirality("4hhb")  # assembly_is_asu True, 4 chains
        out = build_complex_supervision_examples(s, prep_report=r, min_coverage=0.5)
        assert isinstance(out, ComplexSupervisionExamples)
        assert out.chain_order == ["A", "B", "C", "D"]
        assert set(out.chain_examples) == {"A", "B", "C", "D"}
        assert out.assembly_is_asu is True
        assert out.coverage.n_protein_chains == 4

    def test_false_assembly_requires_expansion(self):
        s, r = _prep("1ake")  # assembly_is_asu False (two separate monomers)
        assert r.assembly_is_asu is False
        assert build_complex_supervision_examples(s, prep_report=r, min_coverage=0.5) == (
            "requires_assembly_expansion"
        )

    def test_single_chain_not_a_complex(self):
        s, r = _prep("1crn")  # assembly_is_asu True but ONE chain
        assert r.assembly_is_asu is True
        assert build_complex_supervision_examples(s, prep_report=r, min_coverage=0.5) == (
            "not_a_complex"
        )

    def test_unverified_assembly_dropped(self):
        # No REMARK 350 path annotation -> assembly_is_asu None -> dropped distinctly.
        from proteon import PrepReport
        s = proteon.load(os.path.join(PDBS, "4hhb.pdb"))
        r = proteon.prepare(s, minimize=False)  # no path -> assembly_is_asu None
        assert r.assembly_is_asu is None
        assert build_complex_supervision_examples(s, prep_report=r, min_coverage=0.5) == (
            "assembly_unverified"
        )

    def test_unmasked_hazard_blocks_complex_export(self):
        # A verified multi-chain assembly with high coverage but an UNMASKED
        # coordinate hazard (chirality) must still drop — coverage only masks
        # missing/altloc/clash, not chirality (codex). Patch a chirality outlier
        # onto an otherwise-keepable complex.
        s, r = _prep("4hhb")
        r.n_chirality_outliers = 1
        assert "chirality_outliers" in r.label_hazards
        out = build_complex_supervision_examples(s, prep_report=r, min_coverage=0.3)
        assert out == "unmasked_hazard:chirality_outliers"

    def test_high_floor_drops_on_coverage(self):
        s, r = _prep_no_incidental_chirality("4hhb")  # severe clashes -> chains ~0.63
        assert build_complex_supervision_examples(s, prep_report=r, min_coverage=0.9) == (
            "below_coverage_floor"
        )

    def test_kept_chains_are_masked(self):
        # The per-chain examples carry the cross-chain clash masking.
        s, r = _prep_no_incidental_chirality("4hhb")
        out = build_complex_supervision_examples(s, prep_report=r, min_coverage=0.5)
        a = out.chain_examples["A"]
        masked = int((a.all_atom_mask.sum(axis=1) == 0).sum())
        assert masked > 0  # 4hhb chain A has clashing residues, masked
        # identity untouched
        assert a.seq_mask.sum() == a.length
