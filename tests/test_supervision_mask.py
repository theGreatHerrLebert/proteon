"""Per-residue trustworthiness masking into the supervision export (phase 2)."""

import os

import numpy as np
import pytest

import proteon
from proteon.residue_mask import residue_trustworthy
from proteon.supervision import build_structure_supervision_example
from proteon.supervision_mask import apply_residue_trust_mask

PDBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
ALTLOC = os.path.join(CORPUS, "altloc", "dual_conformer.pdb")


def _residues(path):
    s = proteon.load(path)
    return s, [r for r in s.models[0].chains[0].residues if r.is_amino_acid]


class TestResidueTrustworthy:
    def test_altloc_residue_is_untrustworthy(self):
        _, residues = _residues(ALTLOC)
        trust = residue_trustworthy(residues)
        assert trust.tolist() == [True, False, True]  # VAL2 has altlocs

    def test_clean_structure_all_trustworthy(self):
        _, residues = _residues(os.path.join(PDBS, "1crn.pdb"))
        assert residue_trustworthy(residues).all()

    def test_unknown_hazard_rejected(self):
        _, residues = _residues(os.path.join(PDBS, "1crn.pdb"))
        with pytest.raises(ValueError):
            residue_trustworthy(residues, hazards=("nonsense",))


class TestApplyTrustMask:
    def test_self_mask_zeroed_at_untrusted_residue(self):
        s, _ = _residues(ALTLOC)
        base = build_structure_supervision_example(s)
        masked = build_structure_supervision_example(s, mask_untrustworthy_coords=True)
        # residue index 1 (VAL2) zeroed; neighbours' OWN atoms intact.
        assert base.all_atom_mask.sum(axis=1).tolist() == [4, 4, 4]
        assert masked.all_atom_mask.sum(axis=1).tolist() == [4, 0, 4]
        assert masked.atom14_gt_exists[1].sum() == 0

    def test_phi_uses_prev_neighbour(self):
        s, _ = _residues(ALTLOC)
        base = build_structure_supervision_example(s)
        masked = build_structure_supervision_example(s, mask_untrustworthy_coords=True)
        # phi[i] depends on i-1: untrust at 1 kills phi[1] (self) and phi[2] (i-1).
        assert base.phi_mask.tolist() == [0, 1, 1]
        assert masked.phi_mask.tolist() == [0, 0, 0]

    def test_psi_uses_next_neighbour(self):
        s, _ = _residues(ALTLOC)
        base = build_structure_supervision_example(s)
        masked = build_structure_supervision_example(s, mask_untrustworthy_coords=True)
        # psi[i] depends on i+1: untrust at 1 kills psi[0] (i+1) and psi[1] (self).
        assert base.psi_mask.tolist() == [1, 1, 0]
        assert masked.psi_mask.tolist() == [0, 0, 0]

    def test_torsion_mask_column_dependencies(self):
        s, residues = _residues(ALTLOC)
        base = build_structure_supervision_example(s)
        masked = build_structure_supervision_example(s, mask_untrustworthy_coords=True)
        tb, tm = base.torsion_angles_mask, masked.torsion_angles_mask
        # cols [pre_omega(0), phi(1)] read residue i-1 -> rows 1,2 affected
        for c in (0, 1):
            assert masked_implies(tb[:, c], tm[:, c], killed={1, 2})
        # AF psi(2) is RESIDUE-LOCAL (own N,CA,C,O) -> only row 1, NOT i+1
        assert masked_implies(tb[:, 2], tm[:, 2], killed={1})
        # chi cols (3..6) use self -> only row 1
        for c in (3, 4, 5, 6):
            assert masked_implies(tb[:, c], tm[:, c], killed={1})

    def test_seq_mask_and_positions_untouched(self):
        s, _ = _residues(ALTLOC)
        base = build_structure_supervision_example(s)
        masked = build_structure_supervision_example(s, mask_untrustworthy_coords=True)
        assert np.array_equal(base.seq_mask, masked.seq_mask)
        assert np.array_equal(base.aatype, masked.aatype)
        assert np.array_equal(base.residue_index, masked.residue_index)
        assert np.array_equal(base.all_atom_positions, masked.all_atom_positions)

    def test_default_off_is_byte_identical(self):
        # The whole point of opt-in: default preserves the oracle-gated tensors.
        s, _ = _residues(ALTLOC)
        a = build_structure_supervision_example(s)
        b = build_structure_supervision_example(s)
        assert np.array_equal(a.all_atom_mask, b.all_atom_mask)
        assert np.array_equal(a.phi_mask, b.phi_mask)

    def test_all_trustworthy_returns_unchanged(self):
        s, _ = _residues(os.path.join(PDBS, "1crn.pdb"))
        ex = build_structure_supervision_example(s)
        same = apply_residue_trust_mask(ex, np.ones((ex.length,), dtype=bool))
        assert same is ex  # short-circuit, no copy

    def test_wrong_trust_shape_raises(self):
        s, _ = _residues(os.path.join(PDBS, "1crn.pdb"))
        ex = build_structure_supervision_example(s)
        with pytest.raises(ValueError):
            apply_residue_trust_mask(ex, np.ones((ex.length + 1,), dtype=bool))

    def test_batch_builder_forwards_flag(self):
        s, _ = _residues(ALTLOC)
        out = proteon.supervision.batch_build_structure_supervision_examples(
            [s], mask_untrustworthy_coords=True)
        assert out[0].all_atom_mask.sum(axis=1).tolist() == [4, 0, 4]


class TestClashAttribution:
    def _hhb(self):
        s = proteon.load(os.path.join(PDBS, "4hhb.pdb"))
        return s, proteon.prepare(s, minimize=False)

    def test_clash_indices_sorted_unique_and_protein_scoped(self):
        _, r = self._hhb()
        assert r.clash_residue_indices == sorted(set(r.clash_residue_indices))
        assert len(r.clash_residue_indices) > 0  # 4hhb has clashes

    def test_clash_mask_alignment_sums_to_report(self):
        # The per-chain clash masks, summed, must equal the report's index count:
        # proof the Python residue walk aligns with the Rust topology res_idx.
        s, r = self._hhb()
        total = 0
        for ch in s.models[0].chains:
            m = proteon.residue_clash_mask(s, r.clash_residue_indices, ch.id)
            total += int((~m).sum())
        assert total == len(r.clash_residue_indices)

    def test_clean_structure_no_clash_residues(self):
        s = proteon.load(os.path.join(PDBS, "1crn.pdb"))
        r = proteon.prepare(s, minimize=False)
        assert r.clash_residue_indices == []
        mask = proteon.residue_clash_mask(s, r.clash_residue_indices, "A")
        assert mask.all()

    def test_clashing_residues_masked_in_export(self):
        s, r = self._hhb()
        clash_free = proteon.residue_clash_mask(s, r.clash_residue_indices, "A")
        base = build_structure_supervision_example(s, chain_id="A")
        masked = build_structure_supervision_example(
            s, chain_id="A", prep_report=r, mask_untrustworthy_coords=True)
        bsum, msum = base.all_atom_mask.sum(axis=1), masked.all_atom_mask.sum(axis=1)
        assert (msum[~clash_free] == 0).all()           # clashing residues zeroed
        assert (msum[clash_free] == bsum[clash_free]).all()  # clash-free unchanged
        assert np.array_equal(base.seq_mask, masked.seq_mask)

    def test_no_clash_data_falls_back_to_altloc_only(self):
        # mask flag on but prep_report=None -> only altloc masking, no crash.
        s, residues = _residues(ALTLOC)
        ex = build_structure_supervision_example(s, mask_untrustworthy_coords=True)
        assert ex.all_atom_mask.sum(axis=1).tolist() == [4, 0, 4]  # altloc only


def masked_implies(base_col, masked_col, killed):
    """masked_col is base_col with exactly the `killed` rows zeroed (where base
    had a 1) and all other rows preserved."""
    for i in range(len(base_col)):
        if i in killed:
            if masked_col[i] != 0:
                return False
        elif masked_col[i] != base_col[i]:
            return False
    return True
