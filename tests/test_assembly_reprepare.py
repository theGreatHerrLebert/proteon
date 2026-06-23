"""Re-prepare the built assembly (PR2): new inter-copy interfaces are validated."""

import os

import pytest

import proteon
from proteon.assembly_builder import PreparedAssembly, prepare_assembly

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "validation", "pdbs_10k")
PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")


def _corpus(name):
    p = os.path.join(CORPUS, f"{name}.pdb")
    if not os.path.exists(p):
        pytest.skip(f"validation corpus missing: {name}")
    return p


class TestPrepareAssembly:
    def test_returns_prepared_assembly(self):
        pa = prepare_assembly(_corpus("1doe"), reconstruct=False, minimize=False)
        assert isinstance(pa, PreparedAssembly)
        assert pa.n_chains == 2
        assert pa.report is not None
        assert sorted(set(pa.structure.chain_ids)) == ["A", "B"]

    def test_new_interface_clashes_are_detected(self):
        # The whole point of re-prepare: a steric overlap that exists ONLY between
        # symmetry copies (absent from the deposited ASU) is seen by the normal
        # clash scan. The dimer's clash count exceeds twice the ASU's, i.e. the
        # new A–B interface contributes clashes the ASU scan never saw.
        p = _corpus("1doe")
        asu = proteon.prepare(proteon.load(p), reconstruct=False, minimize=False)
        pa = prepare_assembly(p, reconstruct=False, minimize=False)
        assert pa.report.n_heavy_clashes > 2 * asu.n_heavy_clashes
        # And the assembly heavy-atom count is ~2x the ASU (two copies).
        assert pa.report.n_heavy_atoms == pytest.approx(2 * asu.n_heavy_atoms, rel=0.01)

    def test_clash_residues_span_both_copies(self):
        # Interface clashes implicate residues in BOTH chains -> attribution spans
        # the whole assembly (not just one copy).
        pa = prepare_assembly(_corpus("1doe"), reconstruct=False, minimize=False)
        s = pa.structure
        idx_chain = {}
        i = 0
        for ch in s.models[0].chains:
            for _r in ch.residues:
                idx_chain[i] = ch.id
                i += 1
        chains_hit = {idx_chain[k] for k in pa.report.clash_residue_indices if k in idx_chain}
        assert chains_hit == {"A", "B"}

    def test_materialization_drop_reason_propagates(self):
        # A structure with no REMARK 350 -> the build drop reason passes through,
        # never a crash.
        assert prepare_assembly(os.path.join(PDBS, "1ubq.pdb")) == "no_assembly_metadata"

    def test_too_large_propagates(self):
        # 1mva (180 copies) -> too large for PDB, propagated from build.
        assert prepare_assembly(_corpus("1mva")) == "assembly_too_large_for_pdb"
