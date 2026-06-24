"""Export built assemblies as label-safe interface supervision (PR3)."""

import os

import numpy as np
import pytest

import proteon
from proteon.assembly_builder import build_assembly_supervision_examples
from proteon.supervision import ComplexSupervisionExamples

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "validation", "pdbs_10k")
PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")


def _corpus(name):
    p = os.path.join(CORPUS, f"{name}.pdb")
    if not os.path.exists(p):
        pytest.skip(f"validation corpus missing: {name}")
    return p


class TestAssemblySupervision:
    def test_expansion_bucket_recovered_as_complex(self):
        # 1doe is `requires_assembly_expansion` (the deposited ASU is a monomer);
        # building the assembly recovers it as a verified 2-chain interface complex.
        out = build_assembly_supervision_examples(_corpus("1doe"), min_coverage=0.5)
        assert isinstance(out, ComplexSupervisionExamples)
        assert out.chain_order == ["A", "B"]
        assert out.assembly_is_asu is True            # verified by construction
        assert out.coverage.n_protein_chains == 2

    def test_chain_examples_masked_and_identity_intact(self):
        out = build_assembly_supervision_examples(_corpus("1var"), min_coverage=0.5)
        assert isinstance(out, ComplexSupervisionExamples)
        ex = out.chain_examples[out.chain_order[0]]
        # 1var assembly has severe clashes -> some residues masked; identity intact.
        assert (ex.all_atom_mask.sum(axis=1) == 0).sum() > 0
        assert ex.seq_mask.sum() == ex.length

    def test_high_floor_drops_pervasive_assembly(self):
        # 1var min chain coverage ~0.61 -> dropped at floor 0.8.
        out = build_assembly_supervision_examples(_corpus("1var"), min_coverage=0.8)
        assert out == "below_coverage_floor"

    def test_single_chain_assembly_not_a_complex(self):
        # 1crn is identity-only single chain -> builds 1 chain -> not a complex.
        out = build_assembly_supervision_examples(os.path.join(PDBS, "1crn.pdb"), min_coverage=0.5)
        assert out == "not_a_complex"

    def test_materialization_drop_reason_propagates(self):
        # Only genuine "can't build" reasons propagate now (size builds as mmCIF).
        assert build_assembly_supervision_examples(
            os.path.join(PDBS, "1ubq.pdb"), min_coverage=0.5) == "no_assembly_metadata"

    def test_record_id_is_source_derived(self):
        # The temp PDB the assembly reloads from is coordinate-only, so the record
        # id must come from the SOURCE path, not collide across inputs (codex).
        out = build_assembly_supervision_examples(_corpus("1doe"), min_coverage=0.5)
        assert isinstance(out, ComplexSupervisionExamples)
        assert out.record_id == "1doe:assembly"

    def test_multi_model_source_still_drops(self):
        # A multi-model source must NOT be silently exported from model 1: the
        # source model count is carried forward so `multiple_models` still gates
        # (codex). Build a 2-model wrapper around 1crn (which has REMARK 350).
        lines = open(os.path.join(PDBS, "1crn.pdb")).read().splitlines()
        head = [ln for ln in lines if not ln.startswith(("ATOM", "HETATM", "TER", "END"))]
        atoms = [ln for ln in lines if ln.startswith(("ATOM", "HETATM"))]
        text = "\n".join(
            head
            + ["MODEL        1"] + atoms + ["ENDMDL"]
            + ["MODEL        2"] + atoms + ["ENDMDL"]
            + ["END"]
        )
        out = build_assembly_supervision_examples(text, min_coverage=0.5)
        assert out == "unmasked_hazard:multiple_models"

    def test_reconstruct_minimize_default_off(self):
        # The label-safe path: no fabricated atoms (reconstruct=False) by default,
        # so an assembly with missing atoms isn't silently completed.
        out = build_assembly_supervision_examples(_corpus("1doe"), min_coverage=0.5)
        assert isinstance(out, ComplexSupervisionExamples)
        assert out.prep_report.atoms_reconstructed == 0
