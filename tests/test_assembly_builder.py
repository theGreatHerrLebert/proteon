"""BIOMT assembly materialization (PR1): apply operators, provenance, collisions."""

import os

import numpy as np
import pytest

import proteon
from proteon.assembly_builder import BuiltAssembly, build_assembly

# Real expanders from the validation corpus (skip if that corpus isn't present).
CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "validation", "pdbs_10k")
PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")


def _corpus(name):
    p = os.path.join(CORPUS, f"{name}.pdb")
    if not os.path.exists(p):
        pytest.skip(f"validation corpus missing: {name}")
    return p


class TestBuild:
    def test_dimer_from_two_operators(self):
        # 1doe: 2 operators on chain A -> a 2-chain dimer.
        b = build_assembly(_corpus("1doe"))
        assert isinstance(b, BuiltAssembly)
        assert b.n_operators == 2
        assert b.n_source_chains == 1
        assert b.n_chains == 2
        ids = [c.assembled_chain_id for c in b.chains]
        assert ids == ["A", "B"]                      # deterministic
        assert b.chains[0].is_identity and not b.chains[1].is_identity
        assert all(c.source_chain_id == "A" for c in b.chains)

    def test_built_assembly_loads(self):
        b = build_assembly(_corpus("1doe"))
        s = b.load()
        assert sorted(set(s.chain_ids)) == ["A", "B"]
        # twice the ASU atom count (two copies of chain A).
        asu = proteon.load(_corpus("1doe"))
        assert len(s.coords) == 2 * len(asu.coords)

    def test_identity_copy_reproduces_asu_coordinates(self):
        b = build_assembly(_corpus("1doe"))
        asu = proteon.load(_corpus("1doe"))
        built = b.load()
        # chain A of the built assembly == the ASU exactly (identity operator).
        a_built = built.coords[: len(asu.coords)]
        assert np.allclose(a_built, asu.coords, atol=1e-2)

    def test_emitted_atom_serials_are_unique(self):
        # Each copy must get fresh serials -> a valid, re-loadable PDB (codex).
        b = build_assembly(_corpus("1doe"))
        serials = [int(l[6:11]) for l in b.pdb_text.splitlines()
                   if l[:6] in ("ATOM  ", "HETATM")]
        assert len(serials) == len(set(serials))           # all unique
        assert serials == list(range(1, len(serials) + 1))  # sequential from 1

    def test_expansion_forms_an_interface(self):
        # The non-identity copy must come into contact with the ASU copy (a real
        # biological interface), not land far away or overlap exactly. Min
        # inter-set distance via NumPy (no SciPy test dependency, codex).
        b = build_assembly(_corpus("1doe"))
        s = b.load()
        n = len(s.coords) // 2
        a, copy = s.coords[:n], s.coords[n:]
        mind = np.inf
        for chunk in np.array_split(copy, max(1, len(copy) // 200)):
            d2 = ((chunk[:, None, :] - a[None, :, :]) ** 2).sum(-1)
            mind = min(mind, float(np.sqrt(d2.min())))
        assert 0.5 < mind < 5.0  # an interface, not coincident, not detached


class TestEdgeCases:
    def test_too_many_chains_for_pdb(self):
        # 1mva: 60 operators x 3 chains = 180 > 62 legal PDB chain ids.
        assert build_assembly(_corpus("1mva")) == "assembly_too_large_for_pdb"

    def test_capsid_too_large_by_atom_count(self):
        # 1z14: 60 copies x ~6000 atoms = ~360k > 99999 PDB serial limit -> the
        # atom-count overflow catches it even though 60 chains fit the alphabet.
        assert build_assembly(_corpus("1z14")) == "assembly_too_large_for_pdb"

    def test_blank_identity_plus_full_alphabet_fits(self):
        # Blank-chain identity (preserved as ' ') + 62 expansions = 63 chains, all
        # representable (' ' + the 62-char alphabet). Must build, NOT overflow
        # (codex: capacity is the alphabet PLUS the preserved blank).
        ops = ("REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
               "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
               "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n")  # identity
        ops += "".join(
            f"REMARK 350   BIOMT1{k:4d}  1.000000  0.000000  0.000000  {k * 3.0:9.5f}\n"
            f"REMARK 350   BIOMT2{k:4d}  0.000000  1.000000  0.000000        0.00000\n"
            f"REMARK 350   BIOMT3{k:4d}  0.000000  0.000000  1.000000        0.00000\n"
            for k in range(2, 64)  # 62 distinct expansions
        )
        text = ("REMARK 350 BIOMOLECULE: 1\n"
                "REMARK 350 APPLY THE FOLLOWING TO CHAINS: NULL\n" + ops
                + "ATOM      1  CA  ALA     1       0.000   0.000   0.000  1.00  0.00           C\n")
        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        assert b.n_chains == 63
        assert sum(1 for c in b.chains if c.assembled_chain_id == " ") == 1  # blank preserved

    def test_chain_count_overflow(self):
        # A synthetic 63-chain-copy case: chain-id alphabet overflow (>62), even
        # with few atoms. Build a 1-atom chain replicated by 63 translations.
        ops = "".join(
            f"REMARK 350   BIOMT1{k:4d}  1.000000  0.000000  0.000000  {k * 5.0:9.5f}\n"
            f"REMARK 350   BIOMT2{k:4d}  0.000000  1.000000  0.000000        0.00000\n"
            f"REMARK 350   BIOMT3{k:4d}  0.000000  0.000000  1.000000        0.00000\n"
            for k in range(1, 64)
        )
        text = ("REMARK 350 BIOMOLECULE: 1\n"
                "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n" + ops
                + "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        assert build_assembly(text) == "assembly_too_large_for_pdb"

    def test_no_remark_350(self):
        assert build_assembly(os.path.join(PDBS, "1ubq.pdb")) == "no_assembly_metadata"

    def test_identity_only_reproduces_asu(self):
        # 1crn is already the assembly (single identity operator) -> 1 chain.
        b = build_assembly(os.path.join(PDBS, "1crn.pdb"))
        assert isinstance(b, BuiltAssembly)
        assert b.n_chains == 1 and b.chains[0].is_identity

    def test_identity_copy_preserves_source_chain_id(self):
        # An identity copy keeps the DEPOSITED chain id (here 'X'), not 'A' (codex)
        # — so an identity-only assembly reproduces the ASU's chain ids.
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: X\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000       20.00000\n"
            "REMARK 350   BIOMT2   2  0.000000 -1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA X   1       0.000   0.000   0.000  1.00  0.00           C\n"
        )
        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        ident = next(c for c in b.chains if c.is_identity)
        assert ident.assembled_chain_id == "X"          # preserved, not 'A'
        expansion = next(c for c in b.chains if not c.is_identity)
        assert expansion.assembled_chain_id != "X"       # synthetic, distinct

    def test_biomolecule_not_found(self):
        assert build_assembly(_corpus("1doe"), biomolecule=99) == "biomolecule_not_found"


class TestChainSpecificBlocks:
    def test_operators_apply_only_to_their_block_chains(self):
        # Two blocks: chain A gets 2 operators (identity + a rotation), chain B
        # gets 1 (identity). All-to-all would give 4 copies; correct is 3.
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   2  0.000000 -1.000000  0.000000       10.00000\n"
            "REMARK 350   BIOMT2   2  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: B\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  GLY B   1       5.000   0.000   0.000  1.00  0.00           C\n"
        )
        from proteon.assembly import parse_remark_350
        bio = parse_remark_350(text)[0]
        assert len(bio.blocks) == 2
        assert bio.blocks[0].chains == ["A"] and len(bio.blocks[0].operators) == 2
        assert bio.blocks[1].chains == ["B"] and len(bio.blocks[1].operators) == 1

        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        # A -> 2 copies, B -> 1 copy = 3 chains (NOT 4 from all-to-all).
        srcs = sorted((c.source_chain_id, c.operator_index) for c in b.chains)
        assert srcs == [("A", 0), ("A", 1), ("B", 0)]


class TestCoordinateOverflow:
    def test_transform_outside_pdb_coord_range_rejected(self):
        # A huge translation pushes atoms past the %8.3f field (>9999.999) ->
        # must fail loud, not emit malformed columns (codex).
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000    50000.00000\n"
            "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        )
        assert build_assembly(text) == "assembly_coords_exceed_pdb"


class TestChainIdReservation:
    def test_expansion_before_identity_does_not_steal_id(self):
        # Block for chain B (expansion) precedes chain A's identity. The synthetic
        # copy must NOT grab 'A' — A's identity reserves it (codex two-pass).
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: B\n"
            "REMARK 350   BIOMT1   1 -1.000000  0.000000  0.000000       30.00000\n"
            "REMARK 350   BIOMT2   1  0.000000 -1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  GLY B   1       5.000   0.000   0.000  1.00  0.00           C\n"
        )
        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        a_ident = next(c for c in b.chains if c.source_chain_id == "A" and c.is_identity)
        assert a_ident.assembled_chain_id == "A"   # reserved despite B emitted first
        # B's expansion copy got a synthetic id that is NOT 'A'.
        b_exp = next(c for c in b.chains if c.source_chain_id == "B")
        assert b_exp.assembled_chain_id != "A"


class TestBlankChain:
    def test_blank_chain_identity_preserved_as_space(self):
        # REMARK 350 CHAINS: NULL = a blank chain id; the identity copy must keep
        # it (space in col 22), not get a synthetic 'A' (codex).
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: NULL\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA     1       0.000   0.000   0.000  1.00  0.00           C\n"
        )
        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        assert b.chains[0].is_identity
        assert b.chains[0].assembled_chain_id == " "   # preserved blank, not 'A'


class TestIdentityRepresentative:
    def test_identity_listed_second_still_preserved_and_dedup_holds(self):
        # Identity operator listed AFTER a rotation. It must still be the kept
        # representative (preserve src id), and a non-identity op that coincides
        # with the ASU must be dropped, not duplicated (codex identity-first sort).
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  0.000000 -1.000000  0.000000       40.00000\n"
            "REMARK 350   BIOMT2   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   3  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   3  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   3  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        )
        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        # op1 (rotation) + op2 (identity) + op3 (identity dup) -> 2 chains: the
        # identity (dedup collapses op2/op3) preserves 'A', the rotation expands.
        assert b.n_chains == 2
        ident = [c for c in b.chains if c.is_identity]
        assert len(ident) == 1 and ident[0].assembled_chain_id == "A"


class TestCrossChainOverlap:
    def test_operator_mapping_onto_another_chain_is_skipped(self):
        # Chains A (at x=0) and B (at x=10), identical 1-atom chains. Block applies
        # to A,B: identity + a +10 translation. The +10 copy of A lands exactly on
        # B's deposited position -> must be skipped as coincident across chains
        # (codex flat-placed dedup), so the copy of A@+10 isn't a duplicate of B.
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000\n"
            "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA B   1      10.000   0.000   0.000  1.00  0.00           C\n"
        )
        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        # Without cross-chain dedup: A_id, B_id, A+10(==B), B+10 = 4. With it: the
        # A+10 copy coincides with B_id and is dropped -> 3 chains.
        from collections import Counter
        coords_x = []
        for ln in b.pdb_text.splitlines():
            if ln[:6] in ("ATOM  ", "HETATM"):
                coords_x.append(round(float(ln[30:38]), 1))
        # positions present: 0 (A_id), 10 (B_id, and A+10 would dup it), 20 (B+10)
        assert b.n_chains == 3
        assert Counter(coords_x) == Counter({0.0: 1, 10.0: 1, 20.0: 1})


class TestCrossBlockIdentityPriority:
    def test_earlier_block_expansion_does_not_drop_later_identity(self):
        # Block 1 (chain A) has an expansion that lands on chain B's deposited
        # position; block 2 (chain B) is identity. Global identity-first placement
        # must keep B's identity (preserving 'B') and drop A's coincident expansion,
        # not the reverse (codex cross-block).
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000\n"
            "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: B\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA B   1      10.000   0.000   0.000  1.00  0.00           C\n"
        )
        b = build_assembly(text)
        assert isinstance(b, BuiltAssembly)
        # A_id@0, B_id@10 kept; A+10@10 coincides with B_id -> dropped. 2 chains.
        assert b.n_chains == 2
        b_chain = next(c for c in b.chains if c.source_chain_id == "B")
        assert b_chain.is_identity and b_chain.assembled_chain_id == "B"  # preserved


class TestOperatorDedup:
    def test_duplicate_operators_collapsed(self):
        from proteon.assembly_builder import _dedup_operators

        ident = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
        rot = [[0.0, -1.0, 0, 5.0], [1.0, 0, 0, 0], [0, 0, 1.0, 0]]
        kept = _dedup_operators([ident, rot, ident, rot])  # two unique
        assert len(kept) == 2
