"""REMARK 350 biological-assembly parsing + the three-state assembly_is_asu."""

import os

import pytest

import proteon
from proteon.assembly import assembly_metadata, parse_remark_350

PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")


def _text(name):
    return open(os.path.join(PDBS, f"{name}.pdb")).read()


def _chains(name):
    return sorted(set(proteon.load(os.path.join(PDBS, f"{name}.pdb")).chain_ids))


class TestParse:
    def test_parses_biomolecule_chains_and_operators(self):
        bios = parse_remark_350(_text("4hhb"))
        assert len(bios) == 1
        assert bios[0].chains == ["A", "B", "C", "D"]
        assert len(bios[0].operators) == 1
        assert bios[0].all_identity()

    def test_absent_remark_350(self):
        assert parse_remark_350(_text("1ubq")) == []

    def test_comma_separated_biomolecule_id_does_not_crash(self):
        # "BIOMOLECULE: 1, 2" must parse (first id), not raise ValueError.
        text = (
            "REMARK 350 BIOMOLECULE: 1, 2\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
        )
        bios = parse_remark_350(text)
        assert bios[0].id == 1
        copies, is_asu = assembly_metadata(text, ["A"])
        assert copies == 1
        assert is_asu is True

    def test_null_chain_normalized_to_blank(self):
        # "CHAINS: NULL" encodes a blank chain id; the loader exposes "" — they
        # must compare equal so an identity single-chain ASU is is_asu True.
        text = (
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: NULL\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
        )
        assert parse_remark_350(text)[0].chains == [""]
        _, is_asu = assembly_metadata(text, [" "])  # loader blank chain
        assert is_asu is True

    def test_identity_detection(self):
        from proteon.assembly import Biomolecule
        b = Biomolecule(id=1, chains=["A"], operators=[[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]])
        assert b.all_identity()
        b.operators.append([[0.0, -1.0, 0, 5.0], [1.0, 0, 0, 0], [0, 0, 1.0, 0]])  # a rotation
        assert not b.all_identity()


class TestThreeState:
    def test_asu_is_assembly_tetramer(self):
        # 4hhb: the deposited 4 chains ARE the biological tetramer (identity BIOMT).
        copies, is_asu = assembly_metadata(_text("4hhb"), _chains("4hhb"))
        assert copies == 1
        assert is_asu is True

    def test_asu_is_assembly_monomer(self):
        copies, is_asu = assembly_metadata(_text("1crn"), _chains("1crn"))
        assert is_asu is True

    def test_no_metadata_is_none(self):
        # 1ubq has no REMARK 350 -> no evidence, NOT an assumption of monomer.
        copies, is_asu = assembly_metadata(_text("1ubq"), _chains("1ubq"))
        assert copies is None
        assert is_asu is None

    def test_multiple_biomolecules_is_false(self):
        # 1ake: ASU has chains A,B but REMARK 350 defines them as TWO separate
        # monomeric assemblies -> the ASU is NOT a single biological unit, so an
        # A-B "interface" would be a crystal artifact. is_asu must be False.
        copies, is_asu = assembly_metadata(_text("1ake"), _chains("1ake"))
        assert is_asu is False

    def test_chain_mismatch_is_false(self):
        # If the structure's present chains differ from the assembly's chain list,
        # we cannot claim the ASU is the assembly.
        copies, is_asu = assembly_metadata(_text("4hhb"), ["A", "B"])  # only 2 of 4
        assert is_asu is False


class TestInterfaceProfile:
    def test_label_safe_interface_requires_verified_assembly(self):
        from proteon import PrepReport
        clean = dict(hydrogens_added=50, minimizer_status="converged_gradient")
        # verified assembly + sane coords -> interface safe
        assert PrepReport(assembly_is_asu=True, **clean).label_safe_interface is True
        # not the assembly -> hazard + unsafe
        bad = PrepReport(assembly_is_asu=False, **clean)
        assert bad.label_safe_interface is False
        assert "assembly_mismatch" in bad.label_hazards
        # not determined -> conservatively unsafe, but NOT flagged as a hazard
        unknown = PrepReport(assembly_is_asu=None, **clean)
        assert unknown.label_safe_interface is False
        assert "assembly_mismatch" not in unknown.label_hazards

    def test_clean_report_has_no_assembly_hazard(self):
        # The common prepare() path leaves assembly_is_asu None -> must not
        # pollute label_hazards (the strict label_safe gate is unaffected).
        from proteon import PrepReport
        r = PrepReport(hydrogens_added=50, minimizer_status="converged_gradient")
        assert r.label_hazards == []
        assert r.label_safe is True


class TestIntegration:
    def test_monomer_and_tetramer_assembly_resolved(self):
        res = {os.path.basename(r.path): r for r in
               proteon.prepare_for_supervision(
                   [os.path.join(PDBS, f"{n}.pdb") for n in ("1crn", "4hhb", "1ake", "1ubq")])}
        assert res["1crn.pdb"].report.assembly_is_asu is True
        assert res["1crn.pdb"].report.label_safe_interface is True   # verified + clean
        assert res["4hhb.pdb"].report.assembly_is_asu is True        # tetramer is the assembly
        assert res["1ake.pdb"].report.assembly_is_asu is False       # two separate monomers
        assert "assembly_mismatch" in res["1ake.pdb"].report.label_hazards
        assert res["1ubq.pdb"].report.assembly_is_asu is None        # no REMARK 350

    def test_repair_policy_can_drop_on_assembly_mismatch(self):
        # assembly_mismatch is a recognised policy hazard: a policy can drop the
        # structures whose ASU is not the biological assembly. Annotation happens
        # before the policy evaluation, so the rule is actually applied.
        pol = proteon.RepairPolicy.for_profile(
            "heavy_coords", assembly_mismatch="drop",
            missing_atoms="reconstruct", reconstructed_atoms="accept",
            altlocs="accept_selected", multiple_models="accept_selected",
            heavy_clashes="accept",
        )
        res = {os.path.basename(r.path): r for r in proteon.prepare_for_supervision(
            [os.path.join(PDBS, f"{n}.pdb") for n in ("1crn", "1ake")], repair=pol)}
        assert res["1ake.pdb"].passes_policy is False
        assert "assembly_mismatch" in res["1ake.pdb"].repair.dropped_for
        assert res["1crn.pdb"].passes_policy is True

    def test_assembly_not_set_without_path(self):
        # prepare() (no path) cannot determine the assembly -> stays None.
        r = proteon.prepare(proteon.load(os.path.join(PDBS, "4hhb.pdb")))
        assert r.assembly_is_asu is None
        assert r.label_safe_interface is False
