"""Tests for `PreparedStructureRecord` sequence-length semantics.

The prepared-structure manifest is the Layer-3 artifact downstream
releases key off of; if `sequence_length` overreports on structures
with hetero content, every release summary carrying it skews high.
Specifically pins the fix for the 2026-04-14 review finding that
`_sequence_length` was reading `structure.residue_count` (which is
documented to include waters and ligands) when the attribute existed.
"""

from __future__ import annotations

from types import SimpleNamespace

import proteon


def _residue(name: str, serial: int, *, is_amino_acid: bool):
    return SimpleNamespace(
        name=name,
        serial_number=serial,
        is_amino_acid=is_amino_acid,
        atoms=[],
    )


def _structure_with_hetero(identifier: str = "protein_with_ligand"):
    """2 amino-acid residues + a water + a glycerol ligand.

    residue_count=4 (AA + HOH + GOL + HOH) but sequence_length must
    report 2 because only two of them are amino acids.
    """
    chain_a = SimpleNamespace(
        id="A",
        residues=[
            _residue("ALA", 1, is_amino_acid=True),
            _residue("GLY", 2, is_amino_acid=True),
            _residue("GOL", 3, is_amino_acid=False),  # glycerol ligand
        ],
    )
    chain_w = SimpleNamespace(
        id="W",
        residues=[
            _residue("HOH", 1, is_amino_acid=False),  # water
        ],
    )
    return SimpleNamespace(
        identifier=identifier,
        chain_count=2,
        chains=[chain_a, chain_w],
        # residue_count includes waters + ligands per io.py contract.
        residue_count=4,
        atom_count=10,
    )


def _structure_no_hetero():
    chain = SimpleNamespace(
        id="A",
        residues=[
            _residue("ALA", 1, is_amino_acid=True),
            _residue("GLY", 2, is_amino_acid=True),
            _residue("SER", 3, is_amino_acid=True),
        ],
    )
    return SimpleNamespace(
        identifier="all_aa",
        chain_count=1,
        chains=[chain],
        residue_count=3,
        atom_count=12,
    )


class TestPreparedStructureRecordSequenceLength:
    def test_sequence_length_excludes_waters_and_ligands(self):
        """Regression for 2026-04-14 review finding: _sequence_length
        previously read `residue_count` (which includes HETATMs) and
        overreported. Must count only amino-acid residues."""
        records = proteon.build_prepared_structure_records(
            [_structure_with_hetero()],
            prep_reports=[proteon.PrepReport(hydrogens_added=0, converged=True)],
        )
        assert len(records) == 1
        assert records[0].sequence_length == 2, (
            "sequence_length must count amino acids only, "
            "not the full residue_count that includes HETATMs"
        )

    def test_sequence_length_matches_residue_count_when_no_hetero(self):
        """When every residue is an amino acid, the two values
        coincide. Pins that the fix didn't break the common case."""
        records = proteon.build_prepared_structure_records(
            [_structure_no_hetero()],
            prep_reports=[proteon.PrepReport(hydrogens_added=0, converged=True)],
        )
        assert records[0].sequence_length == 3
