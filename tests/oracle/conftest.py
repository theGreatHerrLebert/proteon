"""Shared fixtures and helpers for oracle tests.

Oracle tests compare proteon output against Biopython and Gemmi
to validate correctness of I/O, hierarchy, and metadata extraction.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# PDB file paths
# ---------------------------------------------------------------------------

PDBTBX_EXAMPLES = os.path.join(
    os.path.dirname(__file__), "..", "..", "test-pdbs"
)
TEST_PDBS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "test-pdbs"
)

# Standard test set: files where all three tools agree on atom/residue counts
STANDARD_FILES = {
    "1ubq": os.path.join(PDBTBX_EXAMPLES, "1ubq.pdb"),       # small protein, 1 chain
    "1ubq_cif": os.path.join(PDBTBX_EXAMPLES, "1ubq.cif"),   # same as mmCIF
    "4hhb": os.path.join(TEST_PDBS, "4hhb.pdb"),              # 4-chain hemoglobin
    "1crn": os.path.join(TEST_PDBS, "1crn.pdb"),              # small protein (crambin)
}

# Edge-case files: tested separately (tools may disagree on counts)
EDGE_CASE_FILES = {
    "models": os.path.join(PDBTBX_EXAMPLES, "models.pdb"),    # multi-model NMR
    "insertion": os.path.join(PDBTBX_EXAMPLES, "insertion_codes.pdb"),  # insertion codes + alt conformers
}

# Combined for backward compat
TEST_FILES = {**STANDARD_FILES, **EDGE_CASE_FILES}


def available_files():
    """Return list of (name, path) for standard files that exist."""
    return [(name, path) for name, path in STANDARD_FILES.items() if os.path.exists(path)]


def available_edge_cases():
    """Return list of (name, path) for edge-case files that exist."""
    return [(name, path) for name, path in EDGE_CASE_FILES.items() if os.path.exists(path)]


# ---------------------------------------------------------------------------
# Unified extraction dataclass
# ---------------------------------------------------------------------------


@dataclass
class AtomRecord:
    name: str
    element: str
    x: float
    y: float
    z: float
    b_factor: float
    occupancy: float
    residue_name: str
    chain_id: str
    residue_serial: int
    hetero: bool


@dataclass
class StructureSummary:
    """Comparable summary extracted from any tool."""
    source: str  # "proteon", "biopython", "gemmi"
    model_count: int
    chain_count: int
    residue_count: int
    atom_count: int
    chain_ids: List[str]
    atoms: List[AtomRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def extract_proteon(path: str) -> StructureSummary:
    from proteon_connector import py_io
    pdb = py_io.load(path)
    atoms = []
    for a in pdb.atoms:
        atoms.append(AtomRecord(
            name=a.name.strip(),
            element=a.element or "?",
            x=a.x, y=a.y, z=a.z,
            b_factor=a.b_factor,
            occupancy=a.occupancy,
            residue_name=a.residue_name.strip(),
            chain_id=a.chain_id.strip(),
            residue_serial=a.residue_serial_number,
            hetero=a.hetero,
        ))
    return StructureSummary(
        source="proteon",
        model_count=pdb.model_count,
        chain_count=pdb.chain_count,
        residue_count=pdb.residue_count,
        atom_count=pdb.atom_count,
        chain_ids=[c.id for c in pdb.chains],
        atoms=atoms,
    )


def extract_gemmi(path: str) -> StructureSummary:
    import gemmi
    st = gemmi.read_structure(path)
    model = st[0]
    chain_ids = [ch.name for ch in model]
    residue_count = sum(len(ch) for ch in model)

    atoms = []
    for ch in model:
        for res in ch:
            for atom in res:
                atoms.append(AtomRecord(
                    name=atom.name.strip(),
                    element=atom.element.name,
                    x=atom.pos.x, y=atom.pos.y, z=atom.pos.z,
                    b_factor=atom.b_iso,
                    occupancy=atom.occ,
                    residue_name=res.name.strip(),
                    chain_id=ch.name.strip(),
                    residue_serial=res.seqid.num,
                    hetero=res.het_flag == "H",
                ))

    return StructureSummary(
        source="gemmi",
        model_count=len(st),
        chain_count=len(model),
        residue_count=residue_count,
        atom_count=len(atoms),
        chain_ids=chain_ids,
        atoms=atoms,
    )


def extract_biopython(path: str) -> StructureSummary:
    import Bio.PDB
    if path.endswith(".cif"):
        parser = Bio.PDB.MMCIFParser(QUIET=True)
    else:
        parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", path)
    models = list(structure.get_models())
    model = models[0]
    chains = list(model.get_chains())
    residues = list(model.get_residues())
    bio_atoms = list(model.get_atoms())

    atoms = []
    for chain in chains:
        for residue in chain.get_residues():
            hetflag = residue.get_id()[0]
            is_het = hetflag.strip() not in ("", "W")  # W = water
            for atom in residue.get_atoms():
                v = atom.get_vector()
                atoms.append(AtomRecord(
                    name=atom.get_name().strip(),
                    element=(atom.element or "?").strip(),
                    x=float(v[0]), y=float(v[1]), z=float(v[2]),
                    b_factor=atom.get_bfactor(),
                    occupancy=atom.get_occupancy(),
                    residue_name=residue.get_resname().strip(),
                    chain_id=chain.get_id().strip(),
                    residue_serial=residue.get_id()[1],
                    hetero=is_het,
                ))

    return StructureSummary(
        source="biopython",
        model_count=len(models),
        chain_count=len(chains),
        residue_count=len(residues),
        atom_count=len(atoms),
        chain_ids=[c.get_id() for c in chains],
        atoms=atoms,
    )
