"""Biological-assembly metadata from PDB ``REMARK 350`` (BIOMT operators).

The asymmetric unit (what you load) is not always the biological assembly. For
interface / contact / SASA / neighbor-graph labels that matters: train on the
wrong oligomeric state and the interfaces are wrong. This module parses the
biological-assembly definition that the loader (pdbtbx) discards, so the
label-safe gate can flag it.

This is a **conservative detector**, NOT an assembly builder. It answers one
question — *does the deposited ASU already equal the biological assembly?* —
with three honest states (see :func:`assembly_metadata`). It does not apply the
transforms or build the oligomer (a deliberate follow-on), and it reads PDB
``REMARK 350`` only (mmCIF ``_pdbx_struct_assembly`` is a follow-on).

Per-chain coordinate / energy / sequence labels are oligomer-invariant, so they
do not depend on any of this; only interface-type labels do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Source tag for the (currently single) metadata backend.
SOURCE = "pdb_remark_350"

_BIOMT_RE = re.compile(
    r"REMARK 350\s+BIOMT([123])\s+(\d+)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)
_IDENTITY_TOL = 1e-4


@dataclass
class Biomolecule:
    """One BIOMOLECULE block: the chains it applies to and its operators."""

    id: int
    chains: List[str] = field(default_factory=list)
    #: Each operator is a 3x4 row-major matrix [[m11..t1],[m21..t2],[m31..t3]].
    operators: List[List[List[float]]] = field(default_factory=list)

    def all_identity(self) -> bool:
        """True iff every operator is the identity transform (no expansion)."""
        ident = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
        for op in self.operators:
            for r in range(3):
                for c in range(4):
                    if abs(op[r][c] - ident[r][c]) > _IDENTITY_TOL:
                        return False
        return True


def parse_remark_350(text: str) -> List[Biomolecule]:
    """Parse the BIOMOLECULE / chain-list / BIOMT operators out of PDB text."""
    bios: List[Biomolecule] = []
    cur: Optional[Biomolecule] = None
    ops: dict = {}  # op_number -> {row: [m,m,m,t]}
    for line in text.splitlines():
        if not line.startswith("REMARK 350"):
            continue
        if "BIOMOLECULE:" in line:
            # The id field is usually a single integer, but some entries list
            # several on one line ("BIOMOLECULE: 1, 2"); take the first, and
            # never let a malformed id abort the parse.
            raw = line.split("BIOMOLECULE:")[1].replace(",", " ").split()
            bio_id = int(raw[0]) if raw and raw[0].lstrip("-").isdigit() else 0
            cur = Biomolecule(id=bio_id)
            bios.append(cur)
            ops = {}
            continue
        if cur is None:
            continue
        if "CHAINS:" in line:
            # "APPLY THE FOLLOWING TO CHAINS: A, B" or a continuation "AND CHAINS: C".
            # A blank chain identifier is encoded as the literal "NULL"; the loader
            # exposes it as an empty id, so normalise it for the chain-set compare.
            chains = line.split("CHAINS:")[1]
            cur.chains.extend(
                ("" if c.strip().upper() == "NULL" else c.strip())
                for c in chains.replace(",", " ").split()
                if c.strip()
            )
            continue
        m = _BIOMT_RE.search(line)
        if m:
            row = int(m.group(1)) - 1
            opnum = int(m.group(2))
            vals = [float(m.group(i)) for i in range(3, 7)]
            ops.setdefault(opnum, {})[row] = vals
            # Once all three rows of an operator are present, materialise it.
            entry = ops[opnum]
            if len(entry) == 3 and all(r in entry for r in (0, 1, 2)):
                op = [entry[0], entry[1], entry[2]]
                if op not in cur.operators:
                    cur.operators.append(op)
    return bios


def assembly_metadata(
    pdb_text: str, present_chains
) -> Tuple[Optional[int], Optional[bool]]:
    """Return ``(biological_assembly_copies, assembly_is_asu)`` for a PDB.

    Three honest states for ``assembly_is_asu`` (conservative — see module docs):

    - ``True``  — REMARK 350 defines a SINGLE biomolecule whose operators are all
      identity and whose chain list equals the present chains exactly: the
      deposited ASU already IS the biological assembly.
    - ``False`` — metadata is present but the ASU is not sufficient/exact:
      operators require expansion, the chain lists don't match (ASU has extras or
      is a subset), or multiple biomolecules are defined (ambiguous).
    - ``None``  — no usable REMARK 350 (no evidence — NOT an assumption of monomer).

    ``copies`` is the number of operators in the first biomolecule (1 = no
    expansion), or ``None`` when there is no metadata.
    """
    bios = parse_remark_350(pdb_text)
    if not bios or not bios[0].operators:
        # No REMARK 350, or a block with no usable BIOMT operators (malformed /
        # unsupported format) -> unknown, NOT vacuously the assembly (codex).
        return (None, None)
    copies = len(bios[0].operators)
    present = {str(c).strip() for c in present_chains}
    is_asu = (
        len(bios) == 1
        and bios[0].all_identity()
        and {c for c in bios[0].chains} == present
        and len(present) > 0
    )
    return (copies, is_asu if is_asu else False)
