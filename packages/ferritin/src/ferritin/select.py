"""Atom selection language for structural queries.

Supports MDAnalysis/PyMOL-style selection strings with boolean logic.

    mask = ferritin.select(structure, "CA and chain A")
    coords = structure.coords[mask]

Syntax:
    name CA              — atom name (case sensitive)
    chain A              — chain ID
    resname ALA          — residue name
    resid 10             — residue number
    resid 10-50          — residue range
    element C            — element symbol
    backbone             — backbone atoms (N, CA, C, O)
    sidechain            — non-backbone heavy atoms
    protein              — amino acid residues
    hetero               — HETATM records
    water                — water molecules (HOH, WAT)
    hydrogen             — hydrogen atoms
    heavy                — non-hydrogen atoms
    all                  — all atoms
    not <expr>           — negation
    <expr> and <expr>    — intersection
    <expr> or <expr>     — union
    (<expr>)             — grouping

Examples:
    "CA"                           → all CA atoms
    "backbone and chain A"         → backbone of chain A
    "resid 1-50 and not hydrogen"  → residues 1-50, heavy atoms only
    "resname ALA or resname GLY"   → all alanine and glycine residues
    "(chain A or chain B) and CA"  → CA atoms of chains A and B
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_KEYWORDS = {
    "name", "chain", "resname", "resid", "element",
    "backbone", "sidechain", "protein", "hetero", "water",
    "hydrogen", "heavy", "all",
    "and", "or", "not",
}

_BACKBONE_NAMES = {"N", "CA", "C", "O"}
_WATER_NAMES = {"HOH", "WAT", "H2O", "TIP", "TIP3", "SPC"}


def _tokenize(query: str) -> List[str]:
    """Split selection string into tokens."""
    tokens = []
    i = 0
    while i < len(query):
        c = query[i]
        if c.isspace():
            i += 1
        elif c in '()':
            tokens.append(c)
            i += 1
        elif c == '-' and tokens and tokens[-1][-1].isdigit():
            # Part of a range like "10-50"
            # Attach to previous token
            j = i + 1
            while j < len(query) and (query[j].isdigit()):
                j += 1
            tokens[-1] += query[i:j]
            i = j
        else:
            j = i
            while j < len(query) and not query[j].isspace() and query[j] not in '()':
                j += 1
            tokens.append(query[i:j])
            i = j
    return tokens


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

class _Parser:
    """Recursive descent parser for selection expressions."""

    def __init__(self, tokens: List[str], atom_data: dict):
        self.tokens = tokens
        self.pos = 0
        self.data = atom_data
        self.n = atom_data["n"]

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, expected: str):
        tok = self.consume()
        if tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}'")

    def parse(self) -> NDArray[np.bool_]:
        result = self.parse_or()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        return result

    def parse_or(self) -> NDArray[np.bool_]:
        left = self.parse_and()
        while self.peek() and self.peek().lower() == "or":
            self.consume()
            right = self.parse_and()
            left = left | right
        return left

    def parse_and(self) -> NDArray[np.bool_]:
        left = self.parse_not()
        while self.peek() and self.peek().lower() == "and":
            self.consume()
            right = self.parse_not()
            left = left & right
        return left

    def parse_not(self) -> NDArray[np.bool_]:
        if self.peek() and self.peek().lower() == "not":
            self.consume()
            operand = self.parse_not()
            return ~operand
        return self.parse_atom()

    def parse_atom(self) -> NDArray[np.bool_]:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        # Grouping
        if tok == "(":
            self.consume()
            result = self.parse_or()
            self.expect(")")
            return result

        self.consume()
        tok_lower = tok.lower()

        # Keywords with arguments
        if tok_lower == "name":
            val = self.consume()
            return np.array([n.strip() == val for n in self.data["atom_names"]])

        if tok_lower == "chain":
            val = self.consume()
            return np.array([c == val for c in self.data["chain_ids"]])

        if tok_lower == "resname":
            val = self.consume().upper()
            return np.array([r == val for r in self.data["residue_names"]])

        if tok_lower == "resid":
            val = self.consume()
            if "-" in val:
                parts = val.split("-")
                lo, hi = int(parts[0]), int(parts[1])
                resnums = self.data["residue_numbers"]
                return (resnums >= lo) & (resnums <= hi)
            else:
                num = int(val)
                return self.data["residue_numbers"] == num

        if tok_lower == "element":
            val = self.consume()
            return np.array([e.strip() == val for e in self.data["elements"]])

        # Standalone keywords
        if tok_lower == "backbone":
            return np.array([n.strip() in _BACKBONE_NAMES for n in self.data["atom_names"]])

        if tok_lower == "sidechain":
            bb = np.array([n.strip() in _BACKBONE_NAMES for n in self.data["atom_names"]])
            h = np.array([e.strip() == "H" for e in self.data["elements"]])
            return ~bb & ~h

        if tok_lower == "protein":
            return np.array([r in _AMINO_ACIDS for r in self.data["residue_names"]])

        if tok_lower == "hetero":
            return np.array([r not in _AMINO_ACIDS and r not in _WATER_NAMES
                           for r in self.data["residue_names"]])

        if tok_lower == "water":
            return np.array([r in _WATER_NAMES for r in self.data["residue_names"]])

        if tok_lower == "hydrogen":
            return np.array([e.strip() in ("H", "D") for e in self.data["elements"]])

        if tok_lower == "heavy":
            return np.array([e.strip() not in ("H", "D") for e in self.data["elements"]])

        if tok_lower == "all":
            return np.ones(self.n, dtype=bool)

        # If it's not a keyword, treat it as an atom name (shorthand)
        # e.g., "CA" is equivalent to "name CA"
        return np.array([n.strip() == tok for n in self.data["atom_names"]])


_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",  # non-standard but common
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select(structure, query: str) -> NDArray[np.bool_]:
    """Select atoms using a query string.

    Returns a boolean mask that can index structure.coords,
    structure.b_factors, and other per-atom arrays.

    Args:
        structure: A ferritin Structure.
        query: Selection string (see module docstring for syntax).

    Returns:
        Boolean mask of length structure.atom_count.

    Examples:
        >>> mask = ferritin.select(structure, "CA and chain A")
        >>> ca_coords = structure.coords[mask]
        >>> print(f"{mask.sum()} atoms selected")

        >>> mask = ferritin.select(structure, "backbone and resid 1-50")
        >>> bb_bfactors = structure.b_factors[mask]

        >>> mask = ferritin.select(structure, "protein and heavy")
        >>> heavy_coords = structure.coords[mask]

    Agent Notes:
        OUTPUT: Boolean numpy mask of length structure.atom_count.
        Use it to index ANY per-atom array: coords, b_factors,
        occupancies, atom_names, elements, etc.

        SYNTAX: Atom names are CASE SENSITIVE ("CA" != "ca").
        Keywords (and, or, not, backbone, etc.) are case insensitive.

        SHORTHAND: Bare atom names work without "name" prefix:
        "CA" is equivalent to "name CA".

        COMMON PATTERNS:
            "CA"                     — alpha carbons only
            "backbone"               — N, CA, C, O atoms
            "protein and heavy"      — amino acid heavy atoms (for ML features)
            "chain A and resid 1-100" — region selection
            "not water and not hydrogen" — clean structure

        COMBINE: Use the mask with numpy indexing:
            structure.coords[mask]      — filtered coordinates
            structure.b_factors[mask]   — filtered B-factors
            np.array(structure.atom_names)[mask]  — filtered names
    """
    atom_data = {
        "n": structure.atom_count,
        "atom_names": structure.atom_names,
        "chain_ids": structure.chain_ids,
        "residue_names": structure.residue_names,
        "residue_numbers": np.asarray(structure.residue_serial_numbers),
        "elements": structure.elements,
    }

    tokens = _tokenize(query)
    if not tokens:
        return np.ones(structure.atom_count, dtype=bool)

    parser = _Parser(tokens, atom_data)
    return parser.parse()
