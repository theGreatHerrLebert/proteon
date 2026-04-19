"""Pythonic wrappers for PDB/mmCIF structure hierarchy.

Provides: Structure, Model, Chain, Residue, Atom — wrapping the
ferritin_connector PyO3 types with a Pythonic API.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .core import RustWrapperObject

try:
    import ferritin_connector

    _pdb_mod = ferritin_connector.py_pdb
except ImportError:  # pragma: no cover
    _pdb_mod = None


class Atom(RustWrapperObject):
    """A single atom in a structure."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> Atom:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def name(self) -> str:
        return self._ptr.name

    @property
    def serial_number(self) -> int:
        return self._ptr.serial_number

    @property
    def x(self) -> float:
        return self._ptr.x

    @property
    def y(self) -> float:
        return self._ptr.y

    @property
    def z(self) -> float:
        return self._ptr.z

    @property
    def pos(self) -> tuple:
        return self._ptr.pos

    @property
    def element(self) -> Optional[str]:
        return self._ptr.element

    @property
    def b_factor(self) -> float:
        return self._ptr.b_factor

    @property
    def occupancy(self) -> float:
        return self._ptr.occupancy

    @property
    def charge(self) -> int:
        return self._ptr.charge

    @property
    def hetero(self) -> bool:
        return self._ptr.hetero

    @property
    def is_backbone(self) -> bool:
        return self._ptr.is_backbone

    @property
    def residue_name(self) -> str:
        return self._ptr.residue_name

    @property
    def chain_id(self) -> str:
        return self._ptr.chain_id

    @property
    def residue_serial_number(self) -> int:
        return self._ptr.residue_serial_number

    def __repr__(self) -> str:
        return repr(self._ptr)


class Residue(RustWrapperObject):
    """A residue (amino acid, nucleotide, or ligand)."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> Residue:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def name(self) -> Optional[str]:
        return self._ptr.name

    @property
    def serial_number(self) -> int:
        return self._ptr.serial_number

    @property
    def insertion_code(self) -> Optional[str]:
        return self._ptr.insertion_code

    @property
    def chain_id(self) -> str:
        return self._ptr.chain_id

    @property
    def is_amino_acid(self) -> bool:
        return self._ptr.is_amino_acid

    @property
    def atoms(self) -> List[Atom]:
        return [Atom.from_py_ptr(a) for a in self._ptr.atoms]

    @property
    def conformer_names(self) -> List[str]:
        return self._ptr.conformer_names

    def __len__(self) -> int:
        return len(self._ptr)

    def __repr__(self) -> str:
        return repr(self._ptr)


class Chain(RustWrapperObject):
    """A chain in a structural model."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> Chain:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def id(self) -> str:
        return self._ptr.id

    @property
    def residue_count(self) -> int:
        return self._ptr.residue_count

    @property
    def atom_count(self) -> int:
        return self._ptr.atom_count

    @property
    def residues(self) -> List[Residue]:
        return [Residue.from_py_ptr(r) for r in self._ptr.residues]

    @property
    def atoms(self) -> List[Atom]:
        return [Atom.from_py_ptr(a) for a in self._ptr.atoms]

    def __len__(self) -> int:
        return len(self._ptr)

    def __repr__(self) -> str:
        return repr(self._ptr)


class Model(RustWrapperObject):
    """A structural model (e.g. NMR ensemble member)."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> Model:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def serial_number(self) -> int:
        return self._ptr.serial_number

    @property
    def chain_count(self) -> int:
        return self._ptr.chain_count

    @property
    def residue_count(self) -> int:
        return self._ptr.residue_count

    @property
    def atom_count(self) -> int:
        return self._ptr.atom_count

    @property
    def chains(self) -> List[Chain]:
        return [Chain.from_py_ptr(c) for c in self._ptr.chains]

    @property
    def residues(self) -> List[Residue]:
        return [Residue.from_py_ptr(r) for r in self._ptr.residues]

    @property
    def atoms(self) -> List[Atom]:
        return [Atom.from_py_ptr(a) for a in self._ptr.atoms]

    def __repr__(self) -> str:
        return repr(self._ptr)


class Structure(RustWrapperObject):
    """A parsed PDB or mmCIF structure.

    Provides hierarchy navigation and bulk numpy array access.

    Examples:
        >>> s = ferritin.load("1crn.pdb")
        >>> s.coords.shape
        (327, 3)
        >>> for chain in s.chains:
        ...     print(chain.id, chain.residue_count)
    """

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> Structure:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    # -- metadata ------------------------------------------------------------

    @property
    def identifier(self) -> Optional[str]:
        return self._ptr.identifier

    # -- counts --------------------------------------------------------------

    @property
    def model_count(self) -> int:
        return self._ptr.model_count

    @property
    def chain_count(self) -> int:
        return self._ptr.chain_count

    @property
    def residue_count(self) -> int:
        return self._ptr.residue_count

    @property
    def atom_count(self) -> int:
        return self._ptr.atom_count

    @property
    def total_atom_count(self) -> int:
        return self._ptr.total_atom_count

    # -- hierarchy -----------------------------------------------------------

    @property
    def models(self) -> List[Model]:
        return [Model.from_py_ptr(m) for m in self._ptr.models]

    @property
    def chains(self) -> List[Chain]:
        return [Chain.from_py_ptr(c) for c in self._ptr.chains]

    @property
    def residues(self) -> List[Residue]:
        return [Residue.from_py_ptr(r) for r in self._ptr.residues]

    @property
    def atoms(self) -> List[Atom]:
        return [Atom.from_py_ptr(a) for a in self._ptr.atoms]

    # -- bulk numpy arrays ---------------------------------------------------

    @property
    def coords(self) -> NDArray[np.float64]:
        """All atom coordinates as Nx3 numpy array."""
        return self._ptr.coords

    @property
    def b_factors(self) -> NDArray[np.float64]:
        """All B-factors as numpy array."""
        return self._ptr.b_factors

    @property
    def occupancies(self) -> NDArray[np.float64]:
        """All occupancies as numpy array."""
        return self._ptr.occupancies

    # -- bulk metadata -------------------------------------------------------

    @property
    def atom_names(self) -> List[str]:
        return self._ptr.atom_names

    @property
    def elements(self) -> List[str]:
        return self._ptr.elements

    @property
    def residue_names(self) -> List[str]:
        return self._ptr.residue_names

    @property
    def chain_ids(self) -> List[str]:
        return self._ptr.chain_ids

    @property
    def residue_serial_numbers(self) -> NDArray[np.int64]:
        return self._ptr.residue_serial_numbers

    # -- dunder --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ptr)

    def __repr__(self) -> str:
        return repr(self._ptr)
