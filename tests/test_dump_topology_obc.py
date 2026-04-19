"""dump_topology must accept the same force-field strings compute_energy does.

Previously dump_topology accepted only 'amber96' / 'charmm19_eef1', but
compute_energy accepted 'amber96_obc' (aliases 'amber96+obc',
'amber96_obc2'). That asymmetry blocked topology / charge diagnostics
for OBC exactly where you'd want them — when debugging OBC-specific
atom-type or charge assignments.

Guards the Rust-side dispatch arms in
`proteon-connector/src/py_forcefield.rs::dump_topology`. Skips when
the connector isn't built.
"""

from __future__ import annotations

from pathlib import Path

import pytest


try:
    from proteon_connector import py_forcefield
    _CONN_AVAILABLE = True
except ImportError:
    py_forcefield = None  # type: ignore
    _CONN_AVAILABLE = False

import proteon

FIXTURE = Path(__file__).resolve().parent.parent / "test-pdbs" / "1crn.pdb"

pytestmark = pytest.mark.skipif(
    not _CONN_AVAILABLE or not FIXTURE.exists(),
    reason="requires the Rust connector build and test-pdbs/1crn.pdb",
)


def _pyo3_pdb():
    from proteon.forcefield import _get_ptr
    return _get_ptr(proteon.load_pdb(str(FIXTURE)))


def test_dump_topology_accepts_amber96_obc():
    pdb = _pyo3_pdb()
    topo_amber = py_forcefield.dump_topology(pdb, "amber96")
    topo_obc = py_forcefield.dump_topology(pdb, "amber96_obc")
    # OBC only affects solvation; bonded topology + atom identities match AMBER96.
    assert len(topo_amber["bonds"]) == len(topo_obc["bonds"])
    assert len(topo_amber["angles"]) == len(topo_obc["angles"])
    assert len(topo_amber["torsions"]) == len(topo_obc["torsions"])
    assert len(topo_amber["impropers"]) == len(topo_obc["impropers"])
    assert topo_amber["atom_types"] == topo_obc["atom_types"]
    assert topo_amber["atom_charges"] == topo_obc["atom_charges"]


def test_dump_topology_accepts_obc_aliases():
    pdb = _pyo3_pdb()
    # Same aliases compute_energy accepts.
    t_canonical = py_forcefield.dump_topology(pdb, "amber96_obc")
    for alias in ("amber96+obc", "amber96_obc2"):
        t = py_forcefield.dump_topology(pdb, alias)
        assert t["atom_types"] == t_canonical["atom_types"]
        assert t["atom_charges"] == t_canonical["atom_charges"]


def test_dump_topology_error_mentions_obc_option():
    pdb = _pyo3_pdb()
    with pytest.raises(ValueError) as excinfo:
        py_forcefield.dump_topology(pdb, "unknown_ff")
    msg = str(excinfo.value)
    assert "amber96_obc" in msg, (
        "error message should list 'amber96_obc' so users know the option "
        f"is supported; got: {msg}"
    )
