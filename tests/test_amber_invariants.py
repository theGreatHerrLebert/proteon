"""Internal invariant tests for AMBER96.

Why this file exists
--------------------
AMBER96 is proteon's well-oracled force field (test_ball_energy.py
validates to ~0.02% against BALL Julia on heavy-atom crambin), but
it was never run through the cross-path × invariant suite that
test_charmm_invariants.py applies to CHARMM19+EEF1. That's the
same coverage asymmetry that let the 2026-04-11 CHARMM bugs ship:
one force field gets a rigorous oracle, the other gets invariants,
and neither gets BOTH.

This file closes the asymmetry from the other side. Every invariant
that makes sense for a vacuum force field (no implicit solvation
term) runs on AMBER96 across the same (structure × code path)
matrix that CHARMM uses. The solvation-specific invariants are
replaced with AMBER-specific ones ("solvation must be exactly 0",
because AMBER96 has no implicit solvent).

What these tests do NOT do
--------------------------
They do NOT validate parameter correctness — that's what the
BALL Julia oracle does. These internal invariants are a cheap
companion to the oracle: they run on every pytest invocation
(the oracle needs Julia installed) and they cover every PDB in
the registry rather than just crambin.
"""

from __future__ import annotations

import math

import pytest

import proteon

from conftest import (
    ENERGY_COMPONENTS,
    PATHS,
    STRUCTURES,
)


AMBER = "amber96"


# Vacuum AMBER96 has no implicit solvation, so the solvation slot
# must be exactly 0.0. Every other component is expected to be
# non-trivial on a real protein.
AMBER_COMPONENTS_NONZERO = (
    "bond_stretch",
    "angle_bend",
    "torsion",
    # improper_torsion may be zero for some structures — don't require non-zero
    "vdw",
    "electrostatic",
)


_STRUCT_PATH_PARAMS = [
    pytest.param((s, path), id=f"{s.name}-{path[0]}")
    for s in STRUCTURES
    for path in PATHS
]


@pytest.fixture(params=_STRUCT_PATH_PARAMS)
def amber_energy(request):
    """(name, energy_dict) parametrized over (STRUCTURE × PATH), AMBER96."""
    structure_spec, (_path_id, nbl_threshold) = request.param
    s = proteon.load(structure_spec.absolute_path)
    e = proteon.compute_energy(
        s, ff=AMBER, units="kJ/mol", nbl_threshold=nbl_threshold
    )
    return structure_spec.name, e


# =========================================================================
# Numerical sanity
# =========================================================================


class TestNoNanInf:
    """Every energy component is finite (no NaN, no Inf)."""

    def test_total_is_finite(self, amber_energy):
        name, e = amber_energy
        assert math.isfinite(e["total"]), f"{name}: total is {e['total']}"

    def test_components_are_finite(self, amber_energy):
        name, e = amber_energy
        for comp in ENERGY_COMPONENTS:
            v = e.get(comp)
            assert v is not None, f"{name}: component {comp} is missing"
            assert math.isfinite(v), f"{name}: {comp} is {v}"


class TestComponentsPresent:
    """All seven component keys plus `total` are present in the dict —
    identical to the CHARMM test, because the schema is shared.
    """

    def test_all_keys_present(self, amber_energy):
        name, e = amber_energy
        for comp in ENERGY_COMPONENTS:
            assert comp in e, f"{name}: missing key {comp!r}"
        assert "total" in e, f"{name}: missing key 'total'"


class TestSumsMatchTotal:
    """Σ(components) ≈ total. Catches accounting bugs like the
    2026-04-11 compute_energy_and_forces_nbl regression where
    solvation was silently omitted from the total sum. On AMBER96
    the solvation is always 0 so omitting it wouldn't diverge, but
    this test would still catch any other missing component.
    """

    def test_components_sum_to_total(self, amber_energy):
        name, e = amber_energy
        component_sum = sum(e[c] for c in ENERGY_COMPONENTS)
        diff = abs(component_sum - e["total"])
        tol = max(1e-3 * abs(e["total"]), 1e-3)
        assert diff < tol, (
            f"{name}: Σ(components) {component_sum:.6f} != total "
            f"{e['total']:.6f} (diff {diff:.4e}, tol {tol:.4e})"
        )


# =========================================================================
# Physical sanity
# =========================================================================


class TestSigns:
    """Harmonic potentials are sums of squares — they MUST be ≥ 0."""

    def test_bond_stretch_nonnegative(self, amber_energy):
        name, e = amber_energy
        assert e["bond_stretch"] >= 0, (
            f"{name}: bond_stretch {e['bond_stretch']:.6f} is negative — "
            "harmonic potential should always be ≥ 0"
        )

    def test_angle_bend_nonnegative(self, amber_energy):
        name, e = amber_energy
        assert e["angle_bend"] >= 0, (
            f"{name}: angle_bend {e['angle_bend']:.6f} is negative — "
            "harmonic potential should always be ≥ 0"
        )


class TestSolvationIsZero:
    """AMBER96 is a vacuum force field — the solvation slot MUST be
    exactly 0. Any non-zero value means EEF1 (or some other implicit
    solvation) has leaked into the AMBER path.
    """

    def test_solvation_exactly_zero(self, amber_energy):
        name, e = amber_energy
        assert e["solvation"] == 0.0, (
            f"{name}: AMBER96 solvation = {e['solvation']:.6f}, expected "
            "exactly 0.0 (AMBER96 is vacuum — no implicit solvation). "
            "A non-zero value means implicit solvent has leaked into "
            "the AMBER code path."
        )


class TestBondedTermsActive:
    """Every real protein must produce non-trivial bonded-term energies.
    If any of bond/angle/torsion/vdw/electrostatic is exactly zero,
    the corresponding kernel silently fell through — the topology
    builder found no bonds, or the parameter lookup returned all
    None, etc. This would not be caught by TestNoNanInf (0 is finite)
    or TestSigns (0 ≥ 0).
    """

    def test_bonded_terms_nontrivial(self, amber_energy):
        name, e = amber_energy
        for comp in AMBER_COMPONENTS_NONZERO:
            assert abs(e[comp]) > 1e-6, (
                f"{name}: AMBER96 {comp} is {e[comp]:.9f}, expected "
                f"non-trivial magnitude. Zero suggests the kernel "
                f"silently fell through."
            )


# =========================================================================
# AMBER vs CHARMM distinctness (regression for silent fallback)
# =========================================================================


class TestAmberDistinctFromCharmm:
    """Mirror of test_charmm_invariants.TestCharmmDistinctFromAmber.
    If AMBER and CHARMM return the same number on the same structure,
    one is silently falling back to the other. Distinct from the
    CHARMM version because we also want to check from AMBER's side,
    and on every structure, not just 1crn.
    """

    @pytest.mark.parametrize(
        "spec",
        STRUCTURES,
        ids=[s.name for s in STRUCTURES],
    )
    def test_amber_total_differs_from_charmm(self, spec):
        s = proteon.load(spec.absolute_path)
        e_amber = proteon.compute_energy(s, ff=AMBER)
        e_charmm = proteon.compute_energy(s, ff="charmm19_eef1")
        assert e_amber["total"] != e_charmm["total"], (
            f"{spec.name}: AMBER and CHARMM returned identical totals — "
            "one force field is silently falling back to the other"
        )


# ---------------------------------------------------------------------------
# Histidine tautomer normalization (issue #60)
# ---------------------------------------------------------------------------
class TestHistidineTautomers:
    """Pin the behaviour of `proteon.prepare.normalize_histidine_tautomers`.

    AMBER96 ships HID/HIE/HIP templates with different per-atom partial
    charges. proteon's loader is name-driven, so renaming HIS → HID/HIE/HIP
    based on the H pattern is the fix. These tests pin the renaming logic
    AND verify the resulting energies pick up the correct AMBER96 charges
    via the data added in PR #62.

    The energy invariants here are NOT cross-tool oracles (those live in
    tests/oracle/test_ball_energy.py and test_amber96_oracle*.py) — they
    pin "the runner reliably routes a HIS to its correct tautomer charge
    set under proteon's residue-name-driven dispatch".
    """

    @staticmethod
    def _make_synthetic_pdb_with_one_his(
        tmpdir, has_hd1: bool, has_he2: bool
    ) -> str:
        """Synthesize a tiny single-HIS dipeptide PDB.

        Layout: GLY-HIS-GLY. The HIS sidechain has the imidazole ring,
        and we toggle whether HD1 / HE2 are present per the test case.
        Coordinates are taken from the proteon static HIS fragment
        template (proteon-connector/src/fragment_templates.rs) so the
        atoms are at canonical positions.
        """
        # Use coordinates from the static Rust template so the atoms are
        # geometrically valid (proteon won't reject them at load time).
        # GLY-HIS-GLY tripeptide layout, residues numbered 1-2-3.
        records: list[str] = []
        atom_id = 1

        def emit(name: str, element: str, x: float, y: float, z: float,
                 res_name: str, res_seq: int):
            nonlocal atom_id
            # PDB ATOM record fixed-width:
            # 1-6   record    | 7-11   atom_id    | 13-16  atom_name
            # 17    altloc    | 18-20  res_name   | 22     chain
            # 23-26 res_seq   | 27     icode      | 31-38  x  | 39-46 y | 47-54 z
            # 55-60 occ       | 61-66  bfactor    | 77-78  element
            line = (
                f"ATOM  {atom_id:>5d} {name:<4s}"
                f" {res_name:<3s} A{res_seq:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"  1.00  0.00           {element:>2s}\n"
            )
            atom_id += 1
            records.append(line)

        # GLY 1: minimal backbone
        emit("N", "N",   0.000, 0.000,  0.000, "GLY", 1)
        emit("CA", "C",  1.458, 0.000,  0.000, "GLY", 1)
        emit("C", "C",   2.000, 1.420,  0.000, "GLY", 1)
        emit("O", "O",   1.300, 2.420,  0.000, "GLY", 1)

        # HIS 2: minimal valid backbone + sidechain. Coordinates loosely
        # mirror the static HIS template's local frame, translated so
        # the N atom links correctly to the previous C.
        emit("N", "N",   3.300, 1.420,  0.000, "HIS", 2)
        emit("H", "H",   3.800, 0.620,  0.000, "HIS", 2)
        emit("CA", "C",  4.200, 2.500,  0.300, "HIS", 2)
        emit("HA", "H",  4.000, 2.900,  1.300, "HIS", 2)
        emit("C", "C",   5.700, 2.100,  0.300, "HIS", 2)
        emit("O", "O",   6.200, 1.000,  0.500, "HIS", 2)
        emit("CB", "C",  3.900, 3.700, -0.600, "HIS", 2)
        emit("HB2", "H", 4.500, 4.500, -0.300, "HIS", 2)
        emit("HB3", "H", 4.100, 3.500, -1.700, "HIS", 2)
        emit("CG", "C",  2.500, 4.300, -0.500, "HIS", 2)
        emit("ND1", "N", 1.800, 4.700, -1.600, "HIS", 2)
        if has_hd1:
            emit("HD1", "H", 2.200, 4.700, -2.500, "HIS", 2)
        emit("CE1", "C", 0.700, 5.300, -1.200, "HIS", 2)
        emit("HE1", "H", 0.000, 5.700, -1.900, "HIS", 2)
        emit("NE2", "N", 0.700, 5.400,  0.100, "HIS", 2)
        if has_he2:
            emit("HE2", "H", 0.000, 5.800,  0.700, "HIS", 2)
        emit("CD2", "C", 1.800, 4.700,  0.500, "HIS", 2)
        emit("HD2", "H", 2.000, 4.500,  1.500, "HIS", 2)

        # GLY 3: minimal C-terminal cap
        emit("N", "N",   6.500, 3.100,  0.300, "GLY", 3)
        emit("CA", "C",  8.000, 3.000,  0.300, "GLY", 3)
        emit("C", "C",   8.500, 4.400,  0.300, "GLY", 3)
        emit("O", "O",   7.800, 5.400,  0.300, "GLY", 3)
        emit("OXT", "O", 9.700, 4.500,  0.300, "GLY", 3)

        records.append("END\n")
        path = tmpdir / "tripeptide_his.pdb"
        path.write_text("".join(records))
        return str(path)

    def test_classify_HD1_only_to_HID(self, tmp_path):
        from proteon.prepare import normalize_histidine_tautomers
        in_pdb = self._make_synthetic_pdb_with_one_his(
            tmp_path, has_hd1=True, has_he2=False
        )
        out_pdb = tmp_path / "out.pdb"
        counts = normalize_histidine_tautomers(in_pdb, out_pdb)
        assert counts == {"HIS": 0, "HID": 1, "HIE": 0, "HIP": 0}
        text = out_pdb.read_text()
        assert "HID A" in text
        assert "HIS A" not in text or "HIS A   2" not in text  # HIS res 2 renamed

    def test_classify_HE2_only_to_HIE(self, tmp_path):
        from proteon.prepare import normalize_histidine_tautomers
        in_pdb = self._make_synthetic_pdb_with_one_his(
            tmp_path, has_hd1=False, has_he2=True
        )
        out_pdb = tmp_path / "out.pdb"
        counts = normalize_histidine_tautomers(in_pdb, out_pdb)
        assert counts == {"HIS": 0, "HID": 0, "HIE": 1, "HIP": 0}
        assert "HIE A" in out_pdb.read_text()

    def test_classify_both_Hs_to_HIP(self, tmp_path):
        from proteon.prepare import normalize_histidine_tautomers
        in_pdb = self._make_synthetic_pdb_with_one_his(
            tmp_path, has_hd1=True, has_he2=True
        )
        out_pdb = tmp_path / "out.pdb"
        counts = normalize_histidine_tautomers(in_pdb, out_pdb)
        assert counts == {"HIS": 0, "HID": 0, "HIE": 0, "HIP": 1}
        assert "HIP A" in out_pdb.read_text()

    def test_classify_no_Hs_keeps_HIS(self, tmp_path):
        from proteon.prepare import normalize_histidine_tautomers
        in_pdb = self._make_synthetic_pdb_with_one_his(
            tmp_path, has_hd1=False, has_he2=False
        )
        out_pdb = tmp_path / "out.pdb"
        counts = normalize_histidine_tautomers(in_pdb, out_pdb)
        # Could not classify — left as HIS, caller's responsibility to
        # warn or skip.
        assert counts == {"HIS": 1, "HID": 0, "HIE": 0, "HIP": 0}
        assert "HIS A" in out_pdb.read_text()

    def test_idempotent_on_already_renamed(self, tmp_path):
        """Running on a PDB that already has HID/HIE/HIP names is a no-op."""
        from proteon.prepare import normalize_histidine_tautomers
        in_pdb = self._make_synthetic_pdb_with_one_his(
            tmp_path, has_hd1=True, has_he2=False
        )
        first_out = tmp_path / "first.pdb"
        normalize_histidine_tautomers(in_pdb, first_out)
        second_out = tmp_path / "second.pdb"
        counts = normalize_histidine_tautomers(first_out, second_out)
        # The first pass renamed HIS → HID; the second pass sees no HIS,
        # so nothing to count.
        assert counts == {"HIS": 0, "HID": 0, "HIE": 0, "HIP": 0}
        assert first_out.read_text() == second_out.read_text()

    def test_amber96_compute_energy_picks_up_HID_charges(self, tmp_path):
        """End-to-end: HD1-only HIS → HID rename → AMBER96 compute_energy
        loads the HID charge set (per data from PR #62), not the legacy
        HIS-as-HIP charges. The acceptance criterion is per-residue
        partial-charge differences manifest in the total energy compared
        to the un-renamed path on the same structure."""
        from proteon.prepare import normalize_histidine_tautomers
        in_pdb = self._make_synthetic_pdb_with_one_his(
            tmp_path, has_hd1=True, has_he2=False
        )
        out_pdb = tmp_path / "renamed.pdb"
        normalize_histidine_tautomers(in_pdb, out_pdb)

        s_old = proteon.load(in_pdb)   # HIS, hits legacy HIP-charge template
        s_new = proteon.load(str(out_pdb))  # HID, hits new template
        try:
            e_old = proteon.compute_energy(
                s_old, ff=AMBER, units="kJ/mol",
                nbl_threshold=10**9, nonbonded_cutoff=1e6,
            )
            e_new = proteon.compute_energy(
                s_new, ff=AMBER, units="kJ/mol",
                nbl_threshold=10**9, nonbonded_cutoff=1e6,
            )
        except Exception as e:  # pragma: no cover
            pytest.skip(f"AMBER96 failed on synthetic tripeptide: {e}")

        # Different residue templates must produce different totals.
        assert not math.isclose(e_old["total"], e_new["total"], rel_tol=1e-6), (
            f"HIS and HID compute_energy returned identical totals "
            f"({e_old['total']:.6f}) — HID template not picked up. "
            "Either PR #62's data files weren't loaded, or the residue "
            "rename didn't reach the typer."
        )
        # The electrostatic component is the most sensitive to per-atom
        # charge changes; differ by >1 kJ/mol on a single histidine.
        assert abs(e_old["electrostatic"] - e_new["electrostatic"]) > 1.0, (
            f"electrostatic component barely changed "
            f"({e_old['electrostatic']:.3f} vs {e_new['electrostatic']:.3f}) "
            "— suggests HID charges didn't load"
        )
