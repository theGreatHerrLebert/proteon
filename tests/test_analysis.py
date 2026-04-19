"""Tests for the pure-Python analysis functions.

Tests: distance_matrix, contact_map, dihedral_angle, backbone_dihedrals,
       centroid, radius_of_gyration, to_dataframe, extract_ca_coords.
"""

import os

import numpy as np
import pytest

import proteon

TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")


def load_crambin():
    return proteon.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))


def load_ubiquitin():
    return proteon.load(os.path.join(EXAMPLE_DIR, "1ubq.pdb"))


# ===========================================================================
# distance_matrix
# ===========================================================================


class TestDistanceMatrix:
    def test_self_distance_shape(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        dm = proteon.distance_matrix(coords)
        assert dm.shape == (3, 3)

    def test_self_distance_diagonal_zero(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        dm = proteon.distance_matrix(coords)
        np.testing.assert_allclose(np.diag(dm), 0.0)

    def test_self_distance_symmetric(self):
        coords = np.random.randn(20, 3)
        dm = proteon.distance_matrix(coords)
        np.testing.assert_allclose(dm, dm.T)

    def test_known_distances(self):
        coords = np.array([[0, 0, 0], [3, 4, 0]], dtype=np.float64)
        dm = proteon.distance_matrix(coords)
        np.testing.assert_allclose(dm[0, 1], 5.0)
        np.testing.assert_allclose(dm[1, 0], 5.0)

    def test_cross_distance_shape(self):
        a = np.random.randn(5, 3)
        b = np.random.randn(7, 3)
        dm = proteon.distance_matrix(a, b)
        assert dm.shape == (5, 7)

    def test_cross_distance_known(self):
        a = np.array([[0, 0, 0]], dtype=np.float64)
        b = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.float64)
        dm = proteon.distance_matrix(a, b)
        np.testing.assert_allclose(dm[0, 0], 1.0)
        np.testing.assert_allclose(dm[0, 1], 2.0)

    def test_on_structure(self):
        s = load_crambin()
        ca = proteon.extract_ca_coords(s)
        dm = proteon.distance_matrix(ca)
        assert dm.shape == (46, 46)
        # Adjacent CAs should be ~3.8 A
        assert 3.5 < dm[0, 1] < 4.1
        # All distances non-negative
        assert np.all(dm >= 0)


# ===========================================================================
# contact_map
# ===========================================================================


class TestContactMap:
    def test_basic(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float64)
        cm = proteon.contact_map(coords, cutoff=2.0)
        assert cm[0, 1] == True
        assert cm[0, 2] == False

    def test_self_contact(self):
        coords = np.random.randn(10, 3)
        cm = proteon.contact_map(coords, cutoff=1000.0)
        assert cm.all()  # all within cutoff

    def test_no_contacts(self):
        coords = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float64)
        cm = proteon.contact_map(coords, cutoff=1.0)
        assert cm[0, 1] == False

    def test_on_structure(self):
        s = load_crambin()
        ca = proteon.extract_ca_coords(s)
        cm8 = proteon.contact_map(ca, cutoff=8.0)
        cm12 = proteon.contact_map(ca, cutoff=12.0)
        # More contacts at larger cutoff
        assert cm12.sum() >= cm8.sum()
        # Diagonal is always True (self-contact)
        assert np.all(np.diag(cm8))

    def test_cross_contacts(self):
        a = np.array([[0, 0, 0]], dtype=np.float64)
        b = np.array([[1, 0, 0], [10, 0, 0]], dtype=np.float64)
        cm = proteon.contact_map(a, cutoff=2.0, coords2=b)
        assert cm.shape == (1, 2)
        assert cm[0, 0] == True
        assert cm[0, 1] == False


# ===========================================================================
# dihedral_angle
# ===========================================================================


class TestDihedralAngle:
    def test_cis(self):
        """Four coplanar points in cis configuration → 0 degrees."""
        p0 = np.array([1.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        d = proteon.dihedral_angle(p0, p1, p2, p3)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_trans(self):
        """Four coplanar points in trans configuration → 180 degrees."""
        p0 = np.array([1.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, -1.0, 0.0])
        d = proteon.dihedral_angle(p0, p1, p2, p3)
        np.testing.assert_allclose(abs(d), 180.0, atol=1e-10)

    def test_right_angle(self):
        """90-degree dihedral."""
        p0 = np.array([1.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])
        d = proteon.dihedral_angle(p0, p1, p2, p3)
        np.testing.assert_allclose(abs(d), 90.0, atol=1e-10)

    def test_vectorized(self):
        """Vectorized computation on multiple dihedral sets."""
        p0 = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float64)
        p1 = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
        p2 = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
        p3 = np.array([[1, 1, 0], [1, -1, 0]], dtype=np.float64)
        d = proteon.dihedral_angle(p0, p1, p2, p3)
        assert d.shape == (2,)
        np.testing.assert_allclose(d[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(abs(d[1]), 180.0, atol=1e-10)

    def test_range(self):
        """All dihedral angles should be in [-180, 180]."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            pts = rng.randn(4, 3)
            d = proteon.dihedral_angle(pts[0], pts[1], pts[2], pts[3])
            assert -180.0 <= d <= 180.0


# ===========================================================================
# backbone_dihedrals
# ===========================================================================


class TestBackboneDihedrals:
    def test_shape(self):
        s = load_crambin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        assert len(phi) == 46
        assert len(psi) == 46
        assert len(omega) == 46

    def test_first_phi_nan(self):
        """First residue has no preceding C, so phi is NaN."""
        s = load_crambin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        assert np.isnan(phi[0])

    def test_last_psi_nan(self):
        """Last residue has no following N, so psi is NaN."""
        s = load_crambin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        assert np.isnan(psi[-1])

    def test_first_omega_nan(self):
        """First residue has no preceding CA/C, so omega is NaN."""
        s = load_crambin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        assert np.isnan(omega[0])

    def test_helix_phi_psi(self):
        """Alpha helix residues should have phi ~ -60, psi ~ -45."""
        s = load_crambin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        # Residues 8-18 are annotated as H (helix) in SS
        helix_phi = phi[7:18]
        helix_psi = psi[7:18]
        assert np.all(helix_phi < -40)
        assert np.all(helix_phi > -90)
        assert np.all(helix_psi < -10)
        assert np.all(helix_psi > -60)

    def test_omega_trans(self):
        """All omega angles should be near +/-180 (trans peptide bonds)."""
        s = load_crambin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        valid_omega = omega[~np.isnan(omega)]
        assert np.all(np.abs(valid_omega) > 150)

    def test_phi_psi_range(self):
        """All defined angles should be in [-180, 180]."""
        s = load_crambin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        for angles in [phi, psi, omega]:
            valid = angles[~np.isnan(angles)]
            assert np.all(valid >= -180.0)
            assert np.all(valid <= 180.0)

    def test_ubiquitin(self):
        """Test on a different structure (ubiquitin)."""
        s = load_ubiquitin()
        phi, psi, omega = proteon.backbone_dihedrals(s)
        # backbone_dihedrals only counts amino acid residues
        assert len(phi) == 76  # 76 amino acids, rest are water
        valid_omega = omega[~np.isnan(omega)]
        # All trans
        assert np.all(np.abs(valid_omega) > 140)


class TestBackboneDihedralsPythonFallback:
    """The public `backbone_dihedrals` goes through the Rust connector
    when it's available; `_backbone_dihedrals_python` is the pure-Python
    fallback kept for array-only usage. The Rust path dominates in
    practice, so the fallback is easy to let rot — these tests pin its
    output against the Rust path so a divergence surfaces as a
    diagnostic CI failure rather than a silent "mysterious results
    only when Rust disabled" bug report.
    """

    def test_matches_rust_on_crambin(self):
        from proteon.analysis import _backbone_dihedrals_python

        s = load_crambin()
        phi_rust, psi_rust, omega_rust = proteon.backbone_dihedrals(s)
        phi_py, psi_py, omega_py = _backbone_dihedrals_python(s)

        # Same residue-count output.
        assert phi_py.shape == phi_rust.shape
        # Valid (non-NaN) positions agree to float precision on both paths.
        for label, rust_arr, py_arr in [
            ("phi", phi_rust, phi_py),
            ("psi", psi_rust, psi_py),
            ("omega", omega_rust, omega_py),
        ]:
            mask = ~np.isnan(rust_arr) & ~np.isnan(py_arr)
            assert mask.sum() > 0, f"{label}: no valid positions to compare"
            assert np.allclose(rust_arr[mask], py_arr[mask]), (
                f"{label}: Rust vs Python fallback diverges on crambin"
            )

    def test_matches_rust_on_ubiquitin(self):
        from proteon.analysis import _backbone_dihedrals_python

        s = load_ubiquitin()
        phi_rust, psi_rust, omega_rust = proteon.backbone_dihedrals(s)
        phi_py, psi_py, omega_py = _backbone_dihedrals_python(s)

        assert phi_py.shape == phi_rust.shape
        for rust_arr, py_arr in [
            (phi_rust, phi_py),
            (psi_rust, psi_py),
            (omega_rust, omega_py),
        ]:
            mask = ~np.isnan(rust_arr) & ~np.isnan(py_arr)
            assert np.allclose(rust_arr[mask], py_arr[mask])

    def test_termini_nan(self):
        """N-terminal phi and omega are undefined (no preceding
        residue) and C-terminal psi is undefined (no following
        residue). Fallback must return NaN at those positions,
        matching the Rust convention."""
        from proteon.analysis import _backbone_dihedrals_python

        s = load_crambin()
        phi, psi, omega = _backbone_dihedrals_python(s)
        # N-terminal residue: phi needs C_{i-1} (doesn't exist) and
        # omega needs CA_{i-1}, C_{i-1} (don't exist). Both NaN.
        assert np.isnan(phi[0])
        assert np.isnan(omega[0])
        # C-terminal residue: psi needs N_{i+1} (doesn't exist).
        assert np.isnan(psi[-1])

    def test_empty_structure_returns_empty_arrays(self):
        """Guards the `if not all_phi: return empty, empty, empty`
        branch — a structure with zero amino-acid residues (e.g. water
        or ligand only) must yield empty float64 arrays, not crash."""
        from proteon.analysis import _backbone_dihedrals_python

        class _EmptyStructure:
            chains = []

        phi, psi, omega = _backbone_dihedrals_python(_EmptyStructure())
        assert phi.shape == (0,)
        assert psi.shape == (0,)
        assert omega.shape == (0,)
        assert phi.dtype == np.float64


# ===========================================================================
# centroid & radius_of_gyration
# ===========================================================================


class TestGeometry:
    def test_centroid_known(self):
        coords = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)
        c = proteon.centroid(coords)
        np.testing.assert_allclose(c, [2/3, 2/3, 0.0])

    def test_centroid_single_point(self):
        coords = np.array([[5.0, 3.0, 1.0]])
        c = proteon.centroid(coords)
        np.testing.assert_allclose(c, [5.0, 3.0, 1.0])

    def test_rg_known(self):
        """Two points at distance d from origin → Rg = d."""
        coords = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float64)
        rg = proteon.radius_of_gyration(coords)
        np.testing.assert_allclose(rg, 1.0)

    def test_rg_on_structure(self):
        s = load_crambin()
        rg = proteon.radius_of_gyration(s.coords)
        # Crambin Rg should be roughly 9-10 A
        assert 8.0 < rg < 12.0


# ===========================================================================
# extract_ca_coords
# ===========================================================================


class TestExtractCA:
    def test_crambin(self):
        s = load_crambin()
        ca = proteon.extract_ca_coords(s)
        assert ca.shape == (46, 3)
        assert ca.dtype == np.float64

    def test_ubiquitin(self):
        s = load_ubiquitin()
        ca = proteon.extract_ca_coords(s)
        # Ubiquitin has 76 amino acid residues (rest are water/HETATM)
        assert ca.shape[0] == 76
        assert ca.shape[1] == 3

    def test_ca_subset_of_all(self):
        """CA coords should be a subset of all coords."""
        s = load_crambin()
        ca = proteon.extract_ca_coords(s)
        all_coords = s.coords
        # Each CA coord should appear in the full coord array
        for i in range(len(ca)):
            diffs = np.linalg.norm(all_coords - ca[i], axis=1)
            assert diffs.min() < 1e-10


# ===========================================================================
# to_dataframe
# ===========================================================================


class TestToDataframe:
    def test_shape(self):
        s = load_crambin()
        df = proteon.to_dataframe(s)
        assert df.shape[0] == s.atom_count
        assert df.shape[1] == 10

    def test_columns(self):
        s = load_crambin()
        df = proteon.to_dataframe(s)
        expected = {"atom_name", "element", "residue_name", "residue_number",
                    "chain_id", "x", "y", "z", "b_factor", "occupancy"}
        assert set(df.columns) == expected

    def test_coords_match(self):
        s = load_crambin()
        df = proteon.to_dataframe(s)
        coords = s.coords
        np.testing.assert_allclose(df["x"].values, coords[:, 0])
        np.testing.assert_allclose(df["y"].values, coords[:, 1])
        np.testing.assert_allclose(df["z"].values, coords[:, 2])

    def test_ca_filter(self):
        s = load_crambin()
        df = proteon.to_dataframe(s)
        ca = df[df.atom_name.str.strip() == "CA"]
        assert len(ca) == 46

    def test_invalid_engine(self):
        s = load_crambin()
        with pytest.raises(ValueError, match="Unknown engine"):
            proteon.to_dataframe(s, engine="spark")


# ===========================================================================
# Integration: distance_matrix + contact_map on real structure
# ===========================================================================


class TestIntegration:
    def test_contact_map_decreases_with_cutoff(self):
        s = load_crambin()
        ca = proteon.extract_ca_coords(s)
        c6 = proteon.contact_map(ca, cutoff=6.0).sum()
        c8 = proteon.contact_map(ca, cutoff=8.0).sum()
        c12 = proteon.contact_map(ca, cutoff=12.0).sum()
        assert c6 <= c8 <= c12

    def test_adjacent_ca_distance(self):
        """Adjacent CA atoms should be ~3.8 A apart."""
        s = load_crambin()
        ca = proteon.extract_ca_coords(s)
        dm = proteon.distance_matrix(ca)
        for i in range(len(ca) - 1):
            d = dm[i, i + 1]
            assert 3.5 < d < 4.1, f"CA {i}-{i+1} distance {d:.2f} out of range"

    def test_cross_distance_between_structures(self):
        s1 = load_crambin()
        s2 = load_ubiquitin()
        ca1 = proteon.extract_ca_coords(s1)
        ca2 = proteon.extract_ca_coords(s2)
        dm = proteon.distance_matrix(ca1, ca2)
        assert dm.shape == (len(ca1), len(ca2))
        assert np.all(dm >= 0)
