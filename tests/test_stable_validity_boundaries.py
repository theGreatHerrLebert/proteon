"""Validity-boundary + correctness suite for the STABLE API tier (readiness #2).

Robustness (47k structures, zero crashes) is not correctness. This suite pins
what each *stable* (promised) function does on ugly-but-common inputs: the
invariant is **fail loud OR carry an explicit validity signal — never a silent
plausible-but-wrong number** (narrowed to functions whose output can be mistaken
for a meaningful scientific result; low-level numeric utilities where NaN/empty
is the conventional signal are exempt from the raise requirement).

Design + rationale: devdocs/STABLE_VALIDITY_BOUNDARY_DESIGN.md. The tier itself
is frozen by tests/test_public_api_surface.py.
"""

import tempfile
import warnings

import numpy as np
import pytest

import proteon
from proteon import ParameterizationError


def _pdb(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    f.write(text)
    f.close()
    return f.name


# --- curated pathological fixtures (synthesized PDB text) --------------------
ONE_CA = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
TWO_CA = (
    ONE_CA.replace("END\n", "")
    + "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C\nEND\n"
)
HET_ONLY = "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n"
EMPTY = "END\n"


def _ca_chain(resname, record="ATOM", n=4):
    rows = [
        f"{record:<6s}{i + 1:5d}  CA  {resname} A{i + 1:4d}    "
        f"{3.8 * i:8.3f}{1.5 * (i % 2):8.3f}   0.000  1.00  0.00           C"
        for i in range(n)
    ]
    return "\n".join(rows) + "\nEND\n"


# A real 4-atom glycine backbone (no hydrogens) — fully typable.
GLY_NOH = """ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.000   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.300   2.420   0.000  1.00  0.00           O
END
"""


class TestComputeEnergyValidityBoundary:
    """The worst gap the probe found: total=0.0 for unparameterizable input."""

    def test_water_only_raises_not_zero(self):
        # HETATM-only water: the protein FF drops every atom (n_topo_atoms == 0).
        # Returning total=0.0 would read as "perfectly relaxed" — must raise.
        with pytest.raises(ParameterizationError):
            proteon.compute_energy(proteon.load(_pdb(HET_ONLY)))

    def test_parameterization_error_is_valueerror(self):
        # Subclasses ValueError so existing `except ValueError` still catches.
        assert issubclass(ParameterizationError, ValueError)

    def test_single_atom_zero_energy_is_legitimate_complete(self):
        # A single CA atom has n_topo_atoms == 1 and genuinely zero internal
        # energy (no interactions) — this is correct, not a degraded sentinel.
        d = proteon.compute_energy(proteon.load(_pdb(ONE_CA)))
        assert d["parameterization_status"] == "complete"
        assert d["is_parameterized"] is True
        assert d["total"] == 0.0

    def test_real_structure_is_complete(self):
        d = proteon.compute_energy(proteon.load("test-pdbs/1crn.pdb"))
        assert d["parameterization_status"] == "complete"
        assert d["is_parameterized"] is True
        assert d["total"] != 0.0

    def test_batch_annotates_and_does_not_raise(self):
        # Batch must NOT abort on one unparameterizable structure — it annotates.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = proteon.batch_compute_energy(
                [proteon.load(_pdb(HET_ONLY)), proteon.load("test-pdbs/1crn.pdb")]
            )
        assert out[0]["parameterization_status"] == "empty"
        assert out[0]["is_parameterized"] is False
        assert out[1]["parameterization_status"] == "complete"


class TestSasaBoundary:
    """total_sasa is atom-set-correct, NOT protein-validated (documented). It is
    a low-level surface calc, so it does NOT raise on non-protein input — but the
    behaviour is pinned so it can't silently drift."""

    def test_sasa_of_arbitrary_atoms_is_finite_positive(self):
        for txt in (ONE_CA, HET_ONLY):
            val = proteon.total_sasa(proteon.load(_pdb(txt)))
            assert np.isfinite(val) and val > 0.0

    def test_sasa_empty_structure_raises_on_load(self):
        # No silent zero — an empty file fails loud at load time.
        with pytest.raises(OSError):
            proteon.total_sasa(proteon.load(_pdb(EMPTY)))


class TestKabschBoundary:
    """Low-level numeric utility: correctness oracles + pinned NaN/degenerate
    conventions (NaN-in/NaN-out, rank-deficient -> finite but underdetermined)."""

    def test_identity_recovery(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(12, 3))
        rmsd = proteon.kabsch_superpose(x, x.copy())[0]
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_known_rigid_transform_recovery(self):
        # Apply a known rotation + translation; Kabsch must recover it (rmsd ~ 0).
        rng = np.random.default_rng(1)
        x = rng.normal(size=(20, 3))
        theta = 0.7
        rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1.0]]
        )
        y = x @ rot.T + np.array([5.0, 2.0, -1.0])
        rmsd = proteon.kabsch_superpose(x, y)[0]
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            proteon.kabsch_superpose(np.zeros((1, 3)), np.zeros((2, 3)))

    def test_nan_in_nan_out_is_the_documented_convention(self):
        # Conventional for a numeric utility; pinned so it can't silently change
        # to a plausible finite number.
        rmsd = proteon.kabsch_superpose(
            np.array([[np.nan, 0, 0.0]]), np.array([[0, 0, 0.0]])
        )[0]
        assert np.isnan(rmsd)


class TestAlignmentBoundary:
    """tm_align already fails loud on degenerate input (good) + MSE->MET is a
    real correctness property."""

    @pytest.mark.parametrize("txt", [ONE_CA, HET_ONLY, EMPTY])
    def test_degenerate_alignment_fails_loud(self, txt):
        ref = proteon.load("test-pdbs/1crn.pdb")
        with pytest.raises((RuntimeError, ValueError, OSError)):
            proteon.tm_align(ref, proteon.load(_pdb(txt)))

    def test_self_alignment_is_one(self):
        ref = proteon.load("test-pdbs/1crn.pdb")
        assert proteon.tm_align(ref, ref).tm_score_chain1 == pytest.approx(1.0, abs=1e-3)

    def test_mse_is_normalized_to_methionine(self):
        # MSE (selenomethionine) as ATOM records must be read as MET ('M'),
        # matching how USalign treats it — identical coords, MSE vs MET, align
        # to TM=1 with seq_identity=1 and an all-'M' aligned sequence.
        mse = proteon.load(_pdb(_ca_chain("MSE", "ATOM")))
        met = proteon.load(_pdb(_ca_chain("MET", "ATOM")))
        r = proteon.tm_align(mse, met)
        assert r.tm_score_chain1 == pytest.approx(1.0, abs=1e-3)
        assert r.seq_identity == pytest.approx(1.0, abs=1e-3)
        assert set(r.aligned_seq_x.replace("-", "")) == {"M"}


class TestDsspBoundary:
    """dssp returns a plain str; '' is a COARSE signal (conflates empty /
    too-short / no-assignable-SS). Pinned + documented as a known limitation."""

    @pytest.mark.parametrize("txt", [ONE_CA, HET_ONLY])
    def test_unassignable_returns_empty_string(self, txt):
        assert proteon.dssp(proteon.load(_pdb(txt))) == ""

    def test_real_structure_assigns_secondary_structure(self):
        ss = proteon.dssp(proteon.load("test-pdbs/1crn.pdb"))
        assert isinstance(ss, str) and len(ss) > 0


class TestBackboneDihedralsBoundary:
    def test_too_short_returns_empty(self):
        for txt in (ONE_CA, TWO_CA):
            phi, _psi, _omega = proteon.backbone_dihedrals(proteon.load(_pdb(txt)))
            assert np.asarray(phi).size == 0


# --- meta-coverage: keep "curated" from becoming "whatever we remembered" ----

#: Stable callables exercised by a boundary/correctness test above (directly or
#: via a representative — many-variant wrappers delegate to these cores).
_COVERED = {
    "compute_energy", "batch_compute_energy", "ParameterizationError",
    "total_sasa", "kabsch_superpose", "tm_align", "dssp", "backbone_dihedrals",
    "load",
}

#: Stable callables intentionally NOT given their own boundary test, each with a
#: reason. A new stable callable that is neither covered nor exempt fails the
#: meta-test below — forcing a deliberate classification (mirrors the #206 guard).
_EXEMPT = {
    # Data model / result types — constructed by the library, not user-facing
    # compute entry points with a validity boundary.
    "Atom": "data type", "Chain": "data type", "Model": "data type",
    "Residue": "data type", "Structure": "data type",
    "AlignResult": "result type", "ChainPairResult": "result type",
    "FlexAlignResult": "result type", "MMAlignResult": "result type",
    "SoiAlignResult": "result type", "LoadRescueResult": "result type",
    # Alignment variants delegate to the tm_align/mm_align/soi_align/flex_align core.
    "tm_align_one_to_many": "delegates to tm_align",
    "tm_align_many_to_many": "delegates to tm_align",
    "mm_align": "delegates to alignment core", "mm_align_one_to_many": "delegates to mm_align",
    "mm_align_many_to_many": "delegates to mm_align",
    "soi_align": "delegates to alignment core", "soi_align_one_to_many": "delegates to soi_align",
    "soi_align_many_to_many": "delegates to soi_align",
    "flex_align": "delegates to alignment core", "flex_align_one_to_many": "delegates to flex_align",
    "flex_align_many_to_many": "delegates to flex_align",
    # SASA variants delegate to the same surface calc as total_sasa.
    "residue_sasa": "delegates to SASA core", "atom_sasa": "delegates to SASA core",
    "relative_sasa": "delegates to SASA core", "batch_total_sasa": "batch of total_sasa",
    "batch_residue_sasa": "batch of residue_sasa", "batch_atom_sasa": "batch of atom_sasa",
    "batch_relative_sasa": "batch of relative_sasa", "load_and_sasa": "load + total_sasa",
    # Analysis: pure geometry on coords/structures (total functions, no domain boundary).
    "contact_map": "pure geometry", "distance_matrix": "pure geometry",
    "extract_ca_coords": "pure geometry", "radius_of_gyration": "pure geometry",
    "centroid": "pure geometry", "dihedral_angle": "pure geometry",
    "to_dataframe": "pure projection", "load_and_analyze": "load + analysis",
    "load_and_contact_maps": "load + contact_map", "load_and_extract_ca": "load + extract_ca",
    "batch_contact_maps": "batch", "batch_dihedrals": "batch", "batch_distance_matrices": "batch",
    "batch_extract_ca": "batch", "batch_radius_of_gyration": "batch",
    # Geometry: pure math, covered transitively (kabsch) or trivial.
    "rmsd": "pure geometry", "rmsd_no_super": "pure geometry", "tm_score": "pure geometry",
    "apply_transform": "pure geometry", "assign_secondary_structure": "pure geometry",
    # DSSP variants delegate to dssp.
    "dssp_array": "delegates to dssp", "batch_dssp": "batch of dssp", "load_and_dssp": "load + dssp",
    # H-bonds: geometric counts, covered by oracle parity elsewhere.
    "backbone_hbonds": "geometric oracle elsewhere", "geometric_hbonds": "geometric oracle elsewhere",
    "hbond_count": "geometric oracle elsewhere", "batch_backbone_hbonds": "batch",
    "batch_hbond_count": "batch",
    # Forcefield: minimize family + gpu introspection.
    "minimize_hydrogens": "H-only relax, parity-tested elsewhere",
    "batch_minimize_hydrogens": "batch", "minimize_structure": "minimizer, tested elsewhere",
    "load_and_minimize_hydrogens": "load + minimize", "gpu_available": "device probe, no structure input",
    "gpu_info": "device probe, no structure input",
    # I/O: load is covered; the rest are format/rescue variants + savers.
    "load_pdb": "format variant of load", "load_mmcif": "format variant of load",
    "load_with_rescue": "tolerant variant of load", "batch_load": "batch of load",
    "batch_load_tolerant": "tolerant batch", "batch_load_tolerant_with_rescue": "tolerant batch",
    "save": "writer", "save_pdb": "writer", "save_mmcif": "writer",
}


def test_every_stable_callable_is_covered_or_exempt():
    callables = {
        n for n in proteon.__stable__ if callable(getattr(proteon, n))
    }
    unclassified = sorted(callables - _COVERED - set(_EXEMPT))
    assert not unclassified, (
        "stable callables with neither a boundary test nor a documented "
        f"exemption: {unclassified}. Add a test to _COVERED or an entry to "
        "_EXEMPT with a reason (see devdocs/STABLE_VALIDITY_BOUNDARY_DESIGN.md)."
    )


def test_covered_names_are_actually_stable():
    # Guard against a covered/exempt name silently leaving the stable tier.
    drifted = sorted((_COVERED | set(_EXEMPT)) - proteon.__stable__)
    assert not drifted, f"names classified here but no longer stable: {drifted}"
