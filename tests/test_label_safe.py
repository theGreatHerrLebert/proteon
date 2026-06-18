"""The label-safe preparation contract (PrepReport label-safety signals).

`prepare` is step 0 of geometric-DL supervision; the prepared coordinates and FF
assignment become training labels, so a silent corruption poisons every example.
These signals make that an explicit, structured decision:

    report = proteon.prepare(structure)
    if report.label_safe:
        use_as_training_label(structure)
    else:
        log(report.label_hazards)   # the specific hazards that fired
"""

import os

import pytest

import proteon
from proteon import PrepReport, PrepStatus

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDBS = os.path.join(REPO, "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def _pdb(name):
    return os.path.join(PDBS, name)


# --- contract logic on constructed reports (no connector needed) ---


class TestContractLogic:
    def test_clean_report_is_label_safe(self):
        r = PrepReport(
            hydrogens_added=50, minimizer_status="converged_gradient",
            n_heavy_clashes=0, n_models=1,
        )
        assert r.label_safe is True
        assert r.label_hazards == []
        assert r.label_safe_heavy_coords is True
        assert r.label_safe_all_atom_coords is True
        assert r.label_safe_energy is True
        assert r.label_safe_sequence_indexed is True

    def test_severe_clash_blocks_label_safe(self):
        # Severe = clashscore > threshold: 3 clashes / 10 heavy atoms = 300.
        r = PrepReport(hydrogens_added=50, n_heavy_clashes=3, n_heavy_atoms=10)
        assert r.has_heavy_clashes is True
        assert r.has_severe_clashes is True
        assert r.label_safe is False
        assert r.label_safe_heavy_coords is False
        assert "severe_heavy_clashes" in r.label_hazards

    def test_mild_clash_is_observation_not_hazard(self):
        # A few clashes in a large structure (clashscore well under 20) is an
        # observation, NOT a label hazard — the whole point of the severity gate.
        r = PrepReport(hydrogens_added=50, n_heavy_clashes=3, n_heavy_atoms=3000)
        assert r.has_heavy_clashes is True       # honest: clashes exist
        assert r.has_severe_clashes is False     # but not severe
        assert r.label_safe_heavy_coords is True
        assert "severe_heavy_clashes" not in r.label_hazards

    def test_deep_overlap_is_severe_regardless_of_clashscore(self):
        # One catastrophic interpenetration in a huge structure: clashscore tiny,
        # but the max-overlap cap makes it severe (interface-dilution guard).
        r = PrepReport(hydrogens_added=50, n_heavy_clashes=1, n_heavy_atoms=5000,
                       max_heavy_overlap=2.0)
        assert r.has_severe_clashes is True
        assert "severe_heavy_clashes" in r.label_hazards

    def test_reconstructed_atoms_block_coords(self):
        r = PrepReport(hydrogens_added=50, atoms_reconstructed=4)
        assert r.has_reconstructed_atoms is True
        assert r.label_safe_heavy_coords is False  # fabricated coords aren't observed
        assert "reconstructed_atoms" in r.label_hazards

    def test_missing_atoms_block_coords(self):
        # Incomplete residues (reconstruct off) — a partial coordinate label.
        r = PrepReport(hydrogens_added=50, n_missing_heavy_atoms=2)
        assert r.has_missing_atoms is True
        assert r.label_safe_heavy_coords is False
        assert r.label_safe is False
        assert "missing_atoms" in r.label_hazards

    def test_altlocs_block_coords_but_not_sequence(self):
        r = PrepReport(hydrogens_added=50, has_altlocs=True)
        assert r.label_safe_heavy_coords is False  # arbitrary conformer pick
        assert r.label_safe_sequence_indexed is True  # residue identity unaffected
        assert "altlocs" in r.label_hazards

    def test_multiple_models_block_everything(self):
        r = PrepReport(hydrogens_added=50, n_models=8)
        assert r.has_multiple_models is True
        assert r.label_safe_heavy_coords is False
        assert r.label_safe_sequence_indexed is False
        assert "multiple_models" in r.label_hazards

    def test_insertion_codes_block_only_sequence_indexed(self):
        r = PrepReport(hydrogens_added=50, has_insertion_codes=True)
        assert r.label_safe_heavy_coords is True   # coordinates are fine
        assert r.label_safe_sequence_indexed is False
        assert r.label_safe is False               # strict gate fails
        assert "insertion_codes" in r.label_hazards

    def test_untyped_cofactor_blocks_energy_not_coords(self):
        # A heme protein: backbone coords are fine, but energy isn't (untyped heme).
        r = PrepReport(
            hydrogens_added=50, untyped_cofactors=True, n_unassigned_nonwater=43,
        )
        assert r.status == PrepStatus.READY_WITH_LIGANDS
        assert r.label_safe_heavy_coords is True
        assert r.label_safe_all_atom_coords is True
        assert r.label_safe_energy is False  # untyped atoms
        assert "untyped_atoms" in r.label_hazards

    def test_numerical_failure_blocks_coords(self):
        r = PrepReport(minimizer_status="numerical_failure")
        assert r.label_safe_heavy_coords is False
        assert r.label_safe is False

    def test_skipped_h_does_not_block_all_atom(self):
        # hydrogens_skipped is dominated by legitimate chemistry (proline, termini)
        # and is NOT a label hazard — a proline-containing protein must stay safe.
        r = PrepReport(hydrogens_added=40, hydrogens_skipped=3)
        assert r.label_safe_heavy_coords is True
        assert r.label_safe_all_atom_coords is True


# --- integration: real structures round-trip to sensible verdicts ---


class TestContractIntegration:
    def test_pristine_structure_is_label_safe(self):
        # 1crn (0.5 A crambin) is the cleanest fixture: no clashes, no hazards.
        r = proteon.prepare(proteon.load(_pdb("1crn.pdb")))
        assert r.n_heavy_clashes == 0
        assert r.label_safe is True, f"1crn hazards: {r.label_hazards}"

    def test_clash_count_tracks_quality(self):
        # The pristine high-res structure has zero protein clashes; the metric is
        # the gate that keeps a clashy structure out of training data.
        r = proteon.prepare(proteon.load(_pdb("1crn.pdb")))
        assert r.n_heavy_clashes == 0

    def test_heme_protein_coords_safe_energy_not(self):
        # 4hhb carries heme: protein heavy-coord labels are usable, but it is not
        # fully typed (energy labels) — and ligand contacts are excluded from the
        # protein clash count (clash_count_inferred flags the un-templated heme).
        r = proteon.prepare(proteon.load(_pdb("4hhb.pdb")))
        assert r.clash_count_inferred is True
        assert r.has_untyped_atoms is True
        assert r.label_safe_energy is False

    def test_label_safe_is_bool_on_corpus_fixtures(self):
        for sub, name in [("waters", "protein_with_waters.pdb"),
                          ("ligands", "protein_with_ligand.pdb")]:
            r = proteon.prepare(proteon.load(os.path.join(CORPUS, sub, name)))
            assert isinstance(r.label_safe, bool)
            assert isinstance(r.label_hazards, list)
