import pytest
import proteon


class TestTopLevelExports:
    def test___all___exists_and_is_unique(self):
        assert isinstance(proteon.__all__, tuple)
        assert proteon.__all__
        assert len(proteon.__all__) == len(set(proteon.__all__))

    def test___all___entries_resolve(self):
        missing = [name for name in proteon.__all__ if not hasattr(proteon, name)]
        assert not missing

    def test_star_import_exposes_core_symbols(self):
        ns = {}
        exec("from proteon import *", ns)

        expected = {
            "__version__",
            "load",
            "save",
            "Structure",
            "tm_align",
            "compute_energy",
            "prepare",
            "build_search_db",
            "build_sequence_example",
            "batch_build_structure_supervision_examples",
        }
        assert expected <= set(ns)

    def test_build_sequence_dataset_is_top_level_canonical_entry_point(self):
        """v0.3.0 Phase A reversed an earlier v0.2.1-era decision to keep
        `build_sequence_dataset` submodule-only. Per codex review on
        TO_V030_TRAINING_CORPUS_FACTORY.md, the MSA-wired sequence-release
        builder is the canonical "structures + MSA -> sequence release"
        path and belongs in the top-level surface alongside
        `build_structure_supervision_dataset_from_prepared`. The earlier
        assertion that it must NOT be exposed is replaced by the
        opposite — it MUST be exposed, by name, in `proteon.__all__`.
        """
        assert "build_sequence_dataset" in proteon.__all__
        assert callable(proteon.build_sequence_dataset)
