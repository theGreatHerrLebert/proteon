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
            "build_sequence_dataset",
            "build_structure_supervision_dataset",
        }
        assert expected <= set(ns)
