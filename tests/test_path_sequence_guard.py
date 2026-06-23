"""The batch convenience functions must not silently iterate a single path string.

A bare ``str`` / ``os.PathLike`` is a valid ``Sequence``, so passing one path to a
batch function used to iterate it character-by-character and fail to load each char
(silent garbage). The `normalize_paths` guard wraps a single path into a
one-element list and emits a DeprecationWarning nudging the list form.
"""

import os
import warnings

import pytest

import proteon
from proteon.io import as_path_sequence

PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")
P = os.path.join(PDBS, "1crn.pdb")

# (name, result-size accessor) for the path-taking convenience functions.
PATH_FUNCS = [
    "load_and_sasa", "load_and_dssp", "load_and_extract_ca", "load_and_contact_maps",
    "load_and_analyze", "load_and_minimize_hydrogens", "batch_load", "batch_load_tolerant",
    "batch_load_tolerant_with_rescue", "batch_load_and_prepare", "prepare_for_supervision",
]


def _size(r):
    return r.n_ok if hasattr(r, "n_ok") else len(r)


class TestAsPathSequence:
    def test_wraps_str_with_warning(self):
        with pytest.warns(DeprecationWarning):
            assert as_path_sequence("a.pdb") == ["a.pdb"]

    def test_wraps_pathlike_with_warning(self):
        from pathlib import Path
        with pytest.warns(DeprecationWarning):
            assert as_path_sequence(Path("a.pdb")) == [Path("a.pdb")]

    def test_list_passthrough_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails
            assert as_path_sequence(["a.pdb", "b.pdb"]) == ["a.pdb", "b.pdb"]


class TestSinglePathFootgunFixed:
    @pytest.mark.parametrize("name", PATH_FUNCS)
    def test_single_string_wraps_and_warns(self, name):
        fn = getattr(proteon, name)
        with pytest.warns(DeprecationWarning):
            r = fn(P)
        # Exactly ONE structure processed (not 18 chars, not 0).
        assert _size(r) == 1

    @pytest.mark.parametrize("name", ["load_and_sasa", "load_and_dssp", "batch_load"])
    def test_list_unaffected(self, name):
        fn = getattr(proteon, name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            r = fn([P])  # must NOT warn
        assert _size(r) == 1

    def test_value_correct_via_wrap(self):
        # The wrapped single-path result equals the direct primitive.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = proteon.load_and_sasa(P)
        assert abs(r.values[0] - proteon.total_sasa(proteon.load(P))) < 0.1
