import os

import numpy as np

import ferritin
import ferritin_connector


REPO = os.path.dirname(os.path.dirname(__file__))
CRAMBIN = os.path.join(REPO, "test-pdbs", "1crn.pdb")
CORPUS_MISSING_CB = os.path.join(REPO, "tests", "corpus", "missing_atoms", "missing_cb.pdb")


class TestAlphabetEncoding:
    def test_encode_alphabet_returns_expected_shapes(self):
        s = ferritin.load(CRAMBIN)
        result = ferritin.encode_alphabet(s)

        n = len(result["states"])
        assert n > 0
        assert len(result["alphabet"]) == n
        assert result["valid_mask"].shape == (n,)
        assert result["partners"].shape == (n,)
        assert result["features"].shape == (n, 10)
        assert len(result["chain_ids"]) == n
        assert len(result["residue_names"]) == n
        assert result["residue_numbers"].shape == (n,)
        assert len(result["insertion_codes"]) == n

    def test_encode_alphabet_has_valid_non_placeholder_states(self):
        s = ferritin.load(CRAMBIN)
        result = ferritin.encode_alphabet(s)

        valid_states = result["states"][result["valid_mask"]]
        assert len(valid_states) > 0
        assert np.all((valid_states >= 0) & (valid_states < 20))
        assert np.any(valid_states != 2)

    def test_encode_alphabet_is_deterministic(self):
        s = ferritin.load(CRAMBIN)
        r1 = ferritin.encode_alphabet(s)
        r2 = ferritin.encode_alphabet(s)

        assert r1["alphabet"] == r2["alphabet"]
        np.testing.assert_array_equal(r1["states"], r2["states"])
        np.testing.assert_array_equal(r1["valid_mask"], r2["valid_mask"])
        np.testing.assert_array_equal(r1["partners"], r2["partners"])
        np.testing.assert_allclose(r1["features"], r2["features"])

    def test_encode_alphabet_handles_missing_cb(self):
        s = ferritin.load(CORPUS_MISSING_CB)
        result = ferritin.encode_alphabet(s)

        assert len(result["states"]) > 0
        assert len(result["alphabet"]) == len(result["states"])
        assert result["features"].shape[1] == 10
        assert result["valid_mask"].shape == result["states"].shape

    def test_batch_encode_alphabet_matches_single_structure_api(self):
        s1 = ferritin.load(CRAMBIN)
        s2 = ferritin.load(CORPUS_MISSING_CB)

        batch = ferritin.batch_encode_alphabet([s1, s2], n_threads=2)
        single1 = ferritin.encode_alphabet(s1)
        single2 = ferritin.encode_alphabet(s2)

        assert len(batch) == 2
        for batched, single in zip(batch, [single1, single2]):
            assert batched["alphabet"] == single["alphabet"]
            np.testing.assert_array_equal(batched["states"], single["states"])
            np.testing.assert_array_equal(batched["valid_mask"], single["valid_mask"])
            np.testing.assert_array_equal(batched["partners"], single["partners"])
            np.testing.assert_allclose(batched["features"], single["features"])
            np.testing.assert_array_equal(batched["residue_numbers"], single["residue_numbers"])
            assert batched["chain_ids"] == single["chain_ids"]
            assert batched["residue_names"] == single["residue_names"]
            assert batched["insertion_codes"] == single["insertion_codes"]

    def test_diagonal_rescore_batch_scores_identical_strings_highest(self):
        scores = ferritin_connector.py_search.diagonal_rescore_batch(
            "AAAA",
            "AAAA",
            ["AAAA", "CCCC"],
            ["AAAA", "CCCC"],
            n_threads=2,
        )

        assert len(scores) == 2
        assert scores[0] > scores[1]
        assert scores[0] > 0.0
