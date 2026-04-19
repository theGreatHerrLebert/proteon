import os
import tempfile
import importlib
import warnings
from pathlib import Path
from unittest.mock import patch

import proteon
import pytest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAMBIN = os.path.join(REPO, "test-pdbs", "1crn.pdb")
UBIQ = os.path.join(REPO, "test-pdbs", "1ubq.pdb")
SEARCH_MOD = importlib.import_module("proteon.search")


class TestSearchDB:
    def test_load_sa_matrix_uses_env_override(self, monkeypatch, tmp_path):
        matrix_path = tmp_path / "mat3di.out"
        matrix_path.write_text(
            "# demo matrix\n"
            "A C X\n"
            "A 6 -2 -1\n"
            "C -2 6 -1\n"
            "X -1 -1 6\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("PROTEON_FOLDSEEK_3DI_MATRIX", str(matrix_path))

        alphabet, matrix, source = SEARCH_MOD._load_sa_matrix()

        assert alphabet == "ACX"
        assert matrix["A"]["C"] == -2
        assert source == str(matrix_path)

    def test_load_sa_matrix_falls_back_without_candidates(self):
        with patch.object(SEARCH_MOD, "_candidate_foldseek_3di_matrix_paths", return_value=[]):
            alphabet, matrix, source = SEARCH_MOD._load_sa_matrix()

        assert alphabet == SEARCH_MOD._DEFAULT_SA_ALPHABET
        assert matrix["A"]["A"] == 6
        assert matrix["A"]["C"] == -2
        assert source == "fallback"

    def test_load_sa_matrix_raises_for_invalid_env_override(self, monkeypatch, tmp_path):
        matrix_path = tmp_path / "mat3di.out"
        matrix_path.write_text("not a valid matrix\n", encoding="utf-8")
        monkeypatch.setenv("PROTEON_FOLDSEEK_3DI_MATRIX", str(matrix_path))

        with pytest.raises(ValueError, match="substitution matrix"):
            SEARCH_MOD._load_sa_matrix()

    def test_encode_alphabet_raises_clean_error_without_native_backend(self):
        with patch.object(SEARCH_MOD, "_search", None):
            with pytest.raises(ImportError, match="proteon-connector"):
                SEARCH_MOD.encode_alphabet(object())

    def test_build_search_db_skips_failed_loads(self):
        paths = [CRAMBIN, "/does/not/exist.pdb", UBIQ]
        db = proteon.build_search_db(paths, k=4, n_threads=2)

        assert isinstance(db, proteon.SearchDB)
        assert db.version == 4
        assert db.k == 4
        assert len(db) == 2
        assert [entry.source_index for entry in db.entries] == [0, 2]
        assert all(entry.aa_sequence for entry in db.entries)
        assert db.aa_postings

    def test_build_search_db_supports_mixed_k_values(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=[4, 5, 6], n_threads=2)
        query = proteon.load(CRAMBIN)
        hits = proteon.search(query, db, top_k=1, rerank=False)

        assert db.k == 6
        assert db.k_values == [4, 5, 6]
        assert all(":" in key for key in db.postings)
        assert hits[0].source_path == CRAMBIN

    def test_query_self_hit_ranks_first(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)
        hits = proteon.search(query, db, top_k=2)

        assert len(hits) >= 1
        assert isinstance(hits[0], proteon.SearchHit)
        assert hits[0].source_path == CRAMBIN
        assert hits[0].tm_score is not None
        assert hits[0].prefilter_score >= 0.0
        assert hits[0].score >= hits[0].prefilter_score

    def test_save_and_load_search_db_roundtrip(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path)
            loaded = proteon.load_search_db(path)

            root = Path(path)
            assert (root / "manifest.json").exists()
            assert (root / "entries.parquet").exists()
            assert (root / "postings").exists()
            assert (root / "positional_postings").exists()
            assert (root / "compiled" / "manifest.json").exists()

            assert isinstance(loaded, proteon.SearchDB)
            assert loaded.version == db.version
            assert loaded.k == db.k
            assert loaded.k_values == db.k_values
            assert loaded.root_path == path
            assert loaded.entries is not None
            assert loaded.postings is not None
            assert len(loaded) == len(db)
            hits = proteon.search(proteon.load(CRAMBIN), loaded, top_k=1, rerank=False)
            assert hits[0].source_path == CRAMBIN

    def test_build_search_db_can_write_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            db = proteon.build_search_db([CRAMBIN, UBIQ], out=path, k=4, n_threads=2)
            loaded = proteon.load_search_db(path)

            assert len(db) == len(loaded)
            assert loaded.entries is not None
            assert loaded.postings is not None
            hits = proteon.search(proteon.load(UBIQ), loaded, top_k=1, rerank=False)
            assert hits[0].source_path == UBIQ

    def test_save_search_db_can_skip_compiled_layout(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)
            root = Path(path)
            loaded = proteon.load_search_db(path, prefer_compiled=False)

            assert not (root / "compiled" / "manifest.json").exists()
            assert loaded.entries is None
            assert loaded.postings is None
            assert loaded.root_path == path

    def test_load_search_db_warns_when_compiled_layout_is_missing(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)

            with pytest.warns(UserWarning, match="Compiled search layout not found"):
                loaded = proteon.load_search_db(path)

            assert loaded.entries is None
            assert loaded.postings is None

    def test_load_search_db_prefer_compiled_false_skips_warning(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = proteon.load_search_db(path, prefer_compiled=False)

            assert caught == []
            assert loaded.entries is None
            assert loaded.postings is None

    def test_load_search_db_can_auto_compile_missing_layout(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            root = Path(path)
            proteon.save_search_db(db, path, write_compiled=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = proteon.load_search_db(path, auto_compile_missing=True)

            assert caught == []
            assert (root / "compiled" / "manifest.json").exists()
            assert loaded.entries is not None
            assert loaded.postings is not None

    def test_search_accepts_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.build_search_db([CRAMBIN, UBIQ], out=path, k=4, n_threads=2)
            query = proteon.load(UBIQ)
            hits = proteon.search(query, path, top_k=1)

        assert len(hits) == 1
        assert hits[0].source_path == UBIQ

    def test_search_path_warns_when_it_falls_back_to_lazy_db(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(UBIQ)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)

            with pytest.warns(UserWarning, match="Compiled search layout not found"):
                hits = proteon.search(query, path, top_k=1, rerank=False)

        assert len(hits) == 1
        assert hits[0].source_path == UBIQ

    def test_search_path_can_auto_compile_missing_layout(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(UBIQ)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            root = Path(path)
            proteon.save_search_db(db, path, write_compiled=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                hits = proteon.search(
                    query,
                    path,
                    top_k=1,
                    rerank=False,
                    auto_compile_missing=True,
                )

            assert caught == []
            assert (root / "compiled" / "manifest.json").exists()
            assert len(hits) == 1
            assert hits[0].source_path == UBIQ

    def test_search_without_rerank_returns_prefilter_scores(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)
        hits = proteon.search(query, db, top_k=2, rerank=False, diagonal_rescore=False)

        assert len(hits) >= 1
        assert hits[0].tm_score is None
        assert hits[0].score == hits[0].prefilter_score
        assert hits[0].alphabet_score >= 0.0
        assert hits[0].aa_score >= 0.0

    def test_search_can_diagonal_rescore_before_rerank(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)
        hits = proteon.search(query, db, top_k=2, rerank=False, diagonal_rescore=True)

        assert len(hits) >= 1
        assert hits[0].tm_score is None
        assert hits[0].diagonal_score is not None
        assert hits[0].score == hits[0].diagonal_score

    def test_rerank_reuses_structure_cache_across_searches(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)

        load_calls = []
        real_load = proteon.load

        def tracked_load(path):
            load_calls.append(path)
            return real_load(path)

        with patch.object(SEARCH_MOD, "_load_structure", side_effect=tracked_load):
            proteon.search(query, db, top_k=2, rerank=True, rerank_top_k=2, cache_max_size=8)
            first_call_count = len(load_calls)
            proteon.search(query, db, top_k=2, rerank=True, rerank_top_k=2, cache_max_size=8)

        assert first_call_count > 0
        assert len(load_calls) == first_call_count

    def test_save_db_writes_bucketed_postings(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path)
            root = Path(path)
            buckets = list((root / "postings").rglob("*.parquet"))

            assert buckets
            assert any("kind=sa" in str(bucket) for bucket in buckets)
            assert any("kind=aa" in str(bucket) for bucket in buckets)

            positional_buckets = list((root / "positional_postings").rglob("*.parquet"))
            assert positional_buckets
            assert any("kind=sa" in str(bucket) for bucket in positional_buckets)
            assert any("kind=aa" in str(bucket) for bucket in positional_buckets)

    def test_lazy_db_reuses_posting_bucket_cache(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)
            loaded = proteon.load_search_db(path, prefer_compiled=False)

            read_calls = []
            real_read_table = SEARCH_MOD.pq.read_table

            def tracked_read_table(*args, **kwargs):
                read_calls.append(str(args[0]))
                return real_read_table(*args, **kwargs)

            with patch.object(SEARCH_MOD.pq, "read_table", side_effect=tracked_read_table):
                proteon.search(query, loaded, top_k=2, rerank=False, posting_cache_max_size=128)
                first_call_count = len(read_calls)
                proteon.search(query, loaded, top_k=2, rerank=False, posting_cache_max_size=128)

            assert first_call_count > 0
            assert len(read_calls) == first_call_count

    def test_lazy_db_reuses_entry_cache(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)
            loaded = proteon.load_search_db(path, prefer_compiled=False)

            read_calls = []
            real_read_table = SEARCH_MOD.pq.read_table

            def tracked_read_table(*args, **kwargs):
                read_calls.append(str(args[0]))
                return real_read_table(*args, **kwargs)

            with patch.object(SEARCH_MOD.pq, "read_table", side_effect=tracked_read_table):
                proteon.search(query, loaded, top_k=2, rerank=False)
                entry_reads_after_first = [
                    call for call in read_calls
                    if call.endswith("entries.parquet")
                ]
                proteon.search(query, loaded, top_k=2, rerank=False)

            entry_reads_after_second = [
                call for call in read_calls
                if call.endswith("entries.parquet")
            ]
            assert len(entry_reads_after_first) > 0
            assert len(entry_reads_after_second) == len(entry_reads_after_first)

    def test_lazy_db_uses_positional_posting_cache(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path)
            loaded = proteon.load_search_db(path, prefer_compiled=False)

            hits = proteon.search(query, loaded, top_k=1, rerank=False, diagonal_rescore=False)

            assert hits[0].source_path == CRAMBIN
            assert hits[0].diagonal_vote_score is not None
            assert len(loaded.positional_posting_bucket_cache) > 0

    def test_warm_search_db_preloads_lazy_posting_cache(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)
            loaded = proteon.load_search_db(path, prefer_compiled=False)

            assert loaded.posting_bucket_cache == {}

            warmed = proteon.warm_search_db(loaded, posting_cache_max_size=128)

            assert warmed is loaded
            assert len(warmed.posting_bucket_cache) > 0
            assert len(warmed.positional_posting_bucket_cache) > 0

    def test_search_path_can_be_warmed(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path, write_compiled=False)

            with pytest.warns(UserWarning, match="Compiled search layout not found"):
                warmed = proteon.warm_search_db(path, posting_cache_max_size=128)

            assert isinstance(warmed, proteon.SearchDB)
            assert warmed.root_path == path
            assert len(warmed.posting_bucket_cache) > 0

    def test_warm_search_db_can_auto_compile_missing_layout(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            root = Path(path)
            proteon.save_search_db(db, path, write_compiled=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warmed = proteon.warm_search_db(path, auto_compile_missing=True)

            assert caught == []
            assert (root / "compiled" / "manifest.json").exists()
            assert warmed.entries is not None
            assert warmed.postings is not None

    def test_compile_search_db_writes_compiled_layout(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path)

            compiled = proteon.compile_search_db(path)
            root = Path(path)

            assert (root / "compiled" / "manifest.json").exists()
            assert (root / "compiled" / "entries.arrow").exists()
            assert (root / "compiled" / "postings_sa.arrow").exists()
            assert (root / "compiled" / "postings_aa.arrow").exists()
            assert (root / "compiled" / "positional_postings_sa.arrow").exists()
            assert (root / "compiled" / "positional_postings_aa.arrow").exists()
            assert compiled.entries is not None
            assert compiled.postings is not None
            assert compiled.aa_postings is not None
            assert compiled.positional_postings is not None
            assert compiled.aa_positional_postings is not None

    def test_load_search_db_prefers_compiled_layout(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = proteon.load(CRAMBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path)
            proteon.compile_search_db(path)

            loaded = proteon.load_search_db(path)
            hits = proteon.search(query, loaded, top_k=1, rerank=False)

            assert loaded.entries is not None
            assert loaded.postings is not None
            assert loaded.positional_postings is not None
            assert hits[0].source_path == CRAMBIN

    def test_compile_search_db_reuses_warmed_bucket_cache(self):
        db = proteon.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            proteon.save_search_db(db, path)
            loaded = proteon.load_search_db(path)
            proteon.warm_search_db(loaded, posting_cache_max_size=128)

            real_read_table = SEARCH_MOD.pq.read_table

            def tracked_read_table(*args, **kwargs):
                if "postings" in str(args[0]):
                    raise AssertionError("compile_search_db reread postings despite warmed cache")
                return real_read_table(*args, **kwargs)

            with patch.object(SEARCH_MOD.pq, "read_table", side_effect=tracked_read_table):
                compiled = proteon.compile_search_db(loaded)

            assert compiled.entries is not None
            assert compiled.postings is not None
