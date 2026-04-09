import os
import tempfile
import importlib
from pathlib import Path
from unittest.mock import patch

import ferritin


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAMBIN = os.path.join(REPO, "test-pdbs", "1crn.pdb")
UBIQ = os.path.join(REPO, "test-pdbs", "1ubq.pdb")
SEARCH_MOD = importlib.import_module("ferritin.search")


class TestSearchDB:
    def test_build_search_db_skips_failed_loads(self):
        paths = [CRAMBIN, "/does/not/exist.pdb", UBIQ]
        db = ferritin.build_search_db(paths, k=4, n_threads=2)

        assert isinstance(db, ferritin.SearchDB)
        assert db.version == 4
        assert db.k == 4
        assert len(db) == 2
        assert [entry.source_index for entry in db.entries] == [0, 2]
        assert all(entry.aa_sequence for entry in db.entries)
        assert db.aa_postings

    def test_build_search_db_supports_mixed_k_values(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=[4, 5, 6], n_threads=2)
        query = ferritin.load(CRAMBIN)
        hits = ferritin.search(query, db, top_k=1, rerank=False)

        assert db.k == 6
        assert db.k_values == [4, 5, 6]
        assert all(":" in key for key in db.postings)
        assert hits[0].source_path == CRAMBIN

    def test_query_self_hit_ranks_first(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = ferritin.load(CRAMBIN)
        hits = ferritin.search(query, db, top_k=2)

        assert len(hits) >= 1
        assert isinstance(hits[0], ferritin.SearchHit)
        assert hits[0].source_path == CRAMBIN
        assert hits[0].tm_score is not None
        assert hits[0].prefilter_score >= 0.0
        assert hits[0].score >= hits[0].prefilter_score

    def test_save_and_load_search_db_roundtrip(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)
            loaded = ferritin.load_search_db(path)

            root = Path(path)
            assert (root / "manifest.json").exists()
            assert (root / "entries.parquet").exists()
            assert (root / "postings").exists()
            assert (root / "positional_postings").exists()

            assert isinstance(loaded, ferritin.SearchDB)
            assert loaded.version == db.version
            assert loaded.k == db.k
            assert loaded.k_values == db.k_values
            assert loaded.root_path == path
            assert len(loaded) == len(db)
            hits = ferritin.search(ferritin.load(CRAMBIN), loaded, top_k=1, rerank=False)
            assert hits[0].source_path == CRAMBIN

    def test_build_search_db_can_write_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            db = ferritin.build_search_db([CRAMBIN, UBIQ], out=path, k=4, n_threads=2)
            loaded = ferritin.load_search_db(path)

            assert len(db) == len(loaded)
            hits = ferritin.search(ferritin.load(UBIQ), loaded, top_k=1, rerank=False)
            assert hits[0].source_path == UBIQ

    def test_search_accepts_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.build_search_db([CRAMBIN, UBIQ], out=path, k=4, n_threads=2)
            query = ferritin.load(UBIQ)
            hits = ferritin.search(query, path, top_k=1)

        assert len(hits) == 1
        assert hits[0].source_path == UBIQ

    def test_search_without_rerank_returns_prefilter_scores(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = ferritin.load(CRAMBIN)
        hits = ferritin.search(query, db, top_k=2, rerank=False, diagonal_rescore=False)

        assert len(hits) >= 1
        assert hits[0].tm_score is None
        assert hits[0].score == hits[0].prefilter_score
        assert hits[0].alphabet_score >= 0.0
        assert hits[0].aa_score >= 0.0

    def test_search_can_diagonal_rescore_before_rerank(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = ferritin.load(CRAMBIN)
        hits = ferritin.search(query, db, top_k=2, rerank=False, diagonal_rescore=True)

        assert len(hits) >= 1
        assert hits[0].tm_score is None
        assert hits[0].diagonal_score is not None
        assert hits[0].score == hits[0].diagonal_score

    def test_rerank_reuses_structure_cache_across_searches(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = ferritin.load(CRAMBIN)

        load_calls = []
        real_load = ferritin.load

        def tracked_load(path):
            load_calls.append(path)
            return real_load(path)

        with patch.object(SEARCH_MOD, "_load_structure", side_effect=tracked_load):
            ferritin.search(query, db, top_k=2, rerank=True, rerank_top_k=2, cache_max_size=8)
            first_call_count = len(load_calls)
            ferritin.search(query, db, top_k=2, rerank=True, rerank_top_k=2, cache_max_size=8)

        assert first_call_count > 0
        assert len(load_calls) == first_call_count

    def test_save_db_writes_bucketed_postings(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)
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
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = ferritin.load(CRAMBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)
            loaded = ferritin.load_search_db(path)

            read_calls = []
            real_read_table = SEARCH_MOD.pq.read_table

            def tracked_read_table(*args, **kwargs):
                read_calls.append(str(args[0]))
                return real_read_table(*args, **kwargs)

            with patch.object(SEARCH_MOD.pq, "read_table", side_effect=tracked_read_table):
                ferritin.search(query, loaded, top_k=2, rerank=False, posting_cache_max_size=128)
                first_call_count = len(read_calls)
                ferritin.search(query, loaded, top_k=2, rerank=False, posting_cache_max_size=128)

            assert first_call_count > 0
            assert len(read_calls) == first_call_count

    def test_lazy_db_uses_positional_posting_cache(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = ferritin.load(CRAMBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)
            loaded = ferritin.load_search_db(path, prefer_compiled=False)

            hits = ferritin.search(query, loaded, top_k=1, rerank=False, diagonal_rescore=False)

            assert hits[0].source_path == CRAMBIN
            assert hits[0].diagonal_vote_score is not None
            assert len(loaded.positional_posting_bucket_cache) > 0

    def test_warm_search_db_preloads_lazy_posting_cache(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)
            loaded = ferritin.load_search_db(path)

            assert loaded.posting_bucket_cache == {}

            warmed = ferritin.warm_search_db(loaded, posting_cache_max_size=128)

            assert warmed is loaded
            assert len(warmed.posting_bucket_cache) > 0
            assert len(warmed.positional_posting_bucket_cache) > 0

    def test_search_path_can_be_warmed(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)

            warmed = ferritin.warm_search_db(path, posting_cache_max_size=128)

            assert isinstance(warmed, ferritin.SearchDB)
            assert warmed.root_path == path
            assert len(warmed.posting_bucket_cache) > 0

    def test_compile_search_db_writes_compiled_layout(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)

            compiled = ferritin.compile_search_db(path)
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
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)
        query = ferritin.load(CRAMBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)
            ferritin.compile_search_db(path)

            loaded = ferritin.load_search_db(path)
            hits = ferritin.search(query, loaded, top_k=1, rerank=False)

            assert loaded.entries is not None
            assert loaded.postings is not None
            assert loaded.positional_postings is not None
            assert hits[0].source_path == CRAMBIN

    def test_compile_search_db_reuses_warmed_bucket_cache(self):
        db = ferritin.build_search_db([CRAMBIN, UBIQ], k=4, n_threads=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "search_db")
            ferritin.save_search_db(db, path)
            loaded = ferritin.load_search_db(path)
            ferritin.warm_search_db(loaded, posting_cache_max_size=128)

            real_read_table = SEARCH_MOD.pq.read_table

            def tracked_read_table(*args, **kwargs):
                if "postings" in str(args[0]):
                    raise AssertionError("compile_search_db reread postings despite warmed cache")
                return real_read_table(*args, **kwargs)

            with patch.object(SEARCH_MOD.pq, "read_table", side_effect=tracked_read_table):
                compiled = ferritin.compile_search_db(loaded)

            assert compiled.entries is not None
            assert compiled.postings is not None
