# Template retrieval adapter — 3Di prefilter → candidate shortlist

## Motivation

`build_structure_template_features(query, candidate_structures, ...)` takes an
**explicit** candidate pool and TM-aligns the query against every one. At corpus
scale that's infeasible — you can't TM-align a query against all of the PDB.

The missing piece is a **prefilter**: shortlist structurally-similar candidates
*before* TM-align. The surprise from the terrain survey: proteon **already has
all the hard infrastructure** —

- a trained 3Di-style structural-alphabet encoder (`encode_alphabet`,
  `proteon-align/src/search/alphabet.rs`, VQ-VAE trained on 9.4k structures),
- an alphabet-agnostic k-mer prefilter (`proteon-search/src/prefilter.rs`),
- a full `search(query, db, top_k=...) -> List[SearchHit]` with diagonal
  rescoring + TM-align rerank (`packages/proteon/src/proteon/search.py:1423`),
- `build_search_db(paths, k=6) -> SearchDB` to index a corpus.

What's missing is the **adapter** that connects `search()` to the template
featurizer. This is STRUCTURE_TEMPLATES_PLAN T2 ("retrieval adapter — wire
proteon's structural search as the candidate engine"). Pure Python, no new Rust,
no encoder work.

## Scope

**In scope**

1. `retrieve_template_candidates(query, db, *, search_top_k, max_candidates,
   exclude_ids, exclude_self, min_tm)` → an ordered list of candidate
   `(structure, SearchHit)` (or just structures) for the featurizer. Runs
   `search()`, applies leakage filtering, loads the hit structures from their
   `source_path`.
2. `build_structure_template_features_from_db(query, db, *, ...)` — the
   convenience one-call path: retrieve → featurize. Returns `TemplateFeatures`.
3. **Leakage filtering** as a first-class input: `exclude_ids` (a caller-supplied
   blocklist — the query's own PDB id, known homologs, post-cutoff-date entries,
   same-cluster ids) and `exclude_self` (drop a hit whose id/source matches the
   query). A leakage caveat is the whole reason structural templating is honest
   only in a known-structure/refinement setting (plan intro) — so the filter is
   prominent and on by default for self.
4. Tests on `test-pdbs`: build a small DB, retrieve for a query, assert the
   shortlist is ordered/capped, self is excluded, a blocklist id is dropped, and
   the featurizer runs end-to-end on the shortlist producing a valid
   `TemplateFeatures`.

**Out of scope (deferred, separate projects)**

- The curated remote-homolog **benchmark** (recall@K vs TM-align ground truth on a
  fold/superfamily-labeled set) — SEARCH_ROADMAP already owns this; it's a
  measurement project, not adapter code.
- Date/clustering **leakage policy engine** — the adapter *consumes* an
  `exclude_ids` set; *computing* it (deposition dates, sequence clusters) is a
  separate corpus-policy concern.
- 3Di encoder changes, FoldSeek-exact reproduction, R*-tree neighbor-search
  optimization — the encoder is done and adequate.

## Design

### `retrieve_template_candidates`

```python
def retrieve_template_candidates(
    query,                       # Structure (with a known id) or path
    db,                          # SearchDB | path
    *,
    search_top_k: int = 50,      # how deep to search before filtering
    max_candidates: int = 25,    # cap after filtering (the TM-align pool size)
    exclude_ids: Optional[Iterable[str]] = None,   # leakage blocklist
    exclude_self: bool = True,   # drop a hit matching the query's id
    min_tm: Optional[float] = None,  # optional rerank TM-score floor
    query_id: Optional[str] = None,  # override for self-matching
) -> list[tuple[Structure, SearchHit]]:
```

- `search(query, db, top_k=search_top_k)` (rerank on, so hits carry `tm_score`).
- Drop hits whose `id` ∈ `exclude_ids`, or (`exclude_self`) whose `id` equals the
  query id (resolved from `query_id`, else the query structure's id).
- If `min_tm` set, drop hits with `tm_score` below it (None tm → kept only if no
  floor; a hit that wasn't reranked has tm=None — document that `min_tm` implies
  it should be within `rerank_top_k`).
- Load each surviving hit's structure from `source_path` via the tolerant loader;
  skip-with-warning any that fail to load (don't abort the shortlist).
- Return at most `max_candidates`, in search-rank order.

### `build_structure_template_features_from_db`

```python
def build_structure_template_features_from_db(
    query, db, *,
    top_k: int = 4,              # templates kept (the featurizer's top_k)
    search_top_k=50, max_candidates=25,
    exclude_ids=None, exclude_self=True, min_tm=None, query_id=None,
    fast=False, n_threads=None,
) -> TemplateFeatures:
    cands = retrieve_template_candidates(...)
    return build_structure_template_features(
        query, [s for s, _hit in cands], top_k=top_k, fast=fast, n_threads=n_threads
    )
```

### Known double-alignment

`search(rerank=True)` TM-aligns the top `rerank_top_k` hits, and
`build_structure_template_features` TM-aligns them **again** (it needs the full
residue correspondence the SearchHit doesn't carry). For `top_k≈4` candidates
this is a small, bounded cost; threading the search's alignment through to the
featurizer is a deferred optimization. Documented, not fixed.

## Test plan

- `build_search_db` over a handful of `test-pdbs`; `retrieve_template_candidates`
  returns ≤ `max_candidates` `(Structure, SearchHit)` in rank order.
- `exclude_self=True` drops the query's own entry (build a DB that includes the
  query, search it, assert the query id is absent).
- An `exclude_ids` blocklist entry never appears in the shortlist.
- `min_tm` floor removes low-TM hits (and the kept ones all have tm ≥ floor).
- `build_structure_template_features_from_db` runs end-to-end → a
  `TemplateFeatures` with `query_len == len(query residues)` and
  `n_templates ≤ top_k`.
- Empty/whiff: a query with no DB hits → empty candidate list →
  `n_templates == 0` (not an error).

## Claudex review outcome (adopted)

1. **`min_tm` vs rerank depth.** Hits beyond `rerank_top_k` carry `tm_score=None`.
   Filter is `hit.tm_score is not None and hit.tm_score >= min_tm` (boundary `==`
   kept). Expose `rerank_top_k`; if `min_tm` is set and `rerank_top_k <
   search_top_k`, **raise** — otherwise valid deep hits are dropped for missing
   data, a hidden recall loss.
2. **Pool starvation.** `search_top_k` is an *effort* limit, not a depth knob —
   self/blocklist/unloadable/below-`min_tm` hits consume it. Validate
   `search_top_k >= max_candidates >= top_k >= 0`. Load failures do **not** consume
   `max_candidates` (load survivors in rank order until the cap is filled).
   Return **diagnostics counts** (searched / excluded-self / excluded-blocklist /
   below-min-tm / load-failed / returned) for auditability. Adaptive over-fetch
   (re-search deeper on starvation) is noted as a future improvement, not in v1.
3. **Self/leakage matching is more than raw `id ==`.** Normalize ids
   (`strip().lower()`) on both sides and on `exclude_ids`. Self identity =
   normalized `query_id` (explicit > `query.identifier` > else) **and**, if a
   `query_path` is given, the canonicalized (`realpath`) `source_path`. If
   `exclude_self=True` but no query identity is resolvable, **raise** (a silent
   self-leak is worse than a loud error; caller passes `query_id`/`query_path` or
   `exclude_self=False`).
4. **Mechanism vs policy.** Also accept `exclude_candidate: Callable[[SearchHit],
   bool]` so callers express richer leakage policy (sequence-identity, date,
   cluster) without the adapter owning that policy. Computing date/cluster/seq
   blocklists stays external.
5. **Typed result, not a bare tuple.** `TemplateCandidate(structure, hit, rank)`
   preserving the original search rank; ordering follows the search's contract
   (reranked hits already promoted — preserve, don't re-sort).
6. **Two entry points.** `retrieve_template_candidate_hits(...)` is hit-only (no
   I/O, returns hits + diagnostics) so remote/object-backed DBs aren't forced
   through stale local `source_path`s; `retrieve_template_candidates(...)` owns
   loading (tolerant loader, aggregate load-failure warning) and returns
   `TemplateCandidate`s.
7. **Double-alignment cost is bounded by `max_candidates`, not `top_k`** — the
   featurizer TM-aligns every candidate before keeping `top_k` (so ~`max_candidates`
   alignments/query, plus the search's `rerank_top_k`). Corrected from the v1
   "top_k≈4" understatement.

Deferred per review: sequence-identity/date/cluster blocklist *computation*;
adaptive over-fetch; chain/model-selection consistency between DB-encode and
re-load (documented assumption: candidates re-load with default chain selection,
matching how the DB was built).
