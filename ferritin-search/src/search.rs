//! End-to-end search orchestrator.
//!
//! Ties the four prefilter sub-phases plus gapped alignment together
//! into a single API:
//!
//! 1. Reduce target sequences to a smaller alphabet (2.3c) and build a
//!    k-mer index over them (2.2).
//! 2. Per query: reduce, run diagonal-voting prefilter (2.3a, optionally
//!    with similar-k-mer expansion 2.3b), ungapped extension along the
//!    best diagonal (2.3d), Smith-Waterman gapped alignment (2.4) on
//!    the full alphabet.
//! 3. Sort hits by gapped alignment score; apply per-query result limit.
//!
//! [`SearchEngine`] is built once over a target corpus and reused across
//! many queries — that's the production usage pattern (one DB, many
//! queries) and matches upstream's `createindex` + `search` split.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::alphabet::Alphabet;
use crate::db::{DBReader, DbError};
use crate::gapped::{smith_waterman, GappedAlignment};
use crate::kmer::{KmerEncoder, KmerHit, KmerIndex, KmerIndexError, KmerLookup};
use crate::kmer_generator::widen_to_i32;
use crate::kmer_index_file::{
    write_kmi, KmerIndexFile, KmiReaderError, KmiWriterError, ReducerSnapshot,
};
use crate::matrix::SubstitutionMatrix;
use crate::prefilter::{diagonal_prefilter, PrefilterHit, PrefilterOptions};
use crate::reduced_alphabet::ReducedAlphabet;
use crate::sequence::Sequence;
use crate::ungapped::ungapped_alignment;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("index build failed: {0}")]
    IndexBuild(#[from] KmerIndexError),
    #[error("reduced-alphabet construction failed: invalid reduce_to or unknown index")]
    BadReduction,
}

/// Error type for [`SearchEngine::build_from_mmseqs_db`].
///
/// Distinct from [`SearchError`] because opening the DB and building the
/// engine are independently fallible and callers may want to handle them
/// separately (e.g. "missing DB" vs "bad k value").
#[derive(Debug, Error)]
pub enum BuildFromDbError {
    #[error("mmseqs DB open failed: {0}")]
    DbOpen(#[from] DbError),
    #[error("search engine build failed: {0}")]
    Build(#[from] SearchError),
    #[error("on-disk k-mer index open failed: {0}")]
    KmiOpen(#[from] KmiReaderError),
    #[error("on-disk k-mer index write failed: {0}")]
    KmiWrite(#[from] KmiWriterError),
    #[error(
        "kmer index parameter mismatch: {field} on disk = {on_disk}, \
         engine expects {expected}"
    )]
    KmiParamMismatch {
        field: &'static str,
        on_disk: u64,
        expected: u64,
    },
    #[error(
        "kmer index reducer mapping differs from the engine's reducer. \
         The substitution matrix used at build time and the matrix \
         passed to open must produce the same full-to-reduced equivalence \
         classes; different matrices at the same reduced_size do not."
    )]
    KmiReducerMismatch,
    #[error(
        "DB has {found} entries; the sorted-keys engine caps entries at \
         {limit} (u32 position index). Bump TargetSource position type \
         to u64 before loading a DB this large."
    )]
    TooManyTargets { found: u64, limit: u64 },
    #[error(
        "DB key {key} appears more than once in the index; the engine \
         requires unique keys. Rebuild the DB via `mmseqs createdb` (or \
         `fasta_to_mmseqs_db`) so each sequence gets a distinct key."
    )]
    DuplicateKey { key: u32 },
}

// Route KmerIndexError (raised by the in-place kmer build inside
// `build_from_mmseqs_db`) through the `Build` variant via `SearchError`'s
// existing `#[from] KmerIndexError` arm. Lets us use `?` directly on the
// kmer build without a manual map_err dance.
impl From<KmerIndexError> for BuildFromDbError {
    fn from(e: KmerIndexError) -> Self {
        BuildFromDbError::Build(SearchError::from(e))
    }
}

/// Tuning knobs for [`SearchEngine`].
///
/// Defaults mirror common MMseqs2 protein-search settings: k=6, BLOSUM62
/// at bit_factor=2, gap_open=-11, gap_extend=-1, alphabet reduction to
/// 13 letters. They're intentionally conservative — tighten / loosen for
/// your workload.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub k: usize,
    /// `Some(n)` reduces the alphabet to `n` letters before building
    /// the k-mer index (collapses 21^k offsets to n^k); `None` indexes
    /// in the full alphabet.
    pub reduce_to: Option<usize>,
    /// Score scaling for [`SubstitutionMatrix::to_integer_matrix`].
    pub bit_factor: f32,
    /// Drop prefilter candidates whose diagonal k-mer count falls below this.
    pub diagonal_score_threshold: u32,
    /// Cap the number of prefilter candidates passed to ungapped + gapped.
    pub max_prefilter_hits: Option<usize>,
    /// Affine gap penalties for Smith-Waterman.
    pub gap_open: i32,
    pub gap_extend: i32,
    /// Drop final hits whose gapped alignment score falls below this.
    pub min_score: i32,
    /// Cap the number of hits returned per query.
    pub max_results: Option<usize>,
    /// Use the GPU-batched dispatch path when the `cuda` feature is
    /// compiled in and a device is detected at runtime. `false` forces
    /// the CPU path even on GPU hosts — useful for debugging or
    /// reproducing exact upstream ordering. Ignored when the feature
    /// is not compiled.
    pub use_gpu: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            k: 6,
            reduce_to: Some(13),
            bit_factor: 2.0,
            diagonal_score_threshold: 0,
            max_prefilter_hits: Some(1000),
            gap_open: -11,
            gap_extend: -1,
            min_score: 0,
            max_results: None,
            use_gpu: true,
        }
    }
}

/// One search hit: a target sequence and the full chain of intermediate
/// scores for it.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub target_id: u32,
    pub prefilter_score: u32,
    pub best_diagonal: i32,
    pub ungapped_score: i32,
    pub alignment: GappedAlignment,
}

/// Where the engine looks up target sequences at query time.
///
/// The `Db` variant keeps the DB memory-mapped (via [`DBReader`]) and
/// encodes payloads on demand, so a full UniRef50 doesn't require the
/// ~12 GB of resident in-memory target bytes that `InMemory` would.
/// The `InMemory` variant is the traditional back-compat path for
/// `SearchEngine::build(targets, ...)` — small corpora, tests, and
/// callers that already have `Sequence` objects in hand.
enum TargetSource {
    Db {
        db: DBReader,
        /// Keys from `db.index`, sorted ascending. Binary search on
        /// this at query time yields the lookup position.
        sorted_keys: Vec<u32>,
        /// Parallel to `sorted_keys`: `sorted_to_db_pos[i]` is the
        /// position in `db.index` whose entry has key `sorted_keys[i]`.
        /// u32 because we cap at ~4G entries (UniRef50 ≈ 50M) — saves
        /// half the memory vs a usize index.
        sorted_to_db_pos: Vec<u32>,
    },
    InMemory(HashMap<u32, Vec<u8>>),
}

impl TargetSource {
    /// Build the `Db` variant, sorting keys so the engine can use
    /// binary search instead of a `HashMap<u32, usize>`. At UniRef50
    /// scale this drops ~1 GB of resident RAM relative to the hashmap
    /// (400 MB for two parallel `Vec<u32>`s vs ~1.7 GB for a
    /// hashmap's bucket + entry + metadata overhead).
    ///
    /// Validates two invariants the binary-search lookup relies on:
    ///  - `db.index.len() <= u32::MAX` (positions fit in `u32` without
    ///    silent truncation).
    ///  - Every key in `db.index` is unique (otherwise binary_search
    ///    returns an arbitrary match, making per-query results
    ///    nondeterministic between builds).
    fn new_db(db: DBReader) -> Result<Self, BuildFromDbError> {
        let n = db.index.len();
        if n > u32::MAX as usize {
            return Err(BuildFromDbError::TooManyTargets {
                found: n as u64,
                limit: u32::MAX as u64,
            });
        }
        let mut positions: Vec<u32> = (0..n as u32).collect();
        positions.sort_unstable_by_key(|&i| db.index[i as usize].key);
        let sorted_keys: Vec<u32> = positions
            .iter()
            .map(|&i| db.index[i as usize].key)
            .collect();
        // Reject duplicates: consecutive equal entries in sorted_keys
        // mean two DB entries share a key, which would make `fetch`
        // return whichever position binary_search lands on —
        // nondeterministic, and different between runs that hash into
        // different sort orderings.
        for w in sorted_keys.windows(2) {
            if w[0] == w[1] {
                return Err(BuildFromDbError::DuplicateKey { key: w[0] });
            }
        }
        Ok(Self::Db {
            db,
            sorted_keys,
            sorted_to_db_pos: positions,
        })
    }

    /// Encoded target bytes for `seq_id`, or `None` if the seq_id isn't
    /// in the corpus. InMemory yields a borrowed slice; Db yields an
    /// owned `Vec<u8>` because the encoded form doesn't live in the
    /// engine — `Sequence::from_ascii` runs per call.
    fn fetch(&self, seq_id: u32, alphabet: &Alphabet) -> Option<Cow<'_, [u8]>> {
        match self {
            Self::Db {
                db,
                sorted_keys,
                sorted_to_db_pos,
            } => {
                // Binary search for `seq_id` in the sorted keys array;
                // fall back to None for absent keys. Duplicate keys in
                // a DB are pathological but would simply resolve to
                // whichever the binary search lands on.
                let sorted_idx = sorted_keys.binary_search(&seq_id).ok()?;
                let db_pos = sorted_to_db_pos[sorted_idx] as usize;
                let entry = &db.index[db_pos];
                let payload = db.get_payload(entry);
                Some(Cow::Owned(
                    Sequence::from_ascii(alphabet.clone(), payload).data,
                ))
            }
            Self::InMemory(map) => map.get(&seq_id).map(|v| Cow::Borrowed(v.as_slice())),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Db { db, .. } => db.len(),
            Self::InMemory(map) => map.len(),
        }
    }
}

/// Where the engine gets its k-mer postings from.
///
/// Mirrors [`TargetSource`]: `InMemory` for `SearchEngine::build` and
/// small corpora, `OnDisk` for archive-scale corpora that can't afford
/// ~100 GB of resident k-mer postings. On the `OnDisk` path the file
/// is memory-mapped; only buckets touched by a given query's prefilter
/// get paged in.
pub enum KmerIndexStorage {
    InMemory(KmerIndex),
    OnDisk(KmerIndexFile),
}

impl KmerLookup for KmerIndexStorage {
    fn encoder(&self) -> &KmerEncoder {
        match self {
            Self::InMemory(k) => k.encoder(),
            Self::OnDisk(k) => k.encoder(),
        }
    }
    fn for_each_hit<F: FnMut(KmerHit)>(&self, hash: u64, f: F) {
        match self {
            Self::InMemory(k) => k.for_each_hit(hash, f),
            Self::OnDisk(k) => k.for_each_hit(hash, f),
        }
    }
}

/// Pre-built search engine over a fixed target corpus.
///
/// Construction does the expensive work once: alphabet reduction,
/// k-mer index build, integer score matrix conversion, and (for the
/// in-memory path) a `seq_id → encoded bytes` map.
pub struct SearchEngine {
    /// Source of target bytes at query time. See [`TargetSource`].
    targets: TargetSource,
    /// Source of k-mer postings at query time. See [`KmerIndexStorage`].
    index: KmerIndexStorage,
    /// Flat alphabet_size² i32 score matrix for ungapped + gapped.
    matrix_int: Vec<i32>,
    full_alphabet_size: usize,
    /// Reducer used for query encoding before prefilter; `None` means the
    /// k-mer index is in the full alphabet.
    reducer: Option<ReducedAlphabet>,
    /// Skip index passed to k-mer iteration (X in reduced or full space).
    skip_idx: u8,
    opts: SearchOptions,
    /// Retained for on-demand encoding of DB-backed targets at query
    /// time. For InMemory-backed engines the field is unused but kept
    /// so the API is storage-agnostic.
    alphabet: Alphabet,
}

impl SearchEngine {
    /// Build the engine over a target corpus held in memory.
    ///
    /// `targets` is consumed and materialized into an owned
    /// `HashMap<u32, Vec<u8>>`. For archive-scale corpora prefer
    /// [`SearchEngine::build_from_mmseqs_db`], which holds target bytes
    /// in a memory-mapped DB and encodes them on demand at query time.
    pub fn build(
        targets: impl IntoIterator<Item = (u32, Sequence)>,
        matrix: &SubstitutionMatrix,
        alphabet: Alphabet,
        opts: SearchOptions,
    ) -> Result<Self, SearchError> {
        let full_alphabet_size = alphabet.size();
        let x_full = alphabet.encode(b'X');

        // Materialize targets into the lookup map and a flat list for indexing.
        let mut targets_full: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut order: Vec<(u32, Vec<u8>)> = Vec::new();
        for (id, seq) in targets {
            targets_full.insert(id, seq.data.clone());
            order.push((id, seq.data));
        }

        let (reducer, kmer_alphabet_size, skip_idx, indexed_targets) =
            Self::prepare_index_inputs(matrix, &opts, full_alphabet_size, x_full, order)?;

        let encoder = KmerEncoder::new(kmer_alphabet_size as u32, opts.k);
        let pairs: Vec<(u32, &[u8])> = indexed_targets
            .iter()
            .map(|(id, s)| (*id, s.as_slice()))
            .collect();
        let index = KmerIndex::build(encoder, pairs, skip_idx)?;

        let matrix_int = widen_to_i32(&matrix.to_integer_matrix(opts.bit_factor, 0.0));

        Ok(Self {
            targets: TargetSource::InMemory(targets_full),
            index: KmerIndexStorage::InMemory(index),
            matrix_int,
            full_alphabet_size,
            reducer,
            skip_idx,
            opts,
            alphabet,
        })
    }

    /// Shared prep between `build` and `build_from_mmseqs_db`: resolves
    /// the optional reduced alphabet and produces the `(seq_id, encoded
    /// bytes)` pairs the k-mer index consumes. Extracted to keep the
    /// two constructors in lock-step on reducer semantics.
    fn prepare_index_inputs(
        matrix: &SubstitutionMatrix,
        opts: &SearchOptions,
        full_alphabet_size: usize,
        x_full: u8,
        encoded_full: Vec<(u32, Vec<u8>)>,
    ) -> Result<(Option<ReducedAlphabet>, usize, u8, Vec<(u32, Vec<u8>)>), SearchError> {
        let reducer_opt = match opts.reduce_to {
            Some(r) => Some(
                ReducedAlphabet::from_matrix(matrix, r, Some(x_full))
                    .ok_or(SearchError::BadReduction)?,
            ),
            None => None,
        };
        // Move `reducer` into the match by value so the borrow-for-reduce
        // ends before we re-wrap and return it. The equivalent `match
        // &reducer_opt { Some(r) => ... Ok((reducer_opt, ...)) }`
        // doesn't type-check: the `r` borrow is live when we try to move
        // reducer_opt into the tuple.
        match reducer_opt {
            Some(r) => {
                let skip = r.unknown_reduced_idx.ok_or(SearchError::BadReduction)?;
                let reduced_size = r.reduced_size;
                let reduced: Vec<(u32, Vec<u8>)> = encoded_full
                    .iter()
                    .map(|(id, s)| (*id, r.reduce_sequence(s)))
                    .collect();
                Ok((Some(r), reduced_size, skip, reduced))
            }
            None => Ok((None, full_alphabet_size, x_full, encoded_full)),
        }
    }

    /// Build the engine from an on-disk MMseqs2-compatible DB.
    ///
    /// Opens the DB at `prefix` (anything [`crate::db::DBReader`] accepts
    /// — e.g. output of `mmseqs createdb` or ferritin-search's own
    /// [`crate::db::DBWriter`]) and builds the k-mer index against it.
    /// The DB is kept memory-mapped for the engine's lifetime; target
    /// bytes are encoded on demand at query time rather than duplicated
    /// into an in-memory `HashMap<u32, Vec<u8>>`.
    ///
    /// Peak resident memory during build:
    ///  - DB mmap (paged in lazily)
    ///  - Transient encoded-payloads buffer (dropped after k-mer index build)
    ///  - K-mer index (retained for the engine's lifetime — Phase 3 will
    ///    move this to an on-disk mmap'd format)
    ///
    /// Peak resident memory during search:
    ///  - DB mmap + k-mer index, plus a per-hit encoded buffer that's
    ///    freed at the end of each loop iteration.
    pub fn build_from_mmseqs_db(
        prefix: impl AsRef<Path>,
        matrix: &SubstitutionMatrix,
        alphabet: Alphabet,
        opts: SearchOptions,
    ) -> Result<Self, BuildFromDbError> {
        let reader = DBReader::open(prefix)?;

        let full_alphabet_size = alphabet.size();
        let x_full = alphabet.encode(b'X');

        // Transient: encode every payload once for the k-mer build. Drops
        // at the end of this fn so the HashMap<u32, Vec<u8>>-equivalent
        // peak cost is bounded to construction time only.
        let encoded_full: Vec<(u32, Vec<u8>)> = reader
            .index
            .iter()
            .map(|entry| {
                let payload = reader.get_payload(entry);
                (
                    entry.key,
                    Sequence::from_ascii(alphabet.clone(), payload).data,
                )
            })
            .collect();

        let (reducer, kmer_alphabet_size, skip_idx, indexed_targets) =
            Self::prepare_index_inputs(matrix, &opts, full_alphabet_size, x_full, encoded_full)?;

        let encoder = KmerEncoder::new(kmer_alphabet_size as u32, opts.k);
        let pairs: Vec<(u32, &[u8])> = indexed_targets
            .iter()
            .map(|(id, s)| (*id, s.as_slice()))
            .collect();
        let index = KmerIndex::build(encoder, pairs, skip_idx)?;

        // Drop the encoded buffer — reclaimed before the engine returns.
        drop(indexed_targets);

        let matrix_int = widen_to_i32(&matrix.to_integer_matrix(opts.bit_factor, 0.0));

        Ok(Self {
            targets: TargetSource::new_db(reader)?,
            index: KmerIndexStorage::InMemory(index),
            matrix_int,
            full_alphabet_size,
            reducer,
            skip_idx,
            opts,
            alphabet,
        })
    }

    /// Write the engine's current in-memory k-mer index to `path` in
    /// `.kmi` format.
    ///
    /// Errors if the engine was constructed with an already-on-disk
    /// index — nothing to re-serialize in that case. Mostly useful
    /// paired with `build` / `build_from_mmseqs_db` to persist the
    /// index for later `open_from_mmseqs_db_with_kmi` calls, so the
    /// expensive k-mer build only runs once per corpus.
    pub fn write_kmer_index(&self, path: impl AsRef<Path>) -> Result<(), KmiWriterError> {
        match &self.index {
            KmerIndexStorage::InMemory(idx) => write_kmi(idx, self.reducer.as_ref(), path),
            KmerIndexStorage::OnDisk(_) => Err(KmiWriterError::Io(std::io::Error::other(
                "engine already backs its k-mer index on disk; nothing to serialize",
            ))),
        }
    }

    /// Open an engine backed by a memory-mapped DB + pre-built `.kmi`.
    ///
    /// Neither the DB nor the k-mer postings are loaded into RAM; both
    /// mmap and page in on demand. Peak resident memory is bounded by
    /// whatever buckets a query's prefilter touches plus per-hit
    /// encoded scratch. Ideal for UniRef50-scale corpora where a
    /// build-from-scratch would budget ~100 GB RAM for the k-mer
    /// index alone.
    ///
    /// Consistency checks on the `.kmi`:
    ///  - `kmer_size == opts.k`
    ///  - `alphabet_size == reducer.reduced_size` if a reducer is
    ///    configured, or `alphabet.size()` otherwise.
    ///
    /// Mismatch raises `BuildFromDbError::KmiParamMismatch`.
    pub fn open_from_mmseqs_db_with_kmi(
        db_prefix: impl AsRef<Path>,
        kmi_path: impl AsRef<Path>,
        matrix: &SubstitutionMatrix,
        alphabet: Alphabet,
        opts: SearchOptions,
    ) -> Result<Self, BuildFromDbError> {
        let reader = DBReader::open(db_prefix)?;
        let kmi = KmerIndexFile::open(kmi_path)?;

        let full_alphabet_size = alphabet.size();
        let x_full = alphabet.encode(b'X');

        // Same reducer + skip_idx logic as the in-memory path, without
        // the encode-targets step — we trust the .kmi was built with a
        // matching reducer and verify via alphabet_size below.
        let reducer: Option<ReducedAlphabet> = match opts.reduce_to {
            Some(r) => Some(
                ReducedAlphabet::from_matrix(matrix, r, Some(x_full))
                    .ok_or(SearchError::BadReduction)?,
            ),
            None => None,
        };
        let (expected_alphabet_size, skip_idx) = match &reducer {
            Some(r) => (
                r.reduced_size,
                r.unknown_reduced_idx.ok_or(SearchError::BadReduction)?,
            ),
            None => (full_alphabet_size, x_full),
        };

        if kmi.kmer_size() != opts.k {
            return Err(BuildFromDbError::KmiParamMismatch {
                field: "kmer_size",
                on_disk: kmi.kmer_size() as u64,
                expected: opts.k as u64,
            });
        }
        if kmi.alphabet_size() as usize != expected_alphabet_size {
            return Err(BuildFromDbError::KmiParamMismatch {
                field: "alphabet_size",
                on_disk: kmi.alphabet_size() as u64,
                expected: expected_alphabet_size as u64,
            });
        }
        // alphabet_size alone is insufficient: ReducedAlphabet::from_matrix
        // is matrix-dependent and two different matrices at the same
        // reduced_size can produce different full_to_reduced mappings.
        // Compare the full reducer snapshot so incompatible indices are
        // refused at open time instead of silently returning wrong hits.
        let expected_snap = ReducerSnapshot::from_reducer(reducer.as_ref());
        if kmi.reducer() != &expected_snap {
            return Err(BuildFromDbError::KmiReducerMismatch);
        }

        let matrix_int = widen_to_i32(&matrix.to_integer_matrix(opts.bit_factor, 0.0));

        Ok(Self {
            targets: TargetSource::new_db(reader)?,
            index: KmerIndexStorage::OnDisk(kmi),
            matrix_int,
            full_alphabet_size,
            reducer,
            skip_idx,
            opts,
            alphabet,
        })
    }

    /// Run a single query against the indexed corpus.
    ///
    /// Returns hits sorted by gapped alignment score descending.
    ///
    /// Dispatches to [`SearchEngine::search_gpu`] when the `cuda` feature
    /// is compiled in, `opts.use_gpu` is `true`, and a device is
    /// detected at runtime. Silent CPU fallback otherwise.
    pub fn search(&self, query: &Sequence) -> Vec<SearchHit> {
        let query_for_prefilter = match &self.reducer {
            Some(r) => r.reduce_sequence(&query.data),
            None => query.data.clone(),
        };

        let prefilter_hits = diagonal_prefilter(
            &self.index,
            &query_for_prefilter,
            self.skip_idx,
            &PrefilterOptions {
                score_threshold: self.opts.diagonal_score_threshold,
                max_hits: self.opts.max_prefilter_hits,
                exclude_self: None,
            },
        );

        #[cfg(feature = "cuda")]
        {
            if self.opts.use_gpu && crate::gpu::is_available() {
                return self.search_gpu(&query.data, &prefilter_hits);
            }
        }

        self.search_cpu(&query.data, &prefilter_hits)
    }

    /// CPU reference path. Also the fallback when GPU dispatch errors
    /// mid-flight. Parity target for the GPU path — same inputs must
    /// produce the same `SearchHit` ordering.
    fn search_cpu(&self, query: &[u8], prefilter_hits: &[PrefilterHit]) -> Vec<SearchHit> {
        let mut results: Vec<SearchHit> = Vec::new();
        for ph in prefilter_hits {
            let target_cow = match self.targets.fetch(ph.seq_id, &self.alphabet) {
                Some(t) => t,
                None => continue,
            };
            let target: &[u8] = &target_cow;

            let ungapped = match ungapped_alignment(
                query,
                target,
                ph.best_diagonal,
                &self.matrix_int,
                self.full_alphabet_size,
            ) {
                Some(u) => u,
                None => continue,
            };

            let gapped = match smith_waterman(
                query,
                target,
                &self.matrix_int,
                self.full_alphabet_size,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Some(a) if a.score >= self.opts.min_score => a,
                _ => continue,
            };

            results.push(SearchHit {
                target_id: ph.seq_id,
                prefilter_score: ph.diagonal_score,
                best_diagonal: ph.best_diagonal,
                ungapped_score: ungapped.score,
                alignment: gapped,
            });
        }

        results.sort_by_key(|r| std::cmp::Reverse(r.alignment.score));
        if let Some(limit) = self.opts.max_results {
            results.truncate(limit);
        }
        results
    }

    /// GPU-batched path: dispatch ungapped + Smith-Waterman scoring
    /// across all prefilter candidates in two batched GPU calls, then
    /// run CPU `smith_waterman` only on the surviving top-K to recover
    /// CIGAR traceback (the GPU score-batch kernel deliberately skips
    /// traceback — see [`crate::gpu::sw`]).
    ///
    /// On any GPU infrastructure error, falls back to the full CPU
    /// path for this query so correctness is never at risk.
    #[cfg(feature = "cuda")]
    fn search_gpu(&self, query: &[u8], prefilter_hits: &[PrefilterHit]) -> Vec<SearchHit> {
        use crate::gpu::{diagonal, sw};

        struct Candidate<'a> {
            /// Owned for DB-backed targets (encoded on-demand), borrowed
            /// for InMemory-backed targets. Either way, `&*c.target`
            /// yields `&[u8]` for the GPU kernel.
            target: Cow<'a, [u8]>,
            seq_id: u32,
            prefilter_score: u32,
            best_diagonal: i32,
        }

        let mut candidates: Vec<Candidate<'_>> = Vec::with_capacity(prefilter_hits.len());
        for ph in prefilter_hits {
            if let Some(target) = self.targets.fetch(ph.seq_id, &self.alphabet) {
                candidates.push(Candidate {
                    target,
                    seq_id: ph.seq_id,
                    prefilter_score: ph.diagonal_score,
                    best_diagonal: ph.best_diagonal,
                });
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }

        let pairs: Vec<(&[u8], i32)> = candidates
            .iter()
            .map(|c| (&*c.target, c.best_diagonal))
            .collect();

        // Batched ungapped extension. Err → full CPU fallback.
        let ungapped = match diagonal::ungapped_alignment_batch_gpu(
            query,
            &pairs,
            &self.matrix_int,
            self.full_alphabet_size,
        ) {
            Ok(u) => u,
            Err(e) => {
                eprintln!("[ferritin-search] GPU ungapped batch failed; CPU fallback: {e:#}");
                return self.search_cpu(query, prefilter_hits);
            }
        };

        // Keep only ungapped survivors (CPU path's "skip on None" semantic).
        let mut surviving: Vec<(usize, i32)> = Vec::with_capacity(candidates.len());
        for (i, u) in ungapped.iter().enumerate() {
            if let Some(hit) = u {
                surviving.push((i, hit.score));
            }
        }
        if surviving.is_empty() {
            return Vec::new();
        }

        let surviving_targets: Vec<&[u8]> = surviving
            .iter()
            .map(|(i, _)| &*candidates[*i].target)
            .collect();

        // Batched SW score+endpoint. Dispatch by query length:
        //   query_len ≤ 256              → warp singletile (4.5a), fastest
        //   256 < query_len ≤ 2048       → warp multitile (4.5b)
        //   query_len > 2048             → thread-per-pair kernel (sw)
        //
        // All three are batched GPU kernels; any infrastructure error
        // falls the whole query back to the CPU path for correctness.
        use crate::gpu::pssm_sw_warp::{self, MAX_QUERY_LEN as SINGLETILE_MAX_Q};
        use crate::gpu::pssm_sw_warp_multitile::{self, MAX_QUERY_LEN as MULTITILE_MAX_Q};
        use crate::pssm::Pssm;
        let sw_scores = if query.len() <= SINGLETILE_MAX_Q {
            let pssm = Pssm::build(query, &self.matrix_int, self.full_alphabet_size);
            match pssm_sw_warp::pssm_sw_warp_batch_gpu(
                &pssm,
                &surviving_targets,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "[ferritin-search] GPU warp singletile SW failed; CPU fallback: {e:#}"
                    );
                    return self.search_cpu(query, prefilter_hits);
                }
            }
        } else if query.len() <= MULTITILE_MAX_Q {
            let pssm = Pssm::build(query, &self.matrix_int, self.full_alphabet_size);
            match pssm_sw_warp_multitile::pssm_sw_warp_multitile_batch_gpu(
                &pssm,
                &surviving_targets,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "[ferritin-search] GPU warp multitile SW failed; CPU fallback: {e:#}"
                    );
                    return self.search_cpu(query, prefilter_hits);
                }
            }
        } else {
            match sw::smith_waterman_score_batch_gpu(
                query,
                &surviving_targets,
                &self.matrix_int,
                self.full_alphabet_size,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[ferritin-search] GPU SW batch failed; CPU fallback: {e:#}");
                    return self.search_cpu(query, prefilter_hits);
                }
            }
        };

        // Threshold + rank on GPU scores so the expensive CPU traceback
        // only runs on the final top-K. This is the whole point of the
        // GPU path: one cheap batched score sweep, then targeted CPU
        // work — mirrors how upstream MMseqs2 hands off to CPU for output.
        let mut ranked: Vec<(usize, i32, i32)> = Vec::with_capacity(surviving.len()); // (idx_in_surviving, sw_score, ungapped_score)
        for (pos, (_, ungapped_score)) in surviving.iter().enumerate() {
            let sw_score = sw_scores[pos].score;
            if sw_score >= self.opts.min_score {
                ranked.push((pos, sw_score, *ungapped_score));
            }
        }
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        if let Some(limit) = self.opts.max_results {
            ranked.truncate(limit);
        }

        // CPU traceback only on the retained hits.
        let mut results: Vec<SearchHit> = Vec::with_capacity(ranked.len());
        for (pos, _, ungapped_score) in ranked {
            let (cand_idx, _) = surviving[pos];
            let c = &candidates[cand_idx];
            let gapped = match smith_waterman(
                query,
                &c.target,
                &self.matrix_int,
                self.full_alphabet_size,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Some(a) if a.score >= self.opts.min_score => a,
                _ => continue,
            };
            results.push(SearchHit {
                target_id: c.seq_id,
                prefilter_score: c.prefilter_score,
                best_diagonal: c.best_diagonal,
                ungapped_score,
                alignment: gapped,
            });
        }
        results
    }

    /// Number of targets indexed.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Convenience: run [`SearchEngine::search`] for the given query and
    /// pipe the hits straight into [`crate::msa::assemble_msa`], returning
    /// an AF2-style MSA tensor bundle.
    ///
    /// Pre-materializes each hit's encoded target bytes into a local
    /// map so [`assemble_msa`]'s closure can borrow `&[u8]` for its
    /// lifetime. For DB-backed engines this is the only place that
    /// fetch()es once per hit; for InMemory engines it's a borrow.
    pub fn search_and_build_msa(
        &self,
        query: &Sequence,
        msa_opts: crate::msa::MsaOptions,
    ) -> crate::msa::MsaAssembly {
        let hits = self.search(query);
        let encoded: HashMap<u32, Vec<u8>> = hits
            .iter()
            .take(msa_opts.max_seqs)
            .filter_map(|h| {
                self.targets
                    .fetch(h.target_id, &self.alphabet)
                    .map(|c| (h.target_id, c.into_owned()))
            })
            .collect();
        crate::msa::assemble_msa(
            query,
            &hits,
            |id| encoded.get(&id).map(Vec::as_slice),
            msa_opts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha_and_matrix() -> (Alphabet, SubstitutionMatrix) {
        (Alphabet::protein(), SubstitutionMatrix::blosum62())
    }

    #[test]
    fn self_search_finds_self_as_top_hit() {
        // Three distinct proteins; querying any one of them must rank
        // that one as the top hit.
        let (alpha, m) = alpha_and_matrix();
        let seqs = [
            (1u32, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ".as_slice()),
            (2u32, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP".as_slice()),
            (3u32, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR".as_slice()),
        ];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            ..Default::default()
        };
        let engine = SearchEngine::build(targets.clone(), &m, alpha.clone(), opts).unwrap();
        for (qid, seq) in &targets {
            let hits = engine.search(seq);
            assert!(!hits.is_empty(), "query {qid} returned no hits");
            assert_eq!(
                hits[0].target_id, *qid,
                "query {qid}'s top hit was {} (expected self)",
                hits[0].target_id,
            );
        }
    }

    #[test]
    fn search_returns_hits_sorted_by_alignment_score_desc() {
        let (alpha, m) = alpha_and_matrix();
        let seqs = [
            (1u32, b"MKLVRQPSTNLKACDFGHIY".as_slice()),
            (2u32, b"WWWWWWWWWWWWWWWWWWWW".as_slice()),
        ];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            ..Default::default()
        };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();

        let query = Sequence::from_ascii(alpha, b"MKLVRQPSTNLKACDFGHIY");
        let hits = engine.search(&query);
        for w in hits.windows(2) {
            assert!(
                w[0].alignment.score >= w[1].alignment.score,
                "hits not sorted by gapped score desc",
            );
        }
    }

    #[test]
    fn search_respects_min_score_threshold() {
        let (alpha, m) = alpha_and_matrix();
        let seqs = [(1u32, b"MKLVRQPSTNL".as_slice())];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();

        // Self-search with reachable score → finds it.
        let opts1 = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            min_score: 0,
            ..Default::default()
        };
        let engine = SearchEngine::build(targets.clone(), &m, alpha.clone(), opts1).unwrap();
        let q = Sequence::from_ascii(alpha.clone(), b"MKLVRQPSTNL");
        assert!(!engine.search(&q).is_empty());

        // Same search with absurdly high threshold → no hits.
        let opts2 = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            min_score: 100_000,
            ..Default::default()
        };
        let engine2 = SearchEngine::build(targets, &m, alpha.clone(), opts2).unwrap();
        assert!(engine2.search(&q).is_empty());
    }

    #[test]
    fn search_respects_max_results_cap() {
        // Five identical targets with distinct ids → all should self-match
        // any query; max_results=2 keeps just the top 2.
        let (alpha, m) = alpha_and_matrix();
        let seq = b"MKLVRQPSTNLAACDF".as_slice();
        let targets: Vec<(u32, Sequence)> = (1..=5)
            .map(|id| (id as u32, Sequence::from_ascii(alpha.clone(), seq)))
            .collect();

        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            max_results: Some(2),
            ..Default::default()
        };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();
        let q = Sequence::from_ascii(alpha, seq);
        let hits = engine.search(&q);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn search_works_without_reduction() {
        // reduce_to: None → k-mer index lives in the full 21-letter
        // alphabet. Smaller k keeps table_size sane.
        let (alpha, m) = alpha_and_matrix();
        let targets: Vec<(u32, Sequence)> = vec![(
            1u32,
            Sequence::from_ascii(alpha.clone(), b"MKLVRQPSTNLAACDF"),
        )];
        let opts = SearchOptions {
            k: 3,
            reduce_to: None,
            ..Default::default()
        };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();
        let q = Sequence::from_ascii(alpha, b"MKLVRQPSTNLAACDF");
        let hits = engine.search(&q);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].target_id, 1);
    }

    /// End-to-end parity: GPU dispatch path returns the same hits in
    /// the same order as the CPU path. Skips when no GPU is present.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_search_matches_cpu_search() {
        if !crate::gpu::is_available() {
            eprintln!("SKIP gpu_search_matches_cpu_search: no GPU available");
            return;
        }
        let (alpha, m) = alpha_and_matrix();
        let seqs = [
            (1u32, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ".as_slice()),
            (2u32, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP".as_slice()),
            (3u32, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR".as_slice()),
            (4u32, b"MKLVRQPSTNLKACDFGHIYMKLVRQPSTNLKACDFGHIY".as_slice()),
            (5u32, b"WWWWWWWWWWWWWWWWWWWW".as_slice()),
        ];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();

        let opts_cpu = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            use_gpu: false,
            ..Default::default()
        };
        let opts_gpu = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            use_gpu: true,
            ..Default::default()
        };
        let engine_cpu = SearchEngine::build(targets.clone(), &m, alpha.clone(), opts_cpu).unwrap();
        let engine_gpu = SearchEngine::build(targets.clone(), &m, alpha.clone(), opts_gpu).unwrap();

        for (qid, seq) in &targets {
            let cpu = engine_cpu.search(seq);
            let gpu = engine_gpu.search(seq);
            assert_eq!(
                cpu.len(),
                gpu.len(),
                "query {qid}: CPU returned {} hits, GPU returned {}",
                cpu.len(),
                gpu.len(),
            );
            for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert_eq!(
                    c.target_id, g.target_id,
                    "query {qid} pos {i}: target_id CPU={} GPU={}",
                    c.target_id, g.target_id,
                );
                assert_eq!(
                    c.alignment.score, g.alignment.score,
                    "query {qid} pos {i}: score CPU={} GPU={}",
                    c.alignment.score, g.alignment.score,
                );
                assert_eq!(c.ungapped_score, g.ungapped_score);
                assert_eq!(c.alignment.query_end, g.alignment.query_end);
                assert_eq!(c.alignment.target_end, g.alignment.target_end);
            }
        }
    }

    #[test]
    fn build_from_mmseqs_db_round_trips_against_in_memory_build() {
        use crate::db::{DBWriter, Dbtype};
        use tempfile::tempdir;

        // Canonical DB payload format: bytes + trailing `\n\0` that
        // `DBWriter::write_entry` appends. `build_from_mmseqs_db` must
        // strip the `\0` (via `get_payload`) and let `Sequence::from_ascii`
        // drop the `\n`, producing the same encoded sequences as an
        // in-memory `SearchEngine::build` over identical inputs.
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("targets");
        let mut w = DBWriter::create(&prefix, Dbtype::AMINO_ACIDS).unwrap();
        let entries = [
            (1u32, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ".as_slice()),
            (2u32, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP".as_slice()),
            (3u32, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR".as_slice()),
        ];
        for (key, payload) in &entries {
            w.write_entry(*key, payload).unwrap();
        }
        w.finish().unwrap();

        let (alpha, m) = alpha_and_matrix();
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            ..Default::default()
        };

        let from_db = SearchEngine::build_from_mmseqs_db(&prefix, &m, alpha.clone(), opts.clone())
            .expect("build_from_mmseqs_db");

        let in_mem_targets: Vec<(u32, Sequence)> = entries
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();
        let in_mem = SearchEngine::build(in_mem_targets.clone(), &m, alpha.clone(), opts)
            .expect("SearchEngine::build");

        // Per-query parity: same top-hit target_id + gapped score for every input.
        for (qid, qseq) in &entries {
            let query = Sequence::from_ascii(alpha.clone(), qseq);
            let db_hits = from_db.search(&query);
            let mem_hits = in_mem.search(&query);
            assert_eq!(
                db_hits.len(),
                mem_hits.len(),
                "hit count mismatch for query {qid}",
            );
            if !db_hits.is_empty() {
                assert_eq!(db_hits[0].target_id, mem_hits[0].target_id);
                assert_eq!(db_hits[0].alignment.score, mem_hits[0].alignment.score);
            }
        }

        // Self-hit sanity: every DB-built query finds itself at rank 0.
        for (qid, qseq) in &entries {
            let q = Sequence::from_ascii(alpha.clone(), qseq);
            let hits = from_db.search(&q);
            assert!(!hits.is_empty(), "DB-built engine: query {qid} had no hits");
            assert_eq!(hits[0].target_id, *qid);
        }
    }

    #[test]
    fn build_from_mmseqs_db_surfaces_missing_db_as_db_open_error() {
        let (alpha, m) = alpha_and_matrix();
        let result = SearchEngine::build_from_mmseqs_db(
            "/nonexistent/path/to/mmseqs-db",
            &m,
            alpha,
            SearchOptions::default(),
        );
        match result {
            Err(BuildFromDbError::DbOpen(_)) => {}
            Err(e) => panic!("expected DbOpen error, got {e}"),
            Ok(_) => panic!("expected error for missing DB, got Ok"),
        }
    }

    #[test]
    fn open_from_mmseqs_db_with_kmi_parity_against_in_memory() {
        use crate::db::{DBWriter, Dbtype};
        use tempfile::tempdir;

        // Build an engine normally, snapshot its k-mer index to disk
        // via write_kmer_index, then reopen via the mmap path — every
        // per-query hit must match byte-for-byte (same target_id,
        // same scores, same ungapped + gapped endpoints).
        let dir = tempdir().unwrap();
        let db_prefix = dir.path().join("targets");
        let kmi_path = dir.path().join("targets.kmi");

        let entries = [
            (1u32, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ".as_slice()),
            (2u32, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP".as_slice()),
            (3u32, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR".as_slice()),
        ];
        let mut w = DBWriter::create(&db_prefix, Dbtype::AMINO_ACIDS).unwrap();
        for (key, payload) in &entries {
            w.write_entry(*key, payload).unwrap();
        }
        w.finish().unwrap();

        let (alpha, m) = alpha_and_matrix();
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            ..Default::default()
        };

        let in_mem =
            SearchEngine::build_from_mmseqs_db(&db_prefix, &m, alpha.clone(), opts.clone())
                .expect("build_from_mmseqs_db");
        in_mem
            .write_kmer_index(&kmi_path)
            .expect("write_kmer_index");

        let on_disk = SearchEngine::open_from_mmseqs_db_with_kmi(
            &db_prefix,
            &kmi_path,
            &m,
            alpha.clone(),
            opts,
        )
        .expect("open_from_mmseqs_db_with_kmi");

        for (_, qseq) in &entries {
            let query = Sequence::from_ascii(alpha.clone(), qseq);
            let mem_hits = in_mem.search(&query);
            let disk_hits = on_disk.search(&query);
            assert_eq!(mem_hits.len(), disk_hits.len());
            for (m_h, d_h) in mem_hits.iter().zip(disk_hits.iter()) {
                assert_eq!(m_h.target_id, d_h.target_id);
                assert_eq!(m_h.alignment.score, d_h.alignment.score);
                assert_eq!(m_h.ungapped_score, d_h.ungapped_score);
                assert_eq!(m_h.alignment.query_end, d_h.alignment.query_end);
                assert_eq!(m_h.alignment.target_end, d_h.alignment.target_end);
            }
        }
    }

    #[test]
    fn new_db_rejects_duplicate_keys() {
        // Duplicate keys in a DB make binary_search's match position
        // arbitrary, so `fetch` would return a different target
        // depending on sort stability. Engine refuses at construction
        // with a clear error pointing at the offending key.
        use crate::db::{DBWriter, Dbtype};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let prefix = dir.path().join("dup");
        let mut w = DBWriter::create(&prefix, Dbtype::AMINO_ACIDS).unwrap();
        w.write_entry(1, b"MKLVR").unwrap();
        w.write_entry(2, b"WWWWWW").unwrap();
        // Same key again → must be rejected on engine construction.
        w.write_entry(1, b"AAAAAA").unwrap();
        w.finish().unwrap();

        let (alpha, m) = alpha_and_matrix();
        let err = SearchEngine::build_from_mmseqs_db(
            &prefix,
            &m,
            alpha,
            SearchOptions {
                k: 3,
                reduce_to: Some(13),
                ..Default::default()
            },
        )
        .err()
        .expect("expected DuplicateKey");
        assert!(
            matches!(err, BuildFromDbError::DuplicateKey { key: 1 }),
            "expected DuplicateKey(1), got {err}",
        );
    }

    #[test]
    fn db_target_lookup_handles_out_of_order_keys() {
        // Phase 3c swapped HashMap<u32, usize> for a sorted-keys
        // binary search. If the sort or parallel-array wiring is
        // wrong, a DB whose keys weren't inserted in ascending order
        // would mis-resolve lookups. Build such a DB and assert every
        // original (key, payload) can be round-tripped via the engine.
        use crate::db::{DBWriter, Dbtype};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let prefix = dir.path().join("db");
        let mut w = DBWriter::create(&prefix, Dbtype::AMINO_ACIDS).unwrap();
        // Intentionally not sorted: 42, 7, 19, 3. Also a zero-value
        // key which older HashMap-based code quietly handled but new
        // sorted-array code must handle too.
        let entries: &[(u32, &[u8])] = &[
            (42, b"MKLVRQPSTNLKACDFGHIY"),
            (7, b"MNALVVKFGGTSVANAERFLR"),
            (19, b"WWWWWWWWWWWWWWWWWWWW"),
            (3, b"MEAFRKQLPCFRSGAQQVKEH"),
            (0, b"ACDEFGHIKLMNPQRSTVWY"),
        ];
        for (k, s) in entries {
            w.write_entry(*k, s).unwrap();
        }
        w.finish().unwrap();

        let (alpha, m) = alpha_and_matrix();
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            ..Default::default()
        };
        let engine = SearchEngine::build_from_mmseqs_db(&prefix, &m, alpha.clone(), opts)
            .expect("build_from_mmseqs_db");
        assert_eq!(engine.target_count(), entries.len());

        // Each (key, payload) must self-hit via search — a correct
        // binary-search lookup is the only way this passes on an
        // out-of-order DB.
        for (key, payload) in entries {
            let q = Sequence::from_ascii(alpha.clone(), payload);
            let hits = engine.search(&q);
            assert!(!hits.is_empty(), "query {key} returned no hits");
            assert_eq!(
                hits[0].target_id, *key,
                "query for key {key}: top hit was {} (expected self)",
                hits[0].target_id,
            );
        }

        // Absent keys must resolve to None (Vec::new hits).
        let missing = Sequence::from_ascii(alpha, b"CCCCCCCCCCCCCCCCCCCCCC");
        let hits = engine.search(&missing);
        for h in &hits {
            assert!(
                entries.iter().any(|(k, _)| *k == h.target_id),
                "search returned a hit for a key not in the DB: {}",
                h.target_id,
            );
        }
    }

    #[test]
    fn open_from_mmseqs_db_with_kmi_rejects_reducer_mapping_mismatch() {
        // Regression for the 2026-04-17 review: alphabet_size parity
        // alone is insufficient because ReducedAlphabet::from_matrix is
        // matrix-dependent — two matrices at the same reduced_size can
        // produce different full_to_reduced mappings. The open path
        // MUST compare the full reducer snapshot.
        //
        // Simulate an incompatible mapping by building normally, then
        // flipping one byte of the embedded reducer section on disk
        // before reopen. The engine's expected snapshot (built from
        // matrix + alphabet + opts) won't equal the tampered on-disk
        // snapshot, so KmiReducerMismatch must fire.
        use crate::db::{DBWriter, Dbtype};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_prefix = dir.path().join("db");
        let kmi_path = dir.path().join("db.kmi");

        let mut w = DBWriter::create(&db_prefix, Dbtype::AMINO_ACIDS).unwrap();
        w.write_entry(1, b"MKLVRQPSTNLKACDFGHIY").unwrap();
        w.finish().unwrap();

        let (alpha, m) = alpha_and_matrix();
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            ..Default::default()
        };
        let built = SearchEngine::build_from_mmseqs_db(&db_prefix, &m, alpha.clone(), opts.clone())
            .unwrap();
        built.write_kmer_index(&kmi_path).unwrap();

        // Tamper: open the .kmi file on disk and swap a byte in the
        // embedded full_to_reduced section. This simulates an index
        // built with a different matrix / reduction strategy.
        let mut bytes = std::fs::read(&kmi_path).unwrap();
        // Reducer section starts at offset 64 (after header). The
        // full_to_reduced body starts at byte 64 + 12 = 76. Flip one
        // byte to force a mapping difference.
        let body_start = (crate::kmer_index_file::KMI_HEADER_SIZE
            + crate::kmer_index_file::REDUCER_SECTION_PROLOGUE) as usize;
        let original = bytes[body_start];
        bytes[body_start] = original.wrapping_add(1);
        std::fs::write(&kmi_path, &bytes).unwrap();

        let err =
            SearchEngine::open_from_mmseqs_db_with_kmi(&db_prefix, &kmi_path, &m, alpha, opts)
                .err()
                .expect("expected KmiReducerMismatch");
        assert!(
            matches!(err, BuildFromDbError::KmiReducerMismatch),
            "expected KmiReducerMismatch, got {err}",
        );
    }

    #[test]
    fn open_from_mmseqs_db_with_kmi_rejects_alphabet_mismatch() {
        use crate::db::{DBWriter, Dbtype};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_prefix = dir.path().join("db");
        let kmi_path = dir.path().join("db.kmi");

        let mut w = DBWriter::create(&db_prefix, Dbtype::AMINO_ACIDS).unwrap();
        w.write_entry(1, b"MKLVRQPSTNLKACDFGHIY").unwrap();
        w.finish().unwrap();

        let (alpha, m) = alpha_and_matrix();
        // Build with reduce_to=13.
        let built = SearchEngine::build_from_mmseqs_db(
            &db_prefix,
            &m,
            alpha.clone(),
            SearchOptions {
                k: 3,
                reduce_to: Some(13),
                ..Default::default()
            },
        )
        .unwrap();
        built.write_kmer_index(&kmi_path).unwrap();

        // Try to open with reduce_to=None → kmi alphabet_size=13, expected=21. Must raise.
        let err = SearchEngine::open_from_mmseqs_db_with_kmi(
            &db_prefix,
            &kmi_path,
            &m,
            alpha,
            SearchOptions {
                k: 3,
                reduce_to: None,
                ..Default::default()
            },
        )
        .err()
        .expect("expected KmiParamMismatch");
        assert!(
            matches!(
                err,
                BuildFromDbError::KmiParamMismatch {
                    field: "alphabet_size",
                    ..
                }
            ),
            "expected alphabet_size mismatch, got {err}",
        );
    }

    #[test]
    fn unrelated_query_returns_no_hits_at_reasonable_threshold() {
        let (alpha, m) = alpha_and_matrix();
        let targets = vec![(
            1u32,
            Sequence::from_ascii(alpha.clone(), b"AAAAAAAAAAAAAAAAAAAA"),
        )];
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            min_score: 50, // generous; A-A under BLOSUM62 won't approach this
            ..Default::default()
        };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();
        let q = Sequence::from_ascii(alpha, b"WWWWWWWWWWWWWWWWWWWW");
        assert!(engine.search(&q).is_empty());
    }
}
