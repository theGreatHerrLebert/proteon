//! K-mer encoding and CSR-style k-mer index.
//!
//! The k-mer hash is upstream's polynomial encoding from
//! `src/prefiltering/Indexer.h`:
//!
//! ```text
//! h(w[0..k]) = sum_{i=0..k}  w[i] * alphabet_size^i
//! ```
//!
//! So with alphabet indices in `0..alphabet_size` and a fixed `k`, every
//! k-mer maps to a unique integer in `0..alphabet_size^k`. The k-mer
//! index (`KmerIndex`) stores, for every possible k-mer, the list of
//! `(seq_id, position)` pairs where that k-mer occurs across a corpus —
//! laid out CSR-style so lookups are O(1) and scans are cache-friendly.
//!
//! Phase 2.2 scope: plain k-mers, no similar-k-mer expansion, no reduced
//! alphabet. Those arrive in phase 2.3 alongside the prefilter.

use thiserror::Error;

use crate::sequence::Sequence;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KmerIndexError {
    /// A k-mer start position in the indexed corpus exceeds the `u16`
    /// range of [`KmerHit::pos`]. Upstream's `IndexEntryLocal` uses the
    /// same 2-byte position field, so this port inherits the limit — but
    /// unlike upstream we refuse to silently truncate. Hit coordinates
    /// must be exact for the prefilter → alignment handoff, so long
    /// sequences (nucleotide contigs, titin-scale proteins) need to be
    /// chunked or indexed with a different entry layout before this
    /// builder will accept them.
    #[error(
        "k-mer position {pos} in sequence {seq_id} exceeds u16::MAX ({limit}); \
         sequence must be chunked before indexing"
    )]
    PositionOverflow {
        seq_id: u32,
        pos: usize,
        limit: usize,
    },
}

pub type Result<T> = std::result::Result<T, KmerIndexError>;

/// Polynomial k-mer hash over a fixed alphabet size and k.
#[derive(Debug, Clone)]
pub struct KmerEncoder {
    alphabet_size: u32,
    kmer_size: usize,
    /// `powers[i] = alphabet_size^i`, stored as u64 to cover up to
    /// `alphabet_size^kmer_size`.
    powers: Vec<u64>,
    /// Total number of possible k-mers: `alphabet_size^kmer_size`.
    table_size: u64,
}

impl KmerEncoder {
    /// Build an encoder for the given alphabet size and k.
    ///
    /// Panics if `kmer_size == 0`, or if `alphabet_size^kmer_size` would
    /// overflow u64 (pathologically large values — a 21-letter alphabet
    /// gets up to k=14 within u64 before overflow, 5-letter to k=27).
    pub fn new(alphabet_size: u32, kmer_size: usize) -> Self {
        assert!(kmer_size > 0, "kmer_size must be >= 1");
        assert!(alphabet_size >= 2, "alphabet_size must be >= 2");

        let mut powers = Vec::with_capacity(kmer_size);
        let mut p: u64 = 1;
        powers.push(p);
        for _ in 1..kmer_size {
            p = p
                .checked_mul(alphabet_size as u64)
                .expect("alphabet_size^kmer_size overflow");
            powers.push(p);
        }
        let table_size = p
            .checked_mul(alphabet_size as u64)
            .expect("alphabet_size^kmer_size overflow");

        Self {
            alphabet_size,
            kmer_size,
            powers,
            table_size,
        }
    }

    pub fn alphabet_size(&self) -> u32 {
        self.alphabet_size
    }

    pub fn kmer_size(&self) -> usize {
        self.kmer_size
    }

    /// Total number of possible k-mer hash values: `alphabet_size^kmer_size`.
    pub fn table_size(&self) -> u64 {
        self.table_size
    }

    /// Encode one k-letter window. `window.len()` must equal `kmer_size`.
    ///
    /// Every byte must satisfy `b < alphabet_size`; caller is responsible
    /// for prior alphabet encoding (via [`crate::alphabet::Alphabet`]).
    pub fn encode(&self, window: &[u8]) -> u64 {
        debug_assert_eq!(window.len(), self.kmer_size);
        let mut h: u64 = 0;
        for (i, &b) in window.iter().enumerate() {
            debug_assert!((b as u32) < self.alphabet_size);
            h += (b as u64) * self.powers[i];
        }
        h
    }

    /// Iterate `(position, kmer_hash)` over every length-`k` window of
    /// `seq` that does not contain `skip_idx` (typically the X alphabet
    /// index, so ambiguous windows are excluded from the index).
    pub fn iter_kmers<'a>(
        &'a self,
        seq: &'a [u8],
        skip_idx: u8,
    ) -> impl Iterator<Item = (usize, u64)> + 'a {
        let k = self.kmer_size;
        (0..seq.len().saturating_sub(k - 1))
            .filter(move |&pos| !seq[pos..pos + k].contains(&skip_idx))
            .map(move |pos| (pos, self.encode(&seq[pos..pos + k])))
    }
}

/// One `(seq_id, position)` hit in a [`KmerIndex`].
///
/// Packed layout (6 bytes on disk) matches upstream's `IndexEntryLocal`;
/// kept unpacked in-memory for Rust-friendliness — the CSR layout means
/// most accesses are sequential so alignment matters more than per-entry
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KmerHit {
    pub seq_id: u32,
    pub pos: u16,
}

/// Abstract access pattern for k-mer posting lookup.
///
/// Lets the prefilter run over either an in-memory [`KmerIndex`] or a
/// memory-mapped [`crate::kmer_index_file::KmerIndexFile`] without a
/// separate code path. The callback form avoids forcing the on-disk
/// backend to allocate a `Vec<KmerHit>` per lookup while still letting
/// the in-memory backend iterate its slice with zero copies.
pub trait KmerLookup {
    /// The k-mer encoder describing this index — needed to iterate
    /// k-mers of a query in the right alphabet / k.
    fn encoder(&self) -> &KmerEncoder;

    /// Invoke `f` for every `(seq_id, pos)` hit recorded for the given
    /// k-mer hash. Caller must not rely on any particular ordering
    /// within a single k-mer's posting list.
    fn for_each_hit<F: FnMut(KmerHit)>(&self, hash: u64, f: F);
}

impl KmerLookup for KmerIndex {
    fn encoder(&self) -> &KmerEncoder {
        &self.encoder
    }
    fn for_each_hit<F: FnMut(KmerHit)>(&self, hash: u64, mut f: F) {
        for hit in self.lookup_hash(hash) {
            f(*hit);
        }
    }
}

/// CSR-layout k-mer index: for each possible k-mer value `k` in
/// `0..encoder.table_size()`, stores the slice `entries[offsets[k]..offsets[k+1]]`
/// of `KmerHit`s where that k-mer occurs.
///
/// Build pattern mirrors upstream `IndexTable`:
///   1. First pass: count k-mers per hash → `offsets` as a histogram.
///   2. Exclusive prefix-sum over `offsets` → now `offsets[k]` is the
///      start offset in `entries` for k-mer `k`, and `offsets[last]`
///      is the total number of hits.
///   3. Allocate `entries` to that size.
///   4. Second pass: write each hit at `entries[offsets[k]++]` (using a
///      temporary running-pointer vector to preserve the final offsets).
#[derive(Debug, Clone)]
pub struct KmerIndex {
    pub encoder: KmerEncoder,
    /// Length `encoder.table_size() + 1`; `entries[offsets[k]..offsets[k+1]]`
    /// are the hits for k-mer `k`.
    pub offsets: Vec<u64>,
    pub entries: Vec<KmerHit>,
}

impl KmerIndex {
    /// Build an index from an iterator of `(seq_id, &encoded_sequence)` pairs.
    /// `skip_idx` is the alphabet index treated as "unknown" and excluded
    /// from k-mer windows — normally the X index for proteins.
    ///
    /// Returns [`KmerIndexError::PositionOverflow`] if any indexed k-mer
    /// position exceeds `u16::MAX`. See [`KmerHit::pos`] for the rationale.
    pub fn build<'a, I>(encoder: KmerEncoder, seqs: I, skip_idx: u8) -> Result<Self>
    where
        I: IntoIterator<Item = (u32, &'a [u8])>,
    {
        let seqs: Vec<(u32, &[u8])> = seqs.into_iter().collect();
        Self::build_two_pass(encoder, &seqs, skip_idx)
    }

    fn build_two_pass(encoder: KmerEncoder, seqs: &[(u32, &[u8])], skip_idx: u8) -> Result<Self> {
        // Upfront overflow guard: any sequence whose last valid k-mer start
        // position exceeds u16::MAX is refused before we touch memory.
        let pos_limit = u16::MAX as usize;
        for &(seq_id, seq) in seqs {
            if seq.len() >= encoder.kmer_size() {
                let last_pos = seq.len() - encoder.kmer_size();
                if last_pos > pos_limit {
                    return Err(KmerIndexError::PositionOverflow {
                        seq_id,
                        pos: last_pos,
                        limit: pos_limit,
                    });
                }
            }
        }

        let table_size = encoder.table_size() as usize;
        let mut offsets = vec![0u64; table_size + 1];

        // Pass 1: histogram.
        for (_, seq) in seqs {
            for (_, h) in encoder.iter_kmers(seq, skip_idx) {
                offsets[h as usize] += 1;
            }
        }

        // Prefix-sum: offsets[k] = total count before k (= start offset for k).
        let mut total: u64 = 0;
        for slot in offsets.iter_mut().take(table_size) {
            let c = *slot;
            *slot = total;
            total += c;
        }
        offsets[table_size] = total;

        // Pass 2: populate entries using a running-pointer copy of offsets.
        let mut entries = vec![KmerHit { seq_id: 0, pos: 0 }; total as usize];
        let mut cursors = offsets.clone();
        for (seq_id, seq) in seqs {
            for (pos, h) in encoder.iter_kmers(seq, skip_idx) {
                let dst = cursors[h as usize] as usize;
                // Upfront guard ensures `pos <= u16::MAX` here; the `as u16`
                // is now safe-by-construction rather than a silent wrap.
                entries[dst] = KmerHit {
                    seq_id: *seq_id,
                    pos: pos as u16,
                };
                cursors[h as usize] += 1;
            }
        }

        Ok(Self {
            encoder,
            offsets,
            entries,
        })
    }

    /// Lookup hits for a k-mer (slice of alphabet indices of length `kmer_size`).
    pub fn lookup_window(&self, window: &[u8]) -> &[KmerHit] {
        let h = self.encoder.encode(window);
        self.lookup_hash(h)
    }

    /// Lookup hits for a precomputed k-mer hash.
    ///
    /// Out-of-range hashes (`>= encoder.table_size()`) return an empty
    /// slice rather than panicking — a public API taking an arbitrary u64
    /// must handle every value. For callers that want to distinguish "no
    /// hits" from "invalid hash", see [`KmerIndex::checked_lookup_hash`].
    pub fn lookup_hash(&self, h: u64) -> &[KmerHit] {
        self.checked_lookup_hash(h).unwrap_or(&[])
    }

    /// Like [`KmerIndex::lookup_hash`] but returns `None` for out-of-range
    /// hashes. Useful for callers that treat invalid input as an error
    /// rather than silently zero-hit.
    pub fn checked_lookup_hash(&self, h: u64) -> Option<&[KmerHit]> {
        if h >= self.encoder.table_size() {
            return None;
        }
        let k = h as usize;
        let start = self.offsets[k] as usize;
        let end = self.offsets[k + 1] as usize;
        Some(&self.entries[start..end])
    }

    /// Total number of (seq_id, pos) entries indexed.
    pub fn total_hits(&self) -> usize {
        self.entries.len()
    }

    /// Number of distinct k-mer hashes with at least one hit.
    pub fn distinct_kmers(&self) -> usize {
        self.offsets.windows(2).filter(|w| w[1] > w[0]).count()
    }
}

/// Build a k-mer index from a collection of encoded sequences.
///
/// Convenience wrapper for the common case of indexing [`Sequence`] objects.
pub fn build_index_from_sequences(
    encoder: KmerEncoder,
    seqs: &[(u32, &Sequence)],
    skip_idx: u8,
) -> Result<KmerIndex> {
    let pairs: Vec<(u32, &[u8])> = seqs
        .iter()
        .map(|(id, s)| (*id, s.data.as_slice()))
        .collect();
    KmerIndex::build(encoder, pairs, skip_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    #[test]
    fn encoder_basic_hash_values() {
        // Trivial 2-letter alphabet, k=3: hashes are base-2 integers.
        // w=[0,0,0] → 0;  w=[1,0,0] → 1;  w=[0,1,0] → 2;  w=[1,1,1] → 7.
        let enc = KmerEncoder::new(2, 3);
        assert_eq!(enc.table_size(), 8);
        assert_eq!(enc.encode(&[0, 0, 0]), 0);
        assert_eq!(enc.encode(&[1, 0, 0]), 1);
        assert_eq!(enc.encode(&[0, 1, 0]), 2);
        assert_eq!(enc.encode(&[1, 1, 1]), 7);
    }

    #[test]
    fn encoder_matches_upstream_polynomial_formula() {
        // Upstream Indexer.h:28-34 for kmer_size=6, alphabet=21:
        //   h = sum_{i=0..6} w[i] * 21^i
        let enc = KmerEncoder::new(21, 6);
        assert_eq!(enc.table_size(), 21u64.pow(6));
        let w = [0u8, 1, 2, 3, 4, 5];
        let expected: u64 = (0..6).map(|i| (w[i] as u64) * 21u64.pow(i as u32)).sum();
        assert_eq!(enc.encode(&w), expected);
    }

    #[test]
    fn iter_kmers_counts_match_formula() {
        // For a seq of length n, expect (n - k + 1) windows when no X.
        let enc = KmerEncoder::new(4, 3);
        let seq = vec![0u8, 1, 2, 3, 0, 1, 2]; // no X
        let kmers: Vec<_> = enc.iter_kmers(&seq, 99).collect(); // skip_idx unused
        assert_eq!(kmers.len(), seq.len() - enc.kmer_size() + 1);
    }

    #[test]
    fn iter_kmers_skips_windows_containing_x() {
        let enc = KmerEncoder::new(4, 3);
        // Positions 0..5 windowed at k=3: [0,1,2] [1,2,X] [2,X,3] [X,3,4] [3,4,5]
        // After skipping X (=99), only [0,1,2] and [3,4,5] survive.
        let seq: Vec<u8> = vec![0, 1, 2, 99, 3, 0, 2];
        let kmers: Vec<_> = enc.iter_kmers(&seq, 99).collect();
        // Windows without 99 in them: positions 0 ([0,1,2]) and 4 ([3,0,2]).
        let positions: Vec<usize> = kmers.iter().map(|(p, _)| *p).collect();
        assert_eq!(positions, vec![0, 4]);
    }

    #[test]
    fn iter_kmers_empty_when_seq_shorter_than_k() {
        let enc = KmerEncoder::new(4, 3);
        let short: [u8; 2] = [0, 1];
        assert_eq!(enc.iter_kmers(&short, 99).count(), 0);
    }

    #[test]
    fn index_build_and_lookup_small_case() {
        // Alphabet size 4, k=2 → table_size=16. One seq "ACGT" = [0,1,2,3].
        // Windows: (0,"AC")=0+1*4=4, (1,"CG")=1+2*4=9, (2,"GT")=2+3*4=14.
        let enc = KmerEncoder::new(4, 2);
        let seq: Vec<u8> = vec![0, 1, 2, 3];
        let idx = KmerIndex::build(enc, [(42u32, seq.as_slice())], 99).unwrap();
        assert_eq!(idx.total_hits(), 3);
        assert_eq!(idx.distinct_kmers(), 3);
        assert_eq!(idx.lookup_hash(4), &[KmerHit { seq_id: 42, pos: 0 }]);
        assert_eq!(idx.lookup_hash(9), &[KmerHit { seq_id: 42, pos: 1 }]);
        assert_eq!(idx.lookup_hash(14), &[KmerHit { seq_id: 42, pos: 2 }]);
        assert_eq!(idx.lookup_hash(0), &[]);
    }

    #[test]
    fn index_deduplicates_nothing_and_records_every_occurrence() {
        // A repeated k-mer must appear with each occurrence's (seq_id, pos).
        let enc = KmerEncoder::new(4, 2);
        let seq: Vec<u8> = vec![0, 1, 0, 1, 0]; // "ACACA": "AC" at 0, "CA" at 1, "AC" at 2, "CA" at 3
        let idx = KmerIndex::build(enc, [(7u32, seq.as_slice())], 99).unwrap();
        let ac_hash = 4; // =4
        let ca_hash = 1; // =1
        let ac_hits = idx.lookup_hash(ac_hash);
        assert_eq!(ac_hits.len(), 2);
        assert_eq!(ac_hits[0].pos, 0);
        assert_eq!(ac_hits[1].pos, 2);
        let ca_hits = idx.lookup_hash(ca_hash);
        assert_eq!(ca_hits.len(), 2);
        assert_eq!(ca_hits[0].pos, 1);
        assert_eq!(ca_hits[1].pos, 3);
    }

    #[test]
    fn index_multiple_sequences_keep_distinct_seq_ids() {
        let enc = KmerEncoder::new(4, 2);
        let a = vec![0u8, 1, 2]; // AC at 0, CG at 1
        let b = vec![0u8, 1]; // AC at 0
        let idx = KmerIndex::build(
            enc.clone(),
            [(10u32, a.as_slice()), (20u32, b.as_slice())],
            99,
        )
        .unwrap();
        assert_eq!(idx.total_hits(), 3);
        let ac = idx.lookup_window(&[0, 1]);
        let seq_ids: Vec<u32> = ac.iter().map(|h| h.seq_id).collect();
        assert_eq!(seq_ids.len(), 2);
        assert!(seq_ids.contains(&10) && seq_ids.contains(&20));
    }

    #[test]
    fn index_total_hits_matches_formula_with_x_skipping() {
        // n-k+1 minus #X-containing windows, per sequence.
        let enc = KmerEncoder::new(4, 2);
        let s1: Vec<u8> = vec![0, 1, 2, 3]; // 3 windows, no X
        let s2: Vec<u8> = vec![0, 99, 2, 99, 1]; // 4 windows, all contain X → 0 kept
        let s3: Vec<u8> = vec![0, 1, 99, 2, 3]; // 4 windows, positions 0 and 3 survive → 2
        let idx = KmerIndex::build(
            enc,
            [
                (1u32, s1.as_slice()),
                (2, s2.as_slice()),
                (3, s3.as_slice()),
            ],
            99,
        )
        .unwrap();
        assert_eq!(idx.total_hits(), 3 + 2);
    }

    #[test]
    fn index_offsets_are_monotone_and_sum_to_total() {
        let enc = KmerEncoder::new(4, 2);
        let s = vec![0u8, 1, 2, 3, 0, 1];
        let idx = KmerIndex::build(enc, [(0u32, s.as_slice())], 99).unwrap();
        // Offsets must be non-decreasing, start at 0, end at total_hits.
        assert_eq!(idx.offsets[0], 0);
        assert_eq!(*idx.offsets.last().unwrap() as usize, idx.total_hits());
        for w in idx.offsets.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn build_rejects_sequences_longer_than_u16_position_range() {
        // A sequence with a k-mer start position at u16::MAX + 1 must fail
        // loudly, not silently wrap. Using alphabet size 2 and k=1 so the
        // sequence itself is cheap: we just need seq.len() > u16::MAX + 1.
        let enc = KmerEncoder::new(2, 1);
        let seq = vec![0u8; u16::MAX as usize + 2]; // last valid pos = 65536 > u16::MAX
        let err = KmerIndex::build(enc, [(99u32, seq.as_slice())], u8::MAX).unwrap_err();
        assert_eq!(
            err,
            KmerIndexError::PositionOverflow {
                seq_id: 99,
                pos: u16::MAX as usize + 1,
                limit: u16::MAX as usize,
            }
        );
    }

    #[test]
    fn build_accepts_sequences_exactly_at_u16_position_limit() {
        // Boundary: last valid k-mer start pos == u16::MAX must succeed.
        // For k=1, that means seq.len() == u16::MAX + 1 = 65536.
        let enc = KmerEncoder::new(2, 1);
        let seq = vec![0u8; u16::MAX as usize + 1];
        let idx = KmerIndex::build(enc, [(1u32, seq.as_slice())], u8::MAX).unwrap();
        assert_eq!(idx.total_hits(), u16::MAX as usize + 1);
        // Last hit must be at position u16::MAX (not wrapped to 0).
        let last_entries = idx.lookup_hash(0);
        assert_eq!(last_entries.last().unwrap().pos, u16::MAX);
    }

    #[test]
    fn lookup_hash_returns_empty_slice_on_out_of_range_hash() {
        // Callers that pass arbitrary u64s (e.g., precomputed hashes from
        // a different encoder or garbage input) must not panic.
        let enc = KmerEncoder::new(4, 2);
        let seq = vec![0u8, 1, 2, 3];
        let idx = KmerIndex::build(enc, [(0u32, seq.as_slice())], 99).unwrap();
        // table_size = 4^2 = 16, so hash 16 is just past the end, and
        // u64::MAX is absurdly far past. Both must return an empty slice.
        assert_eq!(idx.lookup_hash(16), &[]);
        assert_eq!(idx.lookup_hash(u64::MAX), &[]);
    }

    #[test]
    fn checked_lookup_hash_distinguishes_no_hits_from_invalid() {
        let enc = KmerEncoder::new(4, 2);
        let seq = vec![0u8, 1]; // one hit: "AC" at pos 0
        let idx = KmerIndex::build(enc, [(0u32, seq.as_slice())], 99).unwrap();
        // Valid hash with no hits: Some([]).
        assert_eq!(idx.checked_lookup_hash(0), Some(&[] as &[KmerHit]));
        // Invalid hash (>= table_size): None.
        assert_eq!(idx.checked_lookup_hash(16), None);
        assert_eq!(idx.checked_lookup_hash(u64::MAX), None);
    }

    #[test]
    fn index_over_protein_alphabet_with_real_sequence() {
        // End-to-end: protein alphabet → encoded sequence → index → lookup.
        let alpha = Alphabet::protein();
        let seq = Sequence::from_ascii(alpha.clone(), b"MKLV");
        let enc = KmerEncoder::new(alpha.size() as u32, 3);
        let x = alpha.encode(b'X');
        let idx = build_index_from_sequences(enc.clone(), &[(1, &seq)], x).unwrap();
        assert_eq!(idx.total_hits(), 2); // windows MKL and KLV

        // Looking up the k-mer "MKL" should find exactly one hit at seq_id=1, pos=0.
        let mkl_window = vec![alpha.encode(b'M'), alpha.encode(b'K'), alpha.encode(b'L')];
        let hits = idx.lookup_window(&mkl_window);
        assert_eq!(hits, &[KmerHit { seq_id: 1, pos: 0 }]);
    }
}
