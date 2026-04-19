//! Padded target DB layout for coalesced GPU access.
//!
//! Upstream MMseqs2's `makepaddedseqdb` and libmarv's internal database
//! format use the same idea: when dispatching many targets per batch
//! to a GPU kernel, the memory layout that matters for performance is
//! **transposed coalesced** — within a bucket of `bucket_size` targets
//! all padded to a common length `L`, bytes are stored so that the
//! fast-moving axis is "slot-within-bucket" and the slow-moving axis
//! is "position along the target." A warp of threads (one per slot)
//! reading position `p` then issues a single 32-byte coalesced global
//! load instead of 32 separate cache-miss-prone reads.
//!
//! The savings are real only when targets in a bucket have similar
//! lengths (otherwise the short ones waste bucket width on padding).
//! Hence we sort targets by length before bucketing — the cheapest way
//! to minimize padding waste. Upstream does the same via their
//! `length_partitions.hpp` length-class scheme; a simple stable sort is
//! good enough for first-wave perf and is straightforward to reason
//! about.
//!
//! This module ships the CPU-side data structure and its round-trip
//! guarantees. Phase 4.4c wires GPU kernels that consume it.

/// Transposed-coalesced padded target database.
///
/// Construction sorts the input targets by length ascending, packs
/// `bucket_size` of them per bucket, pads each bucket to its longest
/// member, and transposes the byte layout so each "position row"
/// within a bucket is one cache line of bucket_size bytes.
///
/// `original_to_padded[i] = (bucket, slot)` records where the i-th
/// input target ended up. The last bucket may have fewer than
/// `bucket_size` real targets; inactive slots have
/// `slot_lens[bucket][slot] == 0` and their bytes are all
/// `pad_byte`. `get_target_byte(i, p)` hides this bookkeeping.
#[derive(Debug, Clone)]
pub struct PaddedDb {
    /// Targets per bucket. Typically a warp size (32) for GPU use.
    pub bucket_size: usize,
    /// Padded length of each bucket; `bucket_padded_lens[b]` is the
    /// max original length of any real target in bucket `b`.
    pub bucket_padded_lens: Vec<usize>,
    /// Byte offset where each bucket's block starts in `data`.
    /// `data[bucket_starts[b] + p * bucket_size + slot]` is the byte
    /// of target-at-slot at position p.
    pub bucket_starts: Vec<usize>,
    /// Flat storage. Size = `Σ_b bucket_size * bucket_padded_lens[b]`.
    pub data: Vec<u8>,
    /// `original_to_padded[i] = (bucket, slot)` maps an input target
    /// index to its final placement.
    pub original_to_padded: Vec<(usize, usize)>,
    /// Pre-padding length of each original target, indexed by the
    /// *original* input order.
    pub original_lens: Vec<usize>,
    /// Byte used to fill padding in inactive positions. Callers
    /// typically set this to an alphabet's unknown-letter index (e.g.
    /// 20 for the 21-letter protein alphabet) so the rest of the
    /// pipeline treats pads as X — but the pad byte is
    /// user-controlled because the correct choice is alphabet-
    /// dependent.
    pub pad_byte: u8,
}

/// Per-slot metadata returned by [`PaddedDb::slot_lens`]. Exposed for
/// kernels that need to know each slot's real length to stop
/// iterating.
#[derive(Debug, Clone, Copy)]
pub struct SlotInfo {
    /// Real length of this slot's target (0 = inactive / padding-only slot).
    pub real_len: usize,
    /// Original index of this target in the input list, or
    /// `usize::MAX` for inactive slots.
    pub original_index: usize,
}

impl PaddedDb {
    /// Build a PaddedDb from a list of variable-length targets.
    ///
    /// Targets are sorted by length ascending (stable) before being
    /// grouped into buckets of `bucket_size`. `pad_byte` fills padding
    /// positions in both active slots past their real length and in
    /// inactive trailing slots of the final bucket.
    ///
    /// Panics if `bucket_size == 0`.
    pub fn build(targets: &[&[u8]], bucket_size: usize, pad_byte: u8) -> Self {
        assert!(bucket_size > 0, "bucket_size must be >= 1");

        let n = targets.len();
        let original_lens: Vec<usize> = targets.iter().map(|t| t.len()).collect();

        // Sort original indices by length ascending (stable).
        let mut sorted_idx: Vec<usize> = (0..n).collect();
        sorted_idx.sort_by_key(|&i| original_lens[i]);

        let n_buckets = n.div_ceil(bucket_size).max(1);
        let mut bucket_padded_lens = Vec::with_capacity(n_buckets);
        let mut bucket_starts = Vec::with_capacity(n_buckets);
        let mut original_to_padded = vec![(usize::MAX, usize::MAX); n];

        // Compute per-bucket padded lengths + cumulative byte offsets.
        let mut total_bytes: usize = 0;
        for b in 0..n_buckets {
            let start = b * bucket_size;
            let end = (start + bucket_size).min(n);
            let padded_len = (start..end)
                .map(|k| original_lens[sorted_idx[k]])
                .max()
                .unwrap_or(0);
            bucket_padded_lens.push(padded_len);
            bucket_starts.push(total_bytes);
            total_bytes += bucket_size * padded_len;
        }

        let mut data = vec![pad_byte; total_bytes];
        for (b, &padded_len) in bucket_padded_lens.iter().enumerate() {
            let start = b * bucket_size;
            let end = (start + bucket_size).min(n);
            let bucket_start = bucket_starts[b];
            for slot in 0..(end - start) {
                let orig_idx = sorted_idx[start + slot];
                original_to_padded[orig_idx] = (b, slot);
                let target = targets[orig_idx];
                // Transposed layout: position is slow, slot is fast.
                for (p, &byte) in target.iter().enumerate() {
                    debug_assert!(p < padded_len);
                    data[bucket_start + p * bucket_size + slot] = byte;
                }
                // Positions from target.len() to padded_len-1 stay as
                // pad_byte (already initialized).
            }
            // Inactive slots (slot >= end - start) stay fully as pad_byte.
        }

        Self {
            bucket_size,
            bucket_padded_lens,
            bucket_starts,
            data,
            original_to_padded,
            original_lens,
            pad_byte,
        }
    }

    pub fn num_targets(&self) -> usize {
        self.original_lens.len()
    }

    pub fn num_buckets(&self) -> usize {
        self.bucket_padded_lens.len()
    }

    /// Get one byte of the `original_idx`-th target at position `p`.
    /// Returns `None` if `original_idx` is out of range. For `p >=
    /// original_lens[original_idx]` returns the pad byte.
    pub fn get_target_byte(&self, original_idx: usize, p: usize) -> Option<u8> {
        if original_idx >= self.num_targets() {
            return None;
        }
        let (b, slot) = self.original_to_padded[original_idx];
        let padded_len = self.bucket_padded_lens[b];
        if p >= padded_len {
            // Beyond the bucket's padded length: consider it pad.
            return Some(self.pad_byte);
        }
        Some(self.data[self.bucket_starts[b] + p * self.bucket_size + slot])
    }

    /// Decode the `bucket`-th bucket into `bucket_size` per-slot
    /// byte vectors of length `bucket_padded_lens[bucket]`. Useful for
    /// testing and for CPU-side consumers; GPU consumers should index
    /// directly into `data` with the coalesced layout.
    pub fn decode_bucket(&self, bucket: usize) -> Vec<Vec<u8>> {
        let padded_len = self.bucket_padded_lens[bucket];
        let bucket_start = self.bucket_starts[bucket];
        (0..self.bucket_size)
            .map(|slot| {
                (0..padded_len)
                    .map(|p| self.data[bucket_start + p * self.bucket_size + slot])
                    .collect()
            })
            .collect()
    }

    /// Slot info for every slot in every bucket. Returns
    /// `Vec<Vec<SlotInfo>>` with one outer entry per bucket, inner
    /// entries of length `bucket_size` (last bucket's inactive slots
    /// have `real_len == 0` and `original_index == usize::MAX`).
    pub fn slot_lens(&self) -> Vec<Vec<SlotInfo>> {
        let n_buckets = self.num_buckets();
        let mut out = vec![
            vec![
                SlotInfo {
                    real_len: 0,
                    original_index: usize::MAX
                };
                self.bucket_size
            ];
            n_buckets
        ];
        for (orig, &(b, slot)) in self.original_to_padded.iter().enumerate() {
            if b == usize::MAX {
                continue;
            }
            out[b][slot] = SlotInfo {
                real_len: self.original_lens[orig],
                original_index: orig,
            };
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_target_list_produces_zero_buckets_but_allocates_minimum() {
        let db = PaddedDb::build(&[], 4, 0xff);
        assert_eq!(db.num_targets(), 0);
        // `max(1)` on n_buckets keeps at least one slot of machinery so
        // downstream dispatchers don't hit divide-by-zero on the
        // vacuous case. Bucket has padded_len 0 so data is empty.
        assert_eq!(db.num_buckets(), 1);
        assert_eq!(db.bucket_padded_lens[0], 0);
        assert!(db.data.is_empty());
    }

    #[test]
    fn single_full_bucket_round_trips_exactly() {
        // 4 targets, bucket_size=4 → one full bucket, padded to the
        // longest length (6). Sorted ascending by length → order
        // within bucket is [1, 3, 5, 6].
        let t0: Vec<u8> = vec![1, 2, 3, 4, 5, 6]; // len 6
        let t1: Vec<u8> = vec![7]; // len 1
        let t2: Vec<u8> = vec![8, 9, 10]; // len 3
        let t3: Vec<u8> = vec![11, 12, 13, 14, 15]; // len 5
        let targets: Vec<&[u8]> = vec![&t0, &t1, &t2, &t3];
        let db = PaddedDb::build(&targets, 4, 0);

        assert_eq!(db.num_targets(), 4);
        assert_eq!(db.num_buckets(), 1);
        assert_eq!(db.bucket_padded_lens[0], 6);

        // Every byte of every original target recoverable via get_target_byte.
        for (i, target) in targets.iter().enumerate() {
            for (p, &byte) in target.iter().enumerate() {
                assert_eq!(
                    db.get_target_byte(i, p),
                    Some(byte),
                    "round-trip mismatch for target {i} at pos {p}"
                );
            }
        }
    }

    #[test]
    fn short_targets_padded_correctly_past_their_real_length() {
        let t0: Vec<u8> = vec![10, 20, 30, 40, 50]; // len 5
        let t1: Vec<u8> = vec![99]; // len 1
        let targets: Vec<&[u8]> = vec![&t0, &t1];
        let db = PaddedDb::build(&targets, 2, 0xAA);

        // t1 padded with 0xAA at positions 1..5
        for p in 1..5 {
            assert_eq!(db.get_target_byte(1, p), Some(0xAA), "t1 padding at {p}");
        }
        // t0 has real bytes at 0..5
        for (p, expected) in [10u8, 20, 30, 40, 50].iter().enumerate() {
            assert_eq!(db.get_target_byte(0, p), Some(*expected));
        }
    }

    #[test]
    fn partial_last_bucket_uses_pad_byte_for_inactive_slots() {
        // 5 targets, bucket_size=4 → 2 buckets; last has 1 real slot + 3 inactive.
        let targets_owned: Vec<Vec<u8>> = (0..5).map(|i| vec![(i + 1) as u8]).collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let db = PaddedDb::build(&targets, 4, 0xff);

        assert_eq!(db.num_buckets(), 2);

        // Each target is len 1 → padded_len=1 per bucket. Last bucket
        // has 1 real + 3 inactive slots. The inactive slots' data
        // should all be pad_byte.
        let last_bucket = db.decode_bucket(1);
        assert_eq!(last_bucket.len(), 4);
        // Exactly one of the four slots has a real target byte; the
        // other three are all pad_byte.
        let real_bytes: Vec<&Vec<u8>> = last_bucket.iter().filter(|r| r[0] != 0xff).collect();
        let pad_bytes: Vec<&Vec<u8>> = last_bucket.iter().filter(|r| r[0] == 0xff).collect();
        assert_eq!(real_bytes.len(), 1);
        assert_eq!(pad_bytes.len(), 3);
    }

    #[test]
    fn every_target_recoverable_regardless_of_bucket_boundary() {
        // 50 targets of widely varying lengths, bucket_size=8.
        let mut rng_state: u32 = 0xcafe_babe;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };
        let targets_owned: Vec<Vec<u8>> = (0..50)
            .map(|_| {
                let len = 1 + (next() as usize % 40);
                (0..len).map(|_| (next() as u8) % 21).collect()
            })
            .collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let db = PaddedDb::build(&targets, 8, 20);

        for (i, original) in targets.iter().enumerate() {
            for (p, &expected) in original.iter().enumerate() {
                assert_eq!(
                    db.get_target_byte(i, p),
                    Some(expected),
                    "target {i}, pos {p}: round-trip failed"
                );
            }
            // Past the original length, we should hit pad_byte up to
            // the bucket's padded_len.
            let (b, _) = db.original_to_padded[i];
            let padded_len = db.bucket_padded_lens[b];
            for p in original.len()..padded_len {
                assert_eq!(
                    db.get_target_byte(i, p),
                    Some(20),
                    "target {i}, pos {p}: padding mismatch"
                );
            }
        }
    }

    #[test]
    fn sort_by_length_minimizes_padding_waste() {
        // Two targets, one very short, one very long, bucket_size=1
        // → each goes in its own bucket, no padding needed (sort is
        // irrelevant when bucket_size==1, but this exercises the
        // degenerate case).
        let t0: Vec<u8> = vec![1]; // len 1
        let t1: Vec<u8> = (0..1000).map(|i| (i % 21) as u8).collect(); // len 1000
        let targets: Vec<&[u8]> = vec![&t0, &t1];
        let db = PaddedDb::build(&targets, 1, 0);
        assert_eq!(db.num_buckets(), 2);
        // Total data = 1 + 1000 (no padding since each bucket has 1 slot).
        assert_eq!(db.data.len(), 1 + 1000);

        // With bucket_size=2, both would share a bucket padded to 1000
        // → 2000 bytes total (1000 of them padding wasted on t0).
        let db_wide = PaddedDb::build(&targets, 2, 0);
        assert_eq!(db_wide.num_buckets(), 1);
        assert_eq!(db_wide.data.len(), 2 * 1000);
        // Round-trip still works despite the waste.
        assert_eq!(db_wide.get_target_byte(0, 0), Some(1));
        assert_eq!(db_wide.get_target_byte(1, 999), Some(((999) % 21) as u8));
    }

    #[test]
    fn slot_lens_reports_real_lengths_and_originals() {
        let t0: Vec<u8> = vec![1, 2, 3];
        let t1: Vec<u8> = vec![4];
        let t2: Vec<u8> = vec![5, 6];
        let targets: Vec<&[u8]> = vec![&t0, &t1, &t2];
        let db = PaddedDb::build(&targets, 4, 0);
        let lens = db.slot_lens();
        assert_eq!(lens.len(), 1);
        // 3 real slots + 1 inactive
        let active: Vec<&SlotInfo> = lens[0].iter().filter(|s| s.real_len > 0).collect();
        assert_eq!(active.len(), 3);
        let inactive: Vec<&SlotInfo> = lens[0].iter().filter(|s| s.real_len == 0).collect();
        assert_eq!(inactive.len(), 1);
        assert_eq!(inactive[0].original_index, usize::MAX);
    }

    #[test]
    fn transposed_layout_puts_slot_axis_fast() {
        // Explicit check of the coalescing-friendly layout: for a
        // bucket of 4 targets all length 3, the first 4 bytes of
        // `data` should be each target's byte at position 0.
        let t0: Vec<u8> = vec![10, 11, 12];
        let t1: Vec<u8> = vec![20, 21, 22];
        let t2: Vec<u8> = vec![30, 31, 32];
        let t3: Vec<u8> = vec![40, 41, 42];
        let targets: Vec<&[u8]> = vec![&t0, &t1, &t2, &t3];
        let db = PaddedDb::build(&targets, 4, 0);
        // After length-ascending sort, order is t0, t1, t2, t3 (all
        // length 3 — stable sort preserves input order). So:
        //   data[0..4]  = bytes at pos 0 across slots → [10, 20, 30, 40]
        //   data[4..8]  = bytes at pos 1 across slots → [11, 21, 31, 41]
        //   data[8..12] = bytes at pos 2 across slots → [12, 22, 32, 42]
        assert_eq!(&db.data[0..4], &[10, 20, 30, 40]);
        assert_eq!(&db.data[4..8], &[11, 21, 31, 41]);
        assert_eq!(&db.data[8..12], &[12, 22, 32, 42]);
    }
}
