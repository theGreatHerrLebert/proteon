//! On-disk k-mer index file (`.kmi`) — writer + mmap-backed reader.
//!
//! Format spec: `ferritin-search/docs/KMER_INDEX_FORMAT_SPEC.md`.
//!
//! This module handles only serialization of the same CSR-style layout
//! `KmerIndex` uses in-memory (offsets + entries) to/from a file. The
//! reader mmaps the file and exposes `offsets`, `entries_seq_id`, and
//! `entries_pos` as byte-cast slices into the mmap — no decode, no
//! copy. Peak resident memory for a UniRef50-scale index drops from
//! ~100 GB (in-memory) to ~file-size bounded by the OS page cache.
//!
//! Phase 3 of the ferritin-search memmap refactor — see commit history
//! for phases 1 (memmap DBReader) and 2 (SearchEngine drops targets_full).
//!
//! Format v2 (current) embeds the full reducer state used at build time
//! so openers can verify their own reducer produces the same mapping.
//! v1 (short-lived, never shipped as a persisted artifact) only pinned
//! `alphabet_size`, which was insufficient — two different substitution
//! matrices can produce different full-to-reduced mappings at the same
//! reduced_size, and the build/open pair would silently accept an
//! incompatible index.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::{Mmap, MmapMut};
use thiserror::Error;

use crate::alphabet::Alphabet;
use crate::db::DBReader;
use crate::kmer::{KmerEncoder, KmerHit, KmerIndex, KmerIndexError, KmerLookup};
use crate::reduced_alphabet::ReducedAlphabet;
use crate::sequence::Sequence;

pub const KMI_MAGIC: [u8; 4] = *b"FKMI";
pub const KMI_VERSION: u32 = 2;
/// Header is fixed at 64 bytes; the three `*_pos` fields are absolute
/// byte offsets into the file and their consistency is validated on
/// open.
pub const KMI_HEADER_SIZE: u64 = 64;

/// Size of the fixed-width prologue of the reducer section (before the
/// variable-length `full_to_reduced` byte array). See the format spec.
pub const REDUCER_SECTION_PROLOGUE: u64 = 12;

#[derive(Debug, Error)]
pub enum KmiWriterError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum KmiReaderError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("file too small to contain a .kmi header (got {0} bytes, need ≥ {1})")]
    TruncatedHeader(u64, u64),
    #[error("bad magic: expected FKMI, got {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported .kmi version {found} (reader handles {supported})")]
    BadVersion { found: u32, supported: u32 },
    #[error("layout mismatch in .kmi header: {field} = {found}, expected {expected}")]
    LayoutMismatch {
        field: &'static str,
        found: u64,
        expected: u64,
    },
    #[error(
        "offsets tail invariant failed: offsets[table_size] = {found}, \
         expected n_entries = {expected}"
    )]
    OffsetsTail { found: u64, expected: u64 },
    #[error("file size {found} disagrees with layout (expected {expected})")]
    FileSizeMismatch { found: u64, expected: u64 },
    #[error("reducer section malformed: {detail}")]
    BadReducerSection { detail: String },
}

#[derive(Debug, Error)]
pub enum ExternalBuildError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("writer error: {0}")]
    Writer(#[from] KmiWriterError),
    #[error("k-mer indexing failed: {0}")]
    Indexing(#[from] KmerIndexError),
}

/// Snapshot of the reducer used at build time, as embedded in the .kmi.
///
/// Two snapshots compare equal iff they define the same full-to-reduced
/// mapping — which is the contract openers need to guarantee their
/// reducer matches what was indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReducerSnapshot {
    /// No reduction was applied — the k-mer index lives in the full
    /// alphabet.
    None,
    /// A reduction was applied; `full_to_reduced[i]` ∈ `0..reduced_size`
    /// for every full-alphabet index `i`. `unknown_reduced_idx` matches
    /// `ReducedAlphabet::unknown_reduced_idx` at build time.
    Some {
        full_size: u32,
        reduced_size: u32,
        unknown_reduced_idx: Option<u8>,
        full_to_reduced: Vec<u8>,
    },
}

impl ReducerSnapshot {
    /// Capture the state of a (possibly-absent) `ReducedAlphabet`.
    pub fn from_reducer(reducer: Option<&ReducedAlphabet>) -> Self {
        match reducer {
            None => Self::None,
            Some(r) => Self::Some {
                full_size: r.full_size as u32,
                reduced_size: r.reduced_size as u32,
                unknown_reduced_idx: r.unknown_reduced_idx,
                full_to_reduced: r.full_to_reduced.clone(),
            },
        }
    }

    fn encoded_byte_len(&self) -> u64 {
        match self {
            Self::None => 0,
            Self::Some {
                full_to_reduced, ..
            } => REDUCER_SECTION_PROLOGUE + full_to_reduced.len() as u64,
        }
    }
}

/// Write an in-memory [`KmerIndex`] to a `.kmi` file at `path`.
///
/// `reducer` records the full-to-reduced mapping used at build time so
/// openers can verify their own reducer produces the same mapping —
/// matrix-dependent reductions at the same `reduced_size` can disagree
/// on the actual equivalence classes, and loading an index built with
/// a different mapping silently returns wrong hits.
pub fn write_kmi(
    index: &KmerIndex,
    reducer: Option<&ReducedAlphabet>,
    path: impl AsRef<Path>,
) -> Result<(), KmiWriterError> {
    let path = path.as_ref();
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, file);

    let alphabet_size = index.encoder.alphabet_size();
    let kmer_size = index.encoder.kmer_size();
    let table_size = index.encoder.table_size();
    let n_entries = index.entries.len() as u64;

    let snap = ReducerSnapshot::from_reducer(reducer);
    let reducer_section_size = snap.encoded_byte_len();

    let offsets_byte_pos = KMI_HEADER_SIZE + reducer_section_size;
    let entries_seq_id_pos = offsets_byte_pos + 8 * (table_size + 1);
    let entries_pos_pos = entries_seq_id_pos + 4 * n_entries;

    // --- header (64 bytes) ---
    w.write_all(&KMI_MAGIC)?;
    w.write_all(&KMI_VERSION.to_le_bytes())?;
    w.write_all(&alphabet_size.to_le_bytes())?;
    w.write_all(&(kmer_size as u32).to_le_bytes())?;
    w.write_all(&table_size.to_le_bytes())?;
    w.write_all(&n_entries.to_le_bytes())?;
    w.write_all(&offsets_byte_pos.to_le_bytes())?;
    w.write_all(&entries_seq_id_pos.to_le_bytes())?;
    w.write_all(&entries_pos_pos.to_le_bytes())?;
    w.write_all(&(reducer_section_size as u32).to_le_bytes())?;
    w.write_all(&[0u8; 4])?; // reserved to pad the 64-byte header

    // --- reducer section (only if a reducer was used) ---
    if let ReducerSnapshot::Some {
        full_size,
        reduced_size,
        unknown_reduced_idx,
        full_to_reduced,
    } = &snap
    {
        w.write_all(&full_size.to_le_bytes())?;
        w.write_all(&reduced_size.to_le_bytes())?;
        let (marker, idx) = match unknown_reduced_idx {
            Some(i) => (1u8, *i),
            None => (0u8, 0u8),
        };
        w.write_all(&[marker, idx, 0u8, 0u8])?; // pad to 4-byte boundary
        w.write_all(full_to_reduced)?;
    }

    // --- offsets ---
    for off in &index.offsets {
        w.write_all(&off.to_le_bytes())?;
    }

    // --- entries_seq_id ---
    for hit in &index.entries {
        w.write_all(&hit.seq_id.to_le_bytes())?;
    }

    // --- entries_pos ---
    for hit in &index.entries {
        w.write_all(&hit.pos.to_le_bytes())?;
    }

    w.flush()?;
    Ok(())
}

/// Two-pass external-memory builder: writes a `.kmi` directly from a
/// [`DBReader`] without materializing an in-memory [`KmerIndex`].
///
/// Peak resident RAM is bounded by the offsets table
/// (~`8 * (table_size + 1)` bytes) plus a duplicate cursors array of
/// the same size — `2 × 8 × 21^6 ≈ 1.4 GB` for the full protein
/// alphabet, `~77 MB` at the reduced alphabet (`13^6` slots). Orders
/// of magnitude less than the in-memory build path, which at UniRef50
/// scale needs ~100 GB RAM for the postings alone.
///
/// Algorithm:
///  - **Pass 1**: walk every target, count k-mer occurrences per
///    hash into `offsets`. Then exclusive-prefix-sum → `offsets[k]`
///    is the start offset in the entries arrays for k-mer `k`.
///  - **Allocate** a file sized for header + reducer section + offsets
///    + entries_seq_id + entries_pos. Truncate + mmap mutably.
///  - **Pass 2**: walk every target again, for each k-mer write
///    `(seq_id, pos)` into the mmap at `cursors[hash]++`. OS page
///    cache absorbs the random-access writes; no explicit buffering.
///
/// The DB is walked in its stored order both passes; no sorting. The
/// resulting `.kmi` is byte-equivalent to an in-memory build + write
/// for the same inputs, modulo the order of entries within a single
/// kmer's posting list (external build preserves DB-index order within
/// a bucket; in-memory builder happens to produce the same order
/// because it uses a cursors-running-pointer too).
pub fn build_kmi_external(
    db: &DBReader,
    alphabet: Alphabet,
    reducer: Option<&ReducedAlphabet>,
    opts: BuildExternalOptions,
    out_path: impl AsRef<Path>,
) -> Result<(), ExternalBuildError> {
    let out_path = out_path.as_ref();
    let (kmer_alphabet_size, skip_idx) = match reducer {
        Some(r) => (r.reduced_size, r.unknown_reduced_idx.unwrap_or(0)),
        None => (alphabet.size(), alphabet.encode(b'X')),
    };
    let encoder = KmerEncoder::new(kmer_alphabet_size as u32, opts.k);
    let table_size = encoder.table_size() as usize;

    // ---------- Pass 1: histogram ----------
    let mut offsets = vec![0u64; table_size + 1];
    let pos_limit = u16::MAX as usize;
    for entry in &db.index {
        let payload = db.get_payload(entry);
        let encoded = match reducer {
            Some(r) => r.reduce_sequence(&Sequence::from_ascii(alphabet.clone(), payload).data),
            None => Sequence::from_ascii(alphabet.clone(), payload).data,
        };
        if encoded.len() >= opts.k {
            let last_pos = encoded.len() - opts.k;
            if last_pos > pos_limit {
                return Err(ExternalBuildError::Indexing(
                    KmerIndexError::PositionOverflow {
                        seq_id: entry.key,
                        pos: last_pos,
                        limit: pos_limit,
                    },
                ));
            }
        }
        for (_pos, h) in encoder.iter_kmers(&encoded, skip_idx) {
            offsets[h as usize] += 1;
        }
    }

    // Exclusive prefix-sum; `offsets[table_size]` ends up == total.
    let mut total: u64 = 0;
    for slot in offsets.iter_mut().take(table_size) {
        let c = *slot;
        *slot = total;
        total += c;
    }
    offsets[table_size] = total;
    let n_entries = total;

    // ---------- Layout: header + reducer section + offsets + entries ----------
    let snap = ReducerSnapshot::from_reducer(reducer);
    let reducer_section_size = snap.encoded_byte_len();
    let offsets_byte_pos = KMI_HEADER_SIZE + reducer_section_size;
    let entries_seq_id_pos = offsets_byte_pos + 8 * (table_size as u64 + 1);
    let entries_pos_pos = entries_seq_id_pos + 4 * n_entries;
    let total_size = entries_pos_pos + 2 * n_entries;

    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(out_path)?;
    file.set_len(total_size)?;
    // SAFETY: We own the file, have it locked open, and nobody else
    // is expected to mutate it while we're writing. Standard mmap
    // contract applies.
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };

    // --- Write header + reducer section + offsets ---
    write_header_into(
        &mut mmap[..KMI_HEADER_SIZE as usize],
        encoder.alphabet_size(),
        opts.k,
        table_size as u64,
        n_entries,
        offsets_byte_pos,
        entries_seq_id_pos,
        entries_pos_pos,
        reducer_section_size as u32,
    );
    if let ReducerSnapshot::Some {
        full_size,
        reduced_size,
        unknown_reduced_idx,
        full_to_reduced,
    } = &snap
    {
        write_reducer_section_into(
            &mut mmap[KMI_HEADER_SIZE as usize..(KMI_HEADER_SIZE + reducer_section_size) as usize],
            *full_size,
            *reduced_size,
            *unknown_reduced_idx,
            full_to_reduced,
        );
    }
    // Copy offsets into the mmap. Do this before pass 2 — entries
    // layout references these.
    let offsets_slice = &mut mmap
        [offsets_byte_pos as usize..(offsets_byte_pos + 8 * (table_size as u64 + 1)) as usize];
    for (i, off) in offsets.iter().enumerate() {
        offsets_slice[8 * i..8 * i + 8].copy_from_slice(&off.to_le_bytes());
    }

    // ---------- Pass 2: populate entries ----------
    // Running write pointer per kmer hash; starts at offsets[k].
    let mut cursors: Vec<u64> = offsets.clone();

    // Split the mmap so we have disjoint mutable slices for seq_id and
    // pos arrays. Keeps the borrow checker happy during the hot inner
    // loop without unsafe pointer math.
    let (_head, rest) = mmap.split_at_mut(entries_seq_id_pos as usize);
    let (seq_id_region, pos_region) = rest.split_at_mut((4 * n_entries) as usize);

    // Multi-range passes: each pass scans the full DB but only writes
    // entries whose k-mer hash falls in [lo, hi). At K > 1 the writes
    // are localized to 1/K of the output file per pass, which keeps
    // the OS page cache hot instead of thrashing across all ~n_entries
    // output pages. See `BuildExternalOptions::hash_range_passes`.
    let k_passes = opts.hash_range_passes.max(1);
    let table_u64 = table_size as u64;
    for pass in 0..k_passes {
        let lo = (table_u64 * pass as u64) / k_passes as u64;
        let hi = (table_u64 * (pass as u64 + 1)) / k_passes as u64;
        for entry in &db.index {
            let payload = db.get_payload(entry);
            let encoded = match reducer {
                Some(r) => r.reduce_sequence(&Sequence::from_ascii(alphabet.clone(), payload).data),
                None => Sequence::from_ascii(alphabet.clone(), payload).data,
            };
            let seq_id_bytes = entry.key.to_le_bytes();
            for (pos, h) in encoder.iter_kmers(&encoded, skip_idx) {
                if h < lo || h >= hi {
                    continue;
                }
                let cur = cursors[h as usize] as usize;
                let seq_id_off = 4 * cur;
                seq_id_region[seq_id_off..seq_id_off + 4].copy_from_slice(&seq_id_bytes);
                let pos_off = 2 * cur;
                let pos_bytes = (pos as u16).to_le_bytes();
                pos_region[pos_off..pos_off + 2].copy_from_slice(&pos_bytes);
                cursors[h as usize] += 1;
            }
        }
    }

    // Sanity: every cursor must have advanced exactly to the next
    // bucket's start (== offsets[k+1]). Catches any pass-1/pass-2
    // discrepancy up-front instead of deferring to reader-side
    // validation after the file is already on disk.
    for k in 0..table_size {
        if cursors[k] != offsets[k + 1] {
            return Err(ExternalBuildError::Io(std::io::Error::other(format!(
                "external build inconsistency: cursor[{k}] = {} but offsets[{}] = {}",
                cursors[k],
                k + 1,
                offsets[k + 1],
            ))));
        }
    }

    mmap.flush()?;
    Ok(())
}

/// Options for [`build_kmi_external`].
///
/// `hash_range_passes` controls how many write-side DB scans run:
/// each scan writes only entries whose k-mer hash falls in one slice
/// of the full hash range `[0, table_size)`. At `= 1` the builder
/// touches the whole output file in a single pass and thrashes the
/// page cache when the output is much larger than available RAM.
/// Higher values trade DB scans for write locality: `K` passes each
/// touch `~1/K` of the output file. A good default is
/// `max(1, ceil(estimated_output_bytes / 1 GB))` but users who know
/// their machine can pick explicitly.
#[derive(Debug, Clone)]
pub struct BuildExternalOptions {
    pub k: usize,
    pub hash_range_passes: usize,
}

impl Default for BuildExternalOptions {
    fn default() -> Self {
        Self {
            k: 6,
            hash_range_passes: 1,
        }
    }
}

fn write_header_into(
    dst: &mut [u8],
    alphabet_size: u32,
    kmer_size: usize,
    table_size: u64,
    n_entries: u64,
    offsets_byte_pos: u64,
    entries_seq_id_pos: u64,
    entries_pos_pos: u64,
    reducer_section_size: u32,
) {
    debug_assert_eq!(dst.len(), KMI_HEADER_SIZE as usize);
    dst[..4].copy_from_slice(&KMI_MAGIC);
    dst[4..8].copy_from_slice(&KMI_VERSION.to_le_bytes());
    dst[8..12].copy_from_slice(&alphabet_size.to_le_bytes());
    dst[12..16].copy_from_slice(&(kmer_size as u32).to_le_bytes());
    dst[16..24].copy_from_slice(&table_size.to_le_bytes());
    dst[24..32].copy_from_slice(&n_entries.to_le_bytes());
    dst[32..40].copy_from_slice(&offsets_byte_pos.to_le_bytes());
    dst[40..48].copy_from_slice(&entries_seq_id_pos.to_le_bytes());
    dst[48..56].copy_from_slice(&entries_pos_pos.to_le_bytes());
    dst[56..60].copy_from_slice(&reducer_section_size.to_le_bytes());
    dst[60..64].fill(0);
}

fn write_reducer_section_into(
    dst: &mut [u8],
    full_size: u32,
    reduced_size: u32,
    unknown_reduced_idx: Option<u8>,
    full_to_reduced: &[u8],
) {
    dst[0..4].copy_from_slice(&full_size.to_le_bytes());
    dst[4..8].copy_from_slice(&reduced_size.to_le_bytes());
    let (marker, idx) = match unknown_reduced_idx {
        Some(i) => (1u8, i),
        None => (0u8, 0u8),
    };
    dst[8] = marker;
    dst[9] = idx;
    dst[10] = 0;
    dst[11] = 0;
    dst[REDUCER_SECTION_PROLOGUE as usize..].copy_from_slice(full_to_reduced);
}

/// Memory-mapped view of a `.kmi` file.
///
/// Slices (`offsets()`, `entries_seq_id()`, `entries_pos()`) are
/// byte-cast directly into the mmap. Lookup by k-mer hash: see
/// [`KmerIndexFile::lookup_hash`].
pub struct KmerIndexFile {
    mmap: Mmap,
    alphabet_size: u32,
    kmer_size: usize,
    table_size: u64,
    n_entries: u64,
    offsets_byte_pos: usize,
    entries_seq_id_pos: usize,
    entries_pos_pos: usize,
    /// Cached so `KmerLookup::encoder()` can return a stable reference
    /// without reconstructing per call. Reconstructed from the header
    /// at open time; all prefilter runs that touch this index get the
    /// same instance.
    encoder: KmerEncoder,
    /// Snapshot of the reducer used when this .kmi was built. Engine
    /// openers compare this against their own reducer to guarantee
    /// matrix-consistent mappings.
    reducer: ReducerSnapshot,
}

impl KmerIndexFile {
    /// Open and validate a `.kmi` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, KmiReaderError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        if file_size < KMI_HEADER_SIZE {
            return Err(KmiReaderError::TruncatedHeader(file_size, KMI_HEADER_SIZE));
        }

        // SAFETY: the data file is not mutated for the lifetime of the
        // returned mmap; callers are expected to treat the file as
        // read-only on disk.
        let mmap = unsafe { Mmap::map(&file)? };

        let hdr = &mmap[..KMI_HEADER_SIZE as usize];
        let magic = [hdr[0], hdr[1], hdr[2], hdr[3]];
        if magic != KMI_MAGIC {
            return Err(KmiReaderError::BadMagic(magic));
        }
        let version = u32_at(hdr, 4);
        if version != KMI_VERSION {
            return Err(KmiReaderError::BadVersion {
                found: version,
                supported: KMI_VERSION,
            });
        }
        let alphabet_size = u32_at(hdr, 8);
        let kmer_size = u32_at(hdr, 12) as usize;
        let table_size = u64_at(hdr, 16);
        let n_entries = u64_at(hdr, 24);
        let offsets_byte_pos = u64_at(hdr, 32);
        let entries_seq_id_pos = u64_at(hdr, 40);
        let entries_pos_pos = u64_at(hdr, 48);
        let reducer_section_size = u32_at(hdr, 56) as u64;

        // Parse reducer section (between header and offsets) before
        // checking layout positions, since `offsets_byte_pos` must now
        // account for it.
        let reducer = if reducer_section_size == 0 {
            ReducerSnapshot::None
        } else {
            if reducer_section_size < REDUCER_SECTION_PROLOGUE {
                return Err(KmiReaderError::BadReducerSection {
                    detail: format!(
                        "reducer_section_size={reducer_section_size} < prologue={REDUCER_SECTION_PROLOGUE}",
                    ),
                });
            }
            let start = KMI_HEADER_SIZE as usize;
            let end = start + reducer_section_size as usize;
            if end as u64 > file_size {
                return Err(KmiReaderError::BadReducerSection {
                    detail: format!("reducer section end={end} exceeds file size {file_size}"),
                });
            }
            let sec = &mmap[start..end];
            let full_size = u32_at(sec, 0);
            let reduced_size = u32_at(sec, 4);
            let marker = sec[8];
            let idx_byte = sec[9];
            let expected_body_len = full_size as u64;
            if reducer_section_size != REDUCER_SECTION_PROLOGUE + expected_body_len {
                return Err(KmiReaderError::BadReducerSection {
                    detail: format!(
                        "reducer_section_size={reducer_section_size} != \
                         prologue + full_size ({} + {})",
                        REDUCER_SECTION_PROLOGUE, expected_body_len,
                    ),
                });
            }
            let full_to_reduced = sec[REDUCER_SECTION_PROLOGUE as usize..].to_vec();
            let unknown_reduced_idx = match marker {
                0 => None,
                1 => Some(idx_byte),
                other => {
                    return Err(KmiReaderError::BadReducerSection {
                        detail: format!("unknown_marker={other} (expected 0 or 1)"),
                    });
                }
            };
            ReducerSnapshot::Some {
                full_size,
                reduced_size,
                unknown_reduced_idx,
                full_to_reduced,
            }
        };

        let expected_offsets = KMI_HEADER_SIZE + reducer_section_size;
        if offsets_byte_pos != expected_offsets {
            return Err(KmiReaderError::LayoutMismatch {
                field: "offsets_byte_pos",
                found: offsets_byte_pos,
                expected: expected_offsets,
            });
        }
        let expected_seq_id_pos = offsets_byte_pos + 8 * (table_size + 1);
        if entries_seq_id_pos != expected_seq_id_pos {
            return Err(KmiReaderError::LayoutMismatch {
                field: "entries_seq_id_pos",
                found: entries_seq_id_pos,
                expected: expected_seq_id_pos,
            });
        }
        let expected_pos_pos = entries_seq_id_pos + 4 * n_entries;
        if entries_pos_pos != expected_pos_pos {
            return Err(KmiReaderError::LayoutMismatch {
                field: "entries_pos_pos",
                found: entries_pos_pos,
                expected: expected_pos_pos,
            });
        }
        let expected_file_size = entries_pos_pos + 2 * n_entries;
        if file_size != expected_file_size {
            return Err(KmiReaderError::FileSizeMismatch {
                found: file_size,
                expected: expected_file_size,
            });
        }

        // Offsets tail invariant: the last offset must equal n_entries.
        let offsets_start = offsets_byte_pos as usize;
        let last_off_pos = offsets_start + 8 * table_size as usize;
        let last_off = u64_at(&mmap[..], last_off_pos);
        if last_off != n_entries {
            return Err(KmiReaderError::OffsetsTail {
                found: last_off,
                expected: n_entries,
            });
        }

        let encoder = KmerEncoder::new(alphabet_size, kmer_size);

        Ok(Self {
            mmap,
            alphabet_size,
            kmer_size,
            table_size,
            n_entries,
            offsets_byte_pos: offsets_byte_pos as usize,
            entries_seq_id_pos: entries_seq_id_pos as usize,
            entries_pos_pos: entries_pos_pos as usize,
            encoder,
            reducer,
        })
    }

    pub fn alphabet_size(&self) -> u32 {
        self.alphabet_size
    }
    pub fn kmer_size(&self) -> usize {
        self.kmer_size
    }
    pub fn table_size(&self) -> u64 {
        self.table_size
    }
    pub fn n_entries(&self) -> u64 {
        self.n_entries
    }

    /// Snapshot of the reducer used at build time.
    ///
    /// Openers that configure their own `ReducedAlphabet` should
    /// compare `ReducerSnapshot::from_reducer(Some(&mine))` against
    /// this value and refuse to proceed on mismatch — the k-mer hashes
    /// in the file are only meaningful under the mapping recorded
    /// here.
    pub fn reducer(&self) -> &ReducerSnapshot {
        &self.reducer
    }

    /// Byte slice over the `(table_size + 1) × u64 LE` offsets array.
    ///
    /// Returns a `&[u8]` rather than `&[u64]` to sidestep alignment
    /// concerns on platforms where mmap pages may not align to 8-byte
    /// boundaries. Callers decode per lookup via `u64_at`.
    fn offsets_bytes(&self) -> &[u8] {
        let start = self.offsets_byte_pos;
        let len = 8 * (self.table_size as usize + 1);
        &self.mmap[start..start + len]
    }

    fn entries_seq_id_bytes(&self) -> &[u8] {
        let start = self.entries_seq_id_pos;
        let len = 4 * self.n_entries as usize;
        &self.mmap[start..start + len]
    }

    fn entries_pos_bytes(&self) -> &[u8] {
        let start = self.entries_pos_pos;
        let len = 2 * self.n_entries as usize;
        &self.mmap[start..start + len]
    }

    /// Look up the `(seq_id, pos)` postings for a single k-mer hash.
    ///
    /// Matches [`KmerIndex::lookup_hash`]'s contract: out-of-range
    /// hashes (`>= table_size`) return an empty vec rather than
    /// panicking — a public API taking an arbitrary `u64` has to
    /// handle every value.
    ///
    /// Returns a `Vec<KmerHit>` — the SoA layout on disk forces one
    /// small owned allocation per lookup. For hot prefilter loops the
    /// cost is tiny relative to the OS page-fault of the posting
    /// bucket itself, and the API stays compatible with in-memory
    /// `KmerIndex::lookup_hash`.
    pub fn lookup_hash(&self, hash: u64) -> Vec<KmerHit> {
        if hash >= self.table_size {
            return Vec::new();
        }
        let offsets = self.offsets_bytes();
        let start = u64_at(offsets, 8 * hash as usize) as usize;
        let end = u64_at(offsets, 8 * (hash as usize + 1)) as usize;
        if end <= start {
            return Vec::new();
        }
        let seq_ids = self.entries_seq_id_bytes();
        let positions = self.entries_pos_bytes();
        (start..end)
            .map(|i| KmerHit {
                seq_id: u32_at(seq_ids, 4 * i),
                pos: u16_at(positions, 2 * i),
            })
            .collect()
    }
}

impl KmerLookup for KmerIndexFile {
    fn encoder(&self) -> &KmerEncoder {
        &self.encoder
    }

    fn for_each_hit<F: FnMut(KmerHit)>(&self, hash: u64, mut f: F) {
        // Match KmerIndex::lookup_hash: out-of-range hashes yield no
        // hits. Prior `debug_assert!` would panic in release builds
        // on any invalid hash reaching the mmap backend — behavior
        // diverged from the in-memory path behind the shared
        // KmerLookup abstraction.
        if hash >= self.table_size {
            return;
        }
        let offsets = self.offsets_bytes();
        let start = u64_at(offsets, 8 * hash as usize) as usize;
        let end = u64_at(offsets, 8 * (hash as usize + 1)) as usize;
        if end <= start {
            return;
        }
        let seq_ids = self.entries_seq_id_bytes();
        let positions = self.entries_pos_bytes();
        for i in start..end {
            f(KmerHit {
                seq_id: u32_at(seq_ids, 4 * i),
                pos: u16_at(positions, 2 * i),
            });
        }
    }
}

// ---- internal LE readers over &[u8] ----

fn u16_at(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}
fn u32_at(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}
fn u64_at(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer::{KmerEncoder, KmerIndex};
    use tempfile::tempdir;

    fn tiny_index() -> KmerIndex {
        // 13-letter reduced alphabet, k=3, three tiny targets.
        let encoder = KmerEncoder::new(13, 3);
        let targets: Vec<(u32, &[u8])> = vec![
            (1u32, &[0, 1, 2, 3, 4][..]),
            (2u32, &[2, 3, 4, 5, 6][..]),
            (3u32, &[0, 1, 2][..]),
        ];
        KmerIndex::build(encoder, targets, 12).unwrap()
    }

    #[test]
    fn round_trip_recovers_offsets_and_entries() {
        let idx = tiny_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.kmi");
        write_kmi(&idx, None, &path).unwrap();

        let file = KmerIndexFile::open(&path).unwrap();
        assert_eq!(file.alphabet_size(), idx.encoder.alphabet_size());
        assert_eq!(file.kmer_size(), idx.encoder.kmer_size());
        assert_eq!(file.table_size(), idx.encoder.table_size());
        assert_eq!(file.n_entries() as usize, idx.entries.len());
        assert_eq!(*file.reducer(), ReducerSnapshot::None);

        // Every kmer hash yields byte-identical hits between the
        // in-memory index and the mmap-backed file.
        for h in 0..idx.encoder.table_size() {
            let mem: Vec<_> = idx.lookup_hash(h).to_vec();
            let disk = file.lookup_hash(h);
            assert_eq!(
                mem, disk,
                "kmer hash {h}: in-memory and on-disk lookups diverge",
            );
        }
    }

    #[test]
    fn external_build_matches_in_memory_build_byte_for_byte() {
        // External builder must produce a .kmi that's bit-identical to
        // the in-memory `write_kmi(KmerIndex::build(...), ...)` path
        // for the same inputs. Walk order (DB index order) is the same
        // in both paths, so within-bucket entry order matches too.
        use crate::db::{DBWriter, Dbtype};
        use crate::matrix::SubstitutionMatrix;
        use crate::reduced_alphabet::ReducedAlphabet;
        use crate::sequence::Sequence;

        let dir = tempdir().unwrap();
        let db_prefix = dir.path().join("db");
        let mut w = DBWriter::create(&db_prefix, Dbtype::AMINO_ACIDS).unwrap();
        let raw: &[(u32, &[u8])] = &[
            (1, b"MKLVRQPSTNLKACDFGHIY"),
            (2, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP"),
            (3, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR"),
            (4, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ"),
        ];
        for (k, s) in raw {
            w.write_entry(*k, s).unwrap();
        }
        w.finish().unwrap();

        let alphabet = Alphabet::protein();
        let matrix = SubstitutionMatrix::blosum62();
        let reducer = ReducedAlphabet::from_matrix(&matrix, 13, Some(20)).unwrap();

        // --- In-memory path ---
        let db_for_mem = DBReader::open(&db_prefix).unwrap();
        let targets_encoded: Vec<(u32, Vec<u8>)> = db_for_mem
            .index
            .iter()
            .map(|e| {
                let payload = db_for_mem.get_payload(e);
                let full = Sequence::from_ascii(alphabet.clone(), payload).data;
                (e.key, reducer.reduce_sequence(&full))
            })
            .collect();
        let encoder = KmerEncoder::new(13, 3);
        let pairs: Vec<(u32, &[u8])> = targets_encoded
            .iter()
            .map(|(k, s)| (*k, s.as_slice()))
            .collect();
        let mem_idx =
            KmerIndex::build(encoder, pairs, reducer.unknown_reduced_idx.unwrap()).unwrap();
        let mem_path = dir.path().join("mem.kmi");
        write_kmi(&mem_idx, Some(&reducer), &mem_path).unwrap();

        // --- External path ---
        let db = DBReader::open(&db_prefix).unwrap();
        let ext_path = dir.path().join("ext.kmi");
        build_kmi_external(
            &db,
            alphabet.clone(),
            Some(&reducer),
            BuildExternalOptions {
                k: 3,
                hash_range_passes: 1,
            },
            &ext_path,
        )
        .unwrap();

        let mem_bytes = std::fs::read(&mem_path).unwrap();
        let ext_bytes = std::fs::read(&ext_path).unwrap();
        assert_eq!(
            mem_bytes.len(),
            ext_bytes.len(),
            "file sizes diverge: mem={} ext={}",
            mem_bytes.len(),
            ext_bytes.len(),
        );
        assert_eq!(
            mem_bytes, ext_bytes,
            "in-memory and external .kmi diverge byte-for-byte"
        );

        // Also verify both open cleanly and agree on per-hash lookup.
        let mem_file = KmerIndexFile::open(&mem_path).unwrap();
        let ext_file = KmerIndexFile::open(&ext_path).unwrap();
        for h in 0..mem_file.table_size() {
            assert_eq!(mem_file.lookup_hash(h), ext_file.lookup_hash(h));
        }
    }

    #[test]
    fn external_build_multi_pass_matches_single_pass() {
        // hash_range_passes > 1 must produce byte-identical output —
        // each pass is disjoint in hash range, cursors accumulate
        // across passes, and the DB walk order within a pass is
        // preserved.
        use crate::db::{DBWriter, Dbtype};
        use crate::matrix::SubstitutionMatrix;
        use crate::reduced_alphabet::ReducedAlphabet;

        let dir = tempdir().unwrap();
        let db_prefix = dir.path().join("db");
        let mut w = DBWriter::create(&db_prefix, Dbtype::AMINO_ACIDS).unwrap();
        let raw: &[(u32, &[u8])] = &[
            (1, b"MKLVRQPSTNLKACDFGHIY"),
            (2, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP"),
            (3, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR"),
            (4, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ"),
            (5, b"TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"),
        ];
        for (k, s) in raw {
            w.write_entry(*k, s).unwrap();
        }
        w.finish().unwrap();

        let alphabet = Alphabet::protein();
        let matrix = SubstitutionMatrix::blosum62();
        let reducer = ReducedAlphabet::from_matrix(&matrix, 13, Some(20)).unwrap();
        let db = DBReader::open(&db_prefix).unwrap();

        let one_pass_path = dir.path().join("one.kmi");
        build_kmi_external(
            &db,
            alphabet.clone(),
            Some(&reducer),
            BuildExternalOptions {
                k: 3,
                hash_range_passes: 1,
            },
            &one_pass_path,
        )
        .unwrap();

        for k_passes in [2, 5, 13, 100] {
            let multi_pass_path = dir.path().join(format!("multi_{k_passes}.kmi"));
            build_kmi_external(
                &db,
                alphabet.clone(),
                Some(&reducer),
                BuildExternalOptions {
                    k: 3,
                    hash_range_passes: k_passes,
                },
                &multi_pass_path,
            )
            .unwrap();
            let one_bytes = std::fs::read(&one_pass_path).unwrap();
            let multi_bytes = std::fs::read(&multi_pass_path).unwrap();
            assert_eq!(
                one_bytes, multi_bytes,
                "hash_range_passes={k_passes} produced different bytes than single-pass",
            );
        }
    }

    #[test]
    fn reducer_snapshot_round_trips_through_file() {
        use crate::matrix::SubstitutionMatrix;
        use crate::reduced_alphabet::ReducedAlphabet;
        let idx = tiny_index();
        let matrix = SubstitutionMatrix::blosum62();
        let reducer = ReducedAlphabet::from_matrix(&matrix, 13, Some(20)).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("withred.kmi");
        write_kmi(&idx, Some(&reducer), &path).unwrap();

        let file = KmerIndexFile::open(&path).unwrap();
        let got = file.reducer().clone();
        let expected = ReducerSnapshot::from_reducer(Some(&reducer));
        assert_eq!(got, expected);
    }

    #[test]
    fn lookup_out_of_range_hash_returns_empty() {
        // Matches KmerIndex::lookup_hash's empty-on-out-of-range
        // contract — prior behavior panicked via debug_assert and
        // would OOB the offsets array in release builds.
        let idx = tiny_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("range.kmi");
        write_kmi(&idx, None, &path).unwrap();

        let file = KmerIndexFile::open(&path).unwrap();
        let ts = file.table_size();
        assert!(file.lookup_hash(ts).is_empty());
        assert!(file.lookup_hash(ts + 1).is_empty());
        assert!(file.lookup_hash(u64::MAX).is_empty());

        let mut count = 0u32;
        // for_each_hit should also return empty without panicking.
        <KmerIndexFile as KmerLookup>::for_each_hit(&file, ts, |_| count += 1);
        <KmerIndexFile as KmerLookup>::for_each_hit(&file, u64::MAX, |_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nope.kmi");
        std::fs::write(&path, [0u8; 128]).unwrap();
        let err = KmerIndexFile::open(&path).err().expect("expected BadMagic");
        assert!(
            matches!(err, KmiReaderError::BadMagic(m) if m == [0, 0, 0, 0]),
            "expected BadMagic, got {err}",
        );
    }

    #[test]
    fn open_rejects_future_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("v99.kmi");
        let mut hdr = [0u8; 64];
        hdr[..4].copy_from_slice(&KMI_MAGIC);
        hdr[4..8].copy_from_slice(&99u32.to_le_bytes());
        std::fs::write(&path, hdr).unwrap();
        let err = KmerIndexFile::open(&path)
            .err()
            .expect("expected BadVersion");
        assert!(
            matches!(
                err,
                KmiReaderError::BadVersion { found: 99, supported } if supported == KMI_VERSION,
            ),
            "expected BadVersion(99, supported={}), got {err}",
            KMI_VERSION,
        );
    }

    #[test]
    fn open_rejects_truncated_file() {
        let idx = tiny_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("short.kmi");
        write_kmi(&idx, None, &path).unwrap();
        // Truncate a few bytes off the end.
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 4]).unwrap();
        let err = KmerIndexFile::open(&path)
            .err()
            .expect("expected truncation error");
        assert!(
            matches!(err, KmiReaderError::FileSizeMismatch { .. }),
            "expected FileSizeMismatch, got {err}",
        );
    }

    #[test]
    fn empty_buckets_return_empty_vec() {
        // Hash values with zero hits must return an empty slice, not
        // panic on bounds and not return stale data from a neighbor.
        let idx = tiny_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.kmi");
        write_kmi(&idx, None, &path).unwrap();
        let file = KmerIndexFile::open(&path).unwrap();
        // Some hash slots are guaranteed empty at this corpus size
        // (table_size = 13^3 = 2197 slots, only a few are populated).
        let mut empty_seen = 0;
        for h in 0..file.table_size() {
            if file.lookup_hash(h).is_empty() {
                empty_seen += 1;
            }
        }
        assert!(
            empty_seen > 0,
            "expected some empty k-mer buckets in a tiny corpus"
        );
    }
}
