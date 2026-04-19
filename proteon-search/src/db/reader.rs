use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::{index, lookup, source};
use super::{DbError, Dbtype, IndexEntry, LookupEntry, Result, SourceEntry};

/// A read-only view of an MMseqs2 DB at a given path prefix.
///
/// Loads `<prefix>`, `<prefix>.index`, `<prefix>.dbtype`, and — if present —
/// `<prefix>.lookup` and `<prefix>.source`. The data blob is memory-mapped
/// (not read into RAM) so opening a 27 GB UniRef50 DB is a few hundred ms
/// regardless of corpus size; the OS pages in only the pages we actually
/// read during search.
///
/// # Safety on the mmap
///
/// The underlying mmap assumes the data file is not truncated or replaced
/// while this reader is alive. That's the standard mmap contract (see
/// `memmap2::Mmap`). In practice this means: don't `mmseqs createdb`
/// over a prefix that has a live reader open on it.
pub struct DBReader {
    pub prefix: PathBuf,
    pub dbtype: Dbtype,
    pub index: Vec<IndexEntry>,
    data: DataStorage,
    pub lookup: Option<Vec<LookupEntry>>,
    pub source: Option<Vec<SourceEntry>>,
}

/// Storage for the DB's data blob.
///
/// Normal path is [`Mapped`](DataStorage::Mapped): OS-backed memory mapping,
/// near-zero open cost, pages in lazily. The [`InMemory`](DataStorage::InMemory)
/// fallback exists for zero-byte data files (a DB with no entries) where
/// `Mmap::map` would fail on Linux.
enum DataStorage {
    Mapped(Mmap),
    InMemory(Vec<u8>),
}

impl DataStorage {
    fn as_slice(&self) -> &[u8] {
        match self {
            DataStorage::Mapped(m) => &m[..],
            DataStorage::InMemory(v) => &v[..],
        }
    }
}

impl DBReader {
    /// Open the DB at `prefix`.
    ///
    /// Memory-maps the data blob; reads the index / dbtype / optional
    /// lookup + source files eagerly because they're small. Peak RSS on
    /// open is ~`index_size + ~dbtype_plus_aux_size`; the data blob
    /// stays on disk paged by the OS.
    pub fn open(prefix: impl AsRef<Path>) -> Result<Self> {
        let prefix = prefix.as_ref().to_path_buf();

        let data_path = prefix.clone();
        let index_path = with_suffix(&prefix, ".index");
        let dbtype_path = with_suffix(&prefix, ".dbtype");
        let lookup_path = with_suffix(&prefix, ".lookup");
        let source_path = with_suffix(&prefix, ".source");

        let dbtype = Dbtype::read_from_file(&dbtype_path)?;
        if dbtype.is_compressed() {
            return Err(DbError::CompressedNotSupported);
        }
        let index = index::read_all(&index_path)?;

        let data_len = std::fs::metadata(&data_path)?.len();
        let data = if data_len == 0 {
            // Empty DB — no need to mmap, and on Linux Mmap::map on a
            // zero-byte file returns EINVAL. Keep the accessor returning
            // an empty slice either way.
            DataStorage::InMemory(Vec::new())
        } else {
            let file = File::open(&data_path)?;
            // SAFETY: see the struct-level doc. We hold the Mmap for the
            // reader's lifetime and never write through it; external
            // truncation/replacement is the caller's contract to avoid.
            let mmap = unsafe { Mmap::map(&file)? };
            DataStorage::Mapped(mmap)
        };

        let lookup = if lookup_path.exists() {
            Some(lookup::read_all(&lookup_path)?)
        } else {
            None
        };
        let source = if source_path.exists() {
            Some(source::read_all(&source_path)?)
        } else {
            None
        };

        let data_size = data.as_slice().len() as u64;
        for e in &index {
            let end = e.offset.saturating_add(e.length);
            if end > data_size {
                return Err(DbError::OutOfBounds {
                    offset: e.offset,
                    length: e.length,
                    data_size,
                });
            }
        }

        Ok(Self {
            prefix,
            dbtype,
            index,
            data,
            lookup,
            source,
        })
    }

    /// Full data blob as a byte slice. Backed by the mmap when present.
    ///
    /// Replaces the old `pub data: Vec<u8>` field. Pre-memmap, external
    /// callers indexed that field directly; they should now call this
    /// accessor, which yields the same `&[u8]` view without copying.
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Get the raw payload bytes for an index entry.
    ///
    /// Returns the bytes *as stored*, including the trailing `\n\0` upstream
    /// writes after each record. Use [`DBReader::get_payload`] for the payload
    /// with the null terminator stripped.
    pub fn get_raw(&self, entry: &IndexEntry) -> &[u8] {
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        &self.data.as_slice()[start..end]
    }

    /// Get the payload with the single trailing `\0` null terminator stripped.
    /// The preceding `\n` upstream inserts is *not* stripped (it was part of
    /// the caller-supplied payload passed to DBWriter).
    pub fn get_payload(&self, entry: &IndexEntry) -> &[u8] {
        let raw = self.get_raw(entry);
        if raw.last() == Some(&0) {
            &raw[..raw.len() - 1]
        } else {
            raw
        }
    }

    /// Linear scan over the index for a key; O(n). Fine for small DBs;
    /// a later phase adds sorted-index binary search.
    pub fn get_by_key(&self, key: u32) -> Option<&IndexEntry> {
        self.index.iter().find(|e| e.key == key)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut p = prefix.as_os_str().to_owned();
    p.push(suffix);
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::super::DBWriter;
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn rejects_compressed_dbtype_with_clear_error() {
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("db");

        let dbtype = Dbtype::AMINO_ACIDS.with_compressed(true);
        let mut w = DBWriter::create(&prefix, dbtype).unwrap();
        w.write_raw(0, b"compressed-gibberish-not-real\0").unwrap();
        w.finish().unwrap();

        match DBReader::open(&prefix) {
            Err(DbError::CompressedNotSupported) => {}
            Err(e) => panic!("expected CompressedNotSupported, got other error: {e}"),
            Ok(_) => panic!("expected CompressedNotSupported, got Ok(_)"),
        }
    }

    #[test]
    fn empty_data_blob_returns_empty_slice_without_mmap_error() {
        // Linux Mmap::map on a zero-byte file is EINVAL — the reader must
        // still open cleanly with an empty data accessor.
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("empty_db");
        let writer = DBWriter::create(&prefix, Dbtype::AMINO_ACIDS).unwrap();
        writer.finish().unwrap();
        let r = DBReader::open(&prefix).unwrap();
        assert!(r.is_empty());
        assert_eq!(r.data(), &[] as &[u8]);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn mmapped_data_matches_in_memory_read() {
        // Parity check: mmap-backed data() and a direct std::fs::read of
        // the same file must yield identical bytes. Catches any accidental
        // slicing / alignment issue in the mmap path.
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("db");
        let mut w = DBWriter::create(&prefix, Dbtype::AMINO_ACIDS).unwrap();
        for (k, seq) in [
            (1u32, b"MKLVR".as_slice()),
            (2, b"WWWWWWWW"),
            (3, b"ABCDEFGH"),
        ] {
            w.write_entry(k, seq).unwrap();
        }
        w.finish().unwrap();

        let r = DBReader::open(&prefix).unwrap();
        let expected = std::fs::read(&prefix).unwrap();
        assert_eq!(r.data(), &expected[..]);

        // get_payload strips the trailing \0 that write_entry appends.
        for entry in &r.index {
            let raw = r.get_raw(entry);
            assert_eq!(raw.last(), Some(&0u8));
            let payload = r.get_payload(entry);
            assert_eq!(payload.len(), raw.len() - 1);
            assert_ne!(payload.last(), Some(&0u8));
        }
    }
}
