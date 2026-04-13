use std::path::{Path, PathBuf};

use super::{index, lookup, source};
use super::{DbError, Dbtype, IndexEntry, LookupEntry, Result, SourceEntry};

/// A read-only view of an MMseqs2 DB at a given path prefix.
///
/// Loads `<prefix>`, `<prefix>.index`, `<prefix>.dbtype`, and — if present —
/// `<prefix>.lookup` and `<prefix>.source`.
pub struct DBReader {
    pub prefix: PathBuf,
    pub dbtype: Dbtype,
    pub index: Vec<IndexEntry>,
    pub data: Vec<u8>,
    pub lookup: Option<Vec<LookupEntry>>,
    pub source: Option<Vec<SourceEntry>>,
}

impl DBReader {
    /// Open the DB at `prefix`. Loads everything eagerly into memory for phase 1;
    /// memmap + zstd come in later phases.
    pub fn open(prefix: impl AsRef<Path>) -> Result<Self> {
        let prefix = prefix.as_ref().to_path_buf();

        let data_path = prefix.clone();
        let index_path = with_suffix(&prefix, ".index");
        let dbtype_path = with_suffix(&prefix, ".dbtype");
        let lookup_path = with_suffix(&prefix, ".lookup");
        let source_path = with_suffix(&prefix, ".source");

        let dbtype = Dbtype::read_from_file(&dbtype_path)?;
        let index = index::read_all(&index_path)?;
        let data = std::fs::read(&data_path)?;

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

        let data_size = data.len() as u64;
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

        Ok(Self { prefix, dbtype, index, data, lookup, source })
    }

    /// Get the raw payload bytes for an index entry.
    ///
    /// Returns the bytes *as stored*, including the trailing `\n\0` upstream
    /// writes after each record. Use [`DBReader::get_payload`] for the payload
    /// with the null terminator stripped.
    pub fn get_raw(&self, entry: &IndexEntry) -> &[u8] {
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        &self.data[start..end]
    }

    /// Get the payload with the single trailing `\0` null terminator stripped.
    /// The preceding `\n` upstream inserts is *not* stripped (it was part of
    /// the caller-supplied payload passed to DBWriter).
    pub fn get_payload(&self, entry: &IndexEntry) -> &[u8] {
        let raw = self.get_raw(entry);
        if raw.last() == Some(&0) { &raw[..raw.len() - 1] } else { raw }
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
