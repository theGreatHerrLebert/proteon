use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::{lookup, source};
use super::{Dbtype, IndexEntry, LookupEntry, Result, SourceEntry};

/// Streaming writer for an MMseqs2 DB at a given path prefix.
///
/// Matches upstream `DBWriter` layout: writes the data blob and the ASCII index
/// as records come in, then `finish()` writes the `.dbtype`, and optionally
/// `.lookup` / `.source`.
pub struct DBWriter {
    prefix: PathBuf,
    dbtype: Dbtype,
    data: BufWriter<File>,
    index_file: BufWriter<File>,
    offset: u64,
    count: u64,
}

impl DBWriter {
    /// Create a new DB at `prefix`. Truncates any existing files.
    pub fn create(prefix: impl AsRef<Path>, dbtype: Dbtype) -> Result<Self> {
        let prefix = prefix.as_ref().to_path_buf();
        let data = BufWriter::new(File::create(&prefix)?);
        let index_file = BufWriter::new(File::create(with_suffix(&prefix, ".index"))?);
        Ok(Self {
            prefix,
            dbtype,
            data,
            index_file,
            offset: 0,
            count: 0,
        })
    }

    /// Append a record.
    ///
    /// `payload` is written verbatim to the data blob, followed by a single
    /// `\0` null terminator (matching upstream `DBWriter::writeEnd`). The index
    /// entry's `length` is `payload.len() + 1` to include that null byte.
    ///
    /// Upstream `createdb` passes payloads that already end in `\n`, so the
    /// on-disk record ends with `\n\0`. Byte-exact round-tripping of an
    /// upstream-produced DB should pass the payload obtained via
    /// [`super::DBReader::get_payload`] (which strips the `\0` but keeps the `\n`).
    pub fn write_entry(&mut self, key: u32, payload: &[u8]) -> Result<()> {
        self.data.write_all(payload)?;
        self.data.write_all(&[0u8])?;
        let length = payload.len() as u64 + 1;
        IndexEntry {
            key,
            offset: self.offset,
            length,
        }
        .write_line(&mut self.index_file)?;
        self.offset += length;
        self.count += 1;
        Ok(())
    }

    /// Append an entry at an explicit (offset, length), writing raw bytes with
    /// no null terminator appended. Used for byte-exact round-tripping where
    /// the null terminator is already included in the caller's buffer.
    pub fn write_raw(&mut self, key: u32, raw: &[u8]) -> Result<()> {
        let length = raw.len() as u64;
        self.data.write_all(raw)?;
        IndexEntry {
            key,
            offset: self.offset,
            length,
        }
        .write_line(&mut self.index_file)?;
        self.offset += length;
        self.count += 1;
        Ok(())
    }

    pub fn write_lookup(&self, entries: &[LookupEntry]) -> Result<()> {
        lookup::write_all(with_suffix(&self.prefix, ".lookup"), entries)
    }

    pub fn write_source(&self, entries: &[SourceEntry]) -> Result<()> {
        source::write_all(with_suffix(&self.prefix, ".source"), entries)
    }

    /// Flush buffers and emit the `.dbtype` file. Consumes the writer.
    pub fn finish(mut self) -> Result<()> {
        self.data.flush()?;
        self.index_file.flush()?;
        drop(self.data);
        drop(self.index_file);
        self.dbtype
            .write_to_file(with_suffix(&self.prefix, ".dbtype"))?;
        Ok(())
    }

    pub fn entry_count(&self) -> u64 {
        self.count
    }
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut p = prefix.as_os_str().to_owned();
    p.push(suffix);
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::super::index as index_mod;
    use super::super::DBReader;
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn write_then_read_two_entries() {
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("db");

        let mut w = DBWriter::create(&prefix, Dbtype::AMINO_ACIDS).unwrap();
        w.write_entry(0, b"ACDEFG\n").unwrap();
        w.write_entry(1, b"HIKLMN\n").unwrap();
        w.finish().unwrap();

        let r = DBReader::open(&prefix).unwrap();
        assert_eq!(r.dbtype, Dbtype::AMINO_ACIDS);
        assert_eq!(r.len(), 2);

        let e0 = r.get_by_key(0).unwrap();
        assert_eq!(e0.offset, 0);
        assert_eq!(e0.length, 8); // "ACDEFG\n" + \0
        assert_eq!(r.get_raw(e0), b"ACDEFG\n\0");
        assert_eq!(r.get_payload(e0), b"ACDEFG\n");

        let e1 = r.get_by_key(1).unwrap();
        assert_eq!(e1.offset, 8);
        assert_eq!(e1.length, 8);
    }

    #[test]
    fn write_raw_preserves_bytes_exactly() {
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("db");

        let mut w = DBWriter::create(&prefix, Dbtype::GENERIC_DB).unwrap();
        w.write_raw(0, b"some arbitrary bytes\0").unwrap();
        w.finish().unwrap();

        let idx = index_mod::read_all(prefix.with_file_name(format!(
            "{}.index",
            prefix.file_name().unwrap().to_string_lossy()
        )))
        .unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0].length, 21);
    }
}
