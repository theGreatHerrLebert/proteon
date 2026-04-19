use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use super::{DbError, Result};

/// One record in `<prefix>.index`.
///
/// Serialized as `key\toffset\tlength\n` (ASCII decimal).
/// `length` is the byte length of the payload in the data blob at `offset`,
/// **including** any trailing `\n\0` upstream writes after each record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    pub key: u32,
    pub offset: u64,
    pub length: u64,
}

impl IndexEntry {
    pub fn parse_line(line: &[u8]) -> Result<Self> {
        let line = trim_newline(line);
        let text = std::str::from_utf8(line)
            .map_err(|_| DbError::BadIndexLine(String::from_utf8_lossy(line).into_owned()))?;
        let mut it = text.split('\t');
        let key = it.next().and_then(|s| s.parse::<u32>().ok());
        let offset = it.next().and_then(|s| s.parse::<u64>().ok());
        let length = it.next().and_then(|s| s.parse::<u64>().ok());
        let extra = it.next();
        match (key, offset, length, extra) {
            (Some(key), Some(offset), Some(length), None) => Ok(Self {
                key,
                offset,
                length,
            }),
            _ => Err(DbError::BadIndexLine(text.to_owned())),
        }
    }

    pub fn write_line(&self, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(w, "{}\t{}\t{}", self.key, self.offset, self.length)
    }
}

pub(super) fn read_all(path: impl AsRef<Path>) -> Result<Vec<IndexEntry>> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.split(b'\n') {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        out.push(IndexEntry::parse_line(&line)?);
    }
    Ok(out)
}

#[allow(dead_code)] // public batch helper; used in tests, exposed for external callers
pub(super) fn write_all(path: impl AsRef<Path>, entries: &[IndexEntry]) -> Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path.as_ref())?);
    for e in entries {
        e.write_line(&mut file)?;
    }
    file.flush()?;
    Ok(())
}

fn trim_newline(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    while end > 0 && (line[end - 1] == b'\n' || line[end - 1] == b'\r') {
        end -= 1;
    }
    &line[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_first_reference_entry() {
        let e = IndexEntry::parse_line(b"0\t0\t1882\n").unwrap();
        assert_eq!(
            e,
            IndexEntry {
                key: 0,
                offset: 0,
                length: 1882
            }
        );
    }

    #[test]
    fn parse_last_reference_entry() {
        let e = IndexEntry::parse_line(b"19999\t9095261\t308\n").unwrap();
        assert_eq!(
            e,
            IndexEntry {
                key: 19999,
                offset: 9095261,
                length: 308
            }
        );
    }

    #[test]
    fn parse_without_trailing_newline() {
        let e = IndexEntry::parse_line(b"42\t1000\t50").unwrap();
        assert_eq!(
            e,
            IndexEntry {
                key: 42,
                offset: 1000,
                length: 50
            }
        );
    }

    #[test]
    fn rejects_wrong_column_count() {
        assert!(IndexEntry::parse_line(b"0\t0\n").is_err());
        assert!(IndexEntry::parse_line(b"0\t0\t1\t2\n").is_err());
    }

    #[test]
    fn write_line_exact_format() {
        let mut buf = Vec::new();
        IndexEntry {
            key: 19999,
            offset: 9095261,
            length: 308,
        }
        .write_line(&mut buf)
        .unwrap();
        assert_eq!(buf, b"19999\t9095261\t308\n");
    }

    #[test]
    fn roundtrip_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let entries = vec![
            IndexEntry {
                key: 0,
                offset: 0,
                length: 1882,
            },
            IndexEntry {
                key: 1,
                offset: 1882,
                length: 208,
            },
            IndexEntry {
                key: 2,
                offset: 2090,
                length: 449,
            },
        ];
        write_all(tmp.path(), &entries).unwrap();
        let read = read_all(tmp.path()).unwrap();
        assert_eq!(entries, read);
    }
}
