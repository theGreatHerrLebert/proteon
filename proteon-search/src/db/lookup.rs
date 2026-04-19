use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use super::{DbError, Result};

/// One record in `<prefix>.lookup`.
///
/// Serialized as `key\taccession\tfile_number\n` (ASCII decimal for key and file_number).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupEntry {
    pub key: u32,
    pub accession: String,
    pub file_number: u32,
}

impl LookupEntry {
    pub fn parse_line(line: &[u8]) -> Result<Self> {
        let line = trim_newline(line);
        let text = std::str::from_utf8(line)
            .map_err(|_| DbError::BadLookupLine(String::from_utf8_lossy(line).into_owned()))?;
        let mut it = text.split('\t');
        let key = it.next().and_then(|s| s.parse::<u32>().ok());
        let accession = it.next().map(|s| s.to_owned());
        let file_number = it.next().and_then(|s| s.parse::<u32>().ok());
        let extra = it.next();
        match (key, accession, file_number, extra) {
            (Some(key), Some(accession), Some(file_number), None) => Ok(Self {
                key,
                accession,
                file_number,
            }),
            _ => Err(DbError::BadLookupLine(text.to_owned())),
        }
    }

    pub fn write_line(&self, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(w, "{}\t{}\t{}", self.key, self.accession, self.file_number)
    }
}

pub(crate) fn read_all(path: impl AsRef<Path>) -> Result<Vec<LookupEntry>> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.split(b'\n') {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        out.push(LookupEntry::parse_line(&line)?);
    }
    Ok(out)
}

pub(crate) fn write_all(path: impl AsRef<Path>, entries: &[LookupEntry]) -> Result<()> {
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
    fn parse_reference_entry() {
        let e = LookupEntry::parse_line(b"0\tW0FSK4\t0\n").unwrap();
        assert_eq!(
            e,
            LookupEntry {
                key: 0,
                accession: "W0FSK4".to_owned(),
                file_number: 0
            }
        );
    }

    #[test]
    fn write_line_exact_format() {
        let mut buf = Vec::new();
        LookupEntry {
            key: 0,
            accession: "W0FSK4".to_owned(),
            file_number: 0,
        }
        .write_line(&mut buf)
        .unwrap();
        assert_eq!(buf, b"0\tW0FSK4\t0\n");
    }
}
