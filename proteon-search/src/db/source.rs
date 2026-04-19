use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use super::{DbError, Result};

/// One record in `<prefix>.source`: `file_number\tfilename\n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceEntry {
    pub file_number: u32,
    pub filename: String,
}

impl SourceEntry {
    pub fn parse_line(line: &[u8]) -> Result<Self> {
        let line = trim_newline(line);
        let text = std::str::from_utf8(line)
            .map_err(|_| DbError::BadSourceLine(String::from_utf8_lossy(line).into_owned()))?;
        let mut it = text.splitn(2, '\t');
        let file_number = it.next().and_then(|s| s.parse::<u32>().ok());
        let filename = it.next().map(|s| s.to_owned());
        match (file_number, filename) {
            (Some(file_number), Some(filename)) => Ok(Self {
                file_number,
                filename,
            }),
            _ => Err(DbError::BadSourceLine(text.to_owned())),
        }
    }

    pub fn write_line(&self, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(w, "{}\t{}", self.file_number, self.filename)
    }
}

pub(crate) fn read_all(path: impl AsRef<Path>) -> Result<Vec<SourceEntry>> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.split(b'\n') {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        out.push(SourceEntry::parse_line(&line)?);
    }
    Ok(out)
}

pub(crate) fn write_all(path: impl AsRef<Path>, entries: &[SourceEntry]) -> Result<()> {
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
        let e = SourceEntry::parse_line(b"0\tDB.fasta\n").unwrap();
        assert_eq!(
            e,
            SourceEntry {
                file_number: 0,
                filename: "DB.fasta".to_owned()
            }
        );
    }

    #[test]
    fn write_line_exact_format() {
        let mut buf = Vec::new();
        SourceEntry {
            file_number: 0,
            filename: "DB.fasta".to_owned(),
        }
        .write_line(&mut buf)
        .unwrap();
        assert_eq!(buf, b"0\tDB.fasta\n");
    }
}
