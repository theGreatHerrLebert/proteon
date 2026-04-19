//! MMseqs2 on-disk DB format: reader, writer, and auxiliary files.
//!
//! See `MMseqs2/docs/DB_FORMAT_SPEC.md` for the format reference.

mod dbtype;
mod index;
mod lookup;
mod reader;
mod source;
mod writer;

pub use dbtype::{Dbtype, ExtendedDbtype};
pub use index::IndexEntry;
pub use lookup::LookupEntry;
pub use reader::DBReader;
pub use source::SourceEntry;
pub use writer::DBWriter;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("dbtype file must be exactly 4 bytes, got {0}")]
    BadDbtypeSize(usize),
    #[error("malformed index line: {0}")]
    BadIndexLine(String),
    #[error("malformed lookup line: {0}")]
    BadLookupLine(String),
    #[error("malformed source line: {0}")]
    BadSourceLine(String),
    #[error("index offset {offset} + length {length} exceeds data blob size {data_size}")]
    OutOfBounds {
        offset: u64,
        length: u64,
        data_size: u64,
    },
    #[error("compressed DB (dbtype bit 31 set) is not supported yet; use an uncompressed DB")]
    CompressedNotSupported,
}

pub type Result<T> = std::result::Result<T, DbError>;
