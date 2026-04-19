use std::path::Path;

use super::{DbError, Result};

/// 4-byte little-endian int32 stored in `<prefix>.dbtype`.
///
/// Bit layout (matches upstream `Parameters::DBTYPE_MASK = 0x0000FFFF` and
/// `DBReader::getExtendedDbtype`):
/// - bit 31:      compression flag (1 = zstd-compressed payloads)
/// - bits 30..17: extended flags (14 bits; see [`ExtendedDbtype`]). Upstream
///   accesses these via `(u32 >> 16) & 0x7FFE`.
/// - bit 16:      reserved (cleared by upstream's `0x7FFE` extended mask)
/// - bits 15..0:  base dbtype (16 bits; see constants on [`Dbtype`])
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dbtype(pub i32);

impl Dbtype {
    pub const AMINO_ACIDS: Self = Self(0);
    pub const NUCLEOTIDES: Self = Self(1);
    pub const HMM_PROFILE: Self = Self(2);
    pub const ALIGNMENT_RES: Self = Self(5);
    pub const CLUSTER_RES: Self = Self(6);
    pub const PREFILTER_RES: Self = Self(7);
    pub const TAXONOMICAL_RESULT: Self = Self(8);
    pub const INDEX_DB: Self = Self(9);
    pub const CA3M_DB: Self = Self(10);
    pub const MSA_DB: Self = Self(11);
    pub const GENERIC_DB: Self = Self(12);
    pub const OMIT_FILE: Self = Self(13);
    pub const PREFILTER_REV_RES: Self = Self(14);
    pub const OFFSET_DB: Self = Self(15);

    pub fn base(self) -> i32 {
        self.0 & 0x0000_FFFF
    }

    pub fn is_compressed(self) -> bool {
        (self.0 as u32) & (1 << 31) != 0
    }

    pub fn extended(self) -> ExtendedDbtype {
        ExtendedDbtype(((self.0 as u32 >> 16) & 0x7FFE) as u16)
    }

    pub fn with_extended(self, ext: ExtendedDbtype) -> Self {
        Self(self.0 | ((ext.0 as i32 & 0x7FFE) << 16))
    }

    pub fn with_compressed(self, compressed: bool) -> Self {
        let mut v = self.0 as u32;
        if compressed {
            v |= 1 << 31;
        } else {
            v &= !(1 << 31);
        }
        Self(v as i32)
    }

    pub fn read_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        if bytes.len() != 4 {
            return Err(DbError::BadDbtypeSize(bytes.len()));
        }
        Ok(Self(i32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ])))
    }

    pub fn write_to_file(self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path.as_ref(), self.0.to_le_bytes())?;
        Ok(())
    }
}

/// Extended dbtype flags occupying bits 17..30 of the raw dbtype int.
/// The bit values here are the pre-shift flag values as defined by upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExtendedDbtype(pub u16);

impl ExtendedDbtype {
    pub const NONE: Self = Self(0);
    pub const COMPRESSED: Self = Self(1);
    pub const INDEX_NEED_SRC: Self = Self(2);
    pub const CONTEXT_PSEUDO_COUNTS: Self = Self(4);
    pub const GPU: Self = Self(8);
    pub const SET: Self = Self(16);

    pub fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amino_acids_on_disk_is_four_zero_bytes() {
        let d = Dbtype::AMINO_ACIDS;
        assert_eq!(d.0.to_le_bytes(), [0, 0, 0, 0]);
        assert!(!d.is_compressed());
        assert_eq!(d.base(), 0);
    }

    #[test]
    fn generic_db_on_disk_is_0c_le() {
        // upstream's `targetDB_h.dbtype` produces bytes 0x0c 0x00 0x00 0x00
        let d = Dbtype::GENERIC_DB;
        assert_eq!(d.0.to_le_bytes(), [0x0c, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn roundtrip_dbtype_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let d = Dbtype::AMINO_ACIDS.with_extended(ExtendedDbtype::GPU);
        d.write_to_file(tmp.path()).unwrap();
        let read = Dbtype::read_from_file(tmp.path()).unwrap();
        assert_eq!(d, read);
        assert!(read.extended().contains(ExtendedDbtype::GPU));
        assert_eq!(read.base(), 0);
    }

    #[test]
    fn compressed_bit_roundtrips() {
        let d = Dbtype::AMINO_ACIDS.with_compressed(true);
        assert!(d.is_compressed());
        assert_eq!(d.base(), 0);
        let cleared = d.with_compressed(false);
        assert!(!cleared.is_compressed());
    }
}
