//! Parquet writer for batched output.
//!
//! Writes Arrow RecordBatches directly to Parquet files without
//! going through Python. Useful for large-scale extraction pipelines.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Write one or more RecordBatches to a Parquet file.
///
/// Uses Zstd compression by default for good compression ratio and speed.
pub fn write_parquet(
    path: &Path,
    schema: &Schema,
    batches: &[RecordBatch],
) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;

    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::{atom_schema, AtomBatchBuilder};

    #[test]
    fn test_write_parquet() {
        let mut builder = AtomBatchBuilder::new(1);
        builder.append(
            "1crn", 0, "A", "THR", 1, None, None, "CA", 1, Some("C"),
            17.047, 14.099, 3.625, 13.79, 1.0, false, true,
        );
        let batch = builder.finish().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("ferritin_test.parquet");
        write_parquet(&path, &atom_schema(), &[batch]).unwrap();

        // Verify file exists and is non-empty
        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > 0);
        std::fs::remove_file(&path).ok();
    }
}
