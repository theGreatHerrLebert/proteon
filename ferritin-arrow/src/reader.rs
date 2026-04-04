//! Parquet reader for atom-level RecordBatches.
//!
//! Reads Parquet files written by [`crate::writer::write_parquet`] back into
//! Arrow RecordBatches, which can then be converted to PDB structures via
//! [`crate::convert::atom_batch_to_pdbs`].

use std::fs::File;
use std::path::Path;

use arrow::array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Read all rows from a Parquet file into a single RecordBatch.
pub fn read_parquet(path: &Path) -> anyhow::Result<RecordBatch> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }

    if batches.is_empty() {
        anyhow::bail!("Parquet file contains no rows");
    }

    if batches.len() == 1 {
        Ok(batches.into_iter().next().unwrap())
    } else {
        let schema = batches[0].schema();
        Ok(arrow::compute::concat_batches(&schema, &batches)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::{atom_schema, AtomBatchBuilder};
    use crate::writer::write_parquet;

    #[test]
    fn test_roundtrip_parquet() {
        let mut builder = AtomBatchBuilder::new(2);
        builder.append(
            "1crn", 0, "A", "THR", 1, None, None, "CA", 1, Some("C"),
            17.047, 14.099, 3.625, 13.79, 1.0, false, true,
        );
        builder.append(
            "1crn", 0, "A", "THR", 1, None, None, "CB", 2, Some("C"),
            16.967, 12.784, 4.338, 10.80, 1.0, false, false,
        );
        let batch = builder.finish().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("ferritin_reader_test.parquet");
        write_parquet(&path, &atom_schema(), &[batch.clone()]).unwrap();

        let read_back = read_parquet(&path).unwrap();
        assert_eq!(read_back.num_rows(), 2);
        assert_eq!(read_back.num_columns(), batch.num_columns());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_missing_file() {
        let result = read_parquet(Path::new("/nonexistent/file.parquet"));
        assert!(result.is_err());
    }
}
