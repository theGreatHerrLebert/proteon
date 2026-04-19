//! Per-structure Arrow schema and conversion.
//!
//! Each row is one structure (PDB file). Useful for metadata catalogs
//! and summary statistics over large collections.

use arrow::array::{ArrayBuilder, Int64Builder, RecordBatch, StringBuilder, UInt32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Arrow schema for per-structure summary data.
///
/// Columns:
/// - `structure_id` (Utf8) — source file or PDB ID
/// - `atom_count` (Int64)
/// - `residue_count` (Int64)
/// - `chain_count` (UInt32)
/// - `model_count` (UInt32)
/// - `chains` (Utf8) — comma-separated chain IDs
pub fn structure_schema() -> Schema {
    Schema::new(vec![
        Field::new("structure_id", DataType::Utf8, false),
        Field::new("atom_count", DataType::Int64, false),
        Field::new("residue_count", DataType::Int64, false),
        Field::new("chain_count", DataType::UInt32, false),
        Field::new("model_count", DataType::UInt32, false),
        Field::new("chains", DataType::Utf8, false),
    ])
}

/// Builder for structure summary RecordBatches.
pub struct StructureBatchBuilder {
    structure_id: StringBuilder,
    atom_count: Int64Builder,
    residue_count: Int64Builder,
    chain_count: UInt32Builder,
    model_count: UInt32Builder,
    chains: StringBuilder,
}

impl StructureBatchBuilder {
    /// Create a new builder.
    pub fn new(capacity: usize) -> Self {
        Self {
            structure_id: StringBuilder::with_capacity(capacity, capacity * 6),
            atom_count: Int64Builder::with_capacity(capacity),
            residue_count: Int64Builder::with_capacity(capacity),
            chain_count: UInt32Builder::with_capacity(capacity),
            model_count: UInt32Builder::with_capacity(capacity),
            chains: StringBuilder::with_capacity(capacity, capacity * 10),
        }
    }

    /// Append a structure summary record.
    pub fn append(
        &mut self,
        structure_id: &str,
        atom_count: i64,
        residue_count: i64,
        chain_count: u32,
        model_count: u32,
        chains: &str,
    ) {
        self.structure_id.append_value(structure_id);
        self.atom_count.append_value(atom_count);
        self.residue_count.append_value(residue_count);
        self.chain_count.append_value(chain_count);
        self.model_count.append_value(model_count);
        self.chains.append_value(chains);
    }

    /// Number of structures in the builder.
    pub fn len(&self) -> usize {
        self.atom_count.len()
    }

    /// Whether the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Finalize into a RecordBatch.
    pub fn finish(mut self) -> anyhow::Result<RecordBatch> {
        let batch = RecordBatch::try_new(
            Arc::new(structure_schema()),
            vec![
                Arc::new(self.structure_id.finish()),
                Arc::new(self.atom_count.finish()),
                Arc::new(self.residue_count.finish()),
                Arc::new(self.chain_count.finish()),
                Arc::new(self.model_count.finish()),
                Arc::new(self.chains.finish()),
            ],
        )?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_batch() {
        let mut builder = StructureBatchBuilder::new(1);
        builder.append("1crn", 327, 46, 1, 1, "A");
        let batch = builder.finish().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 6);
    }
}
