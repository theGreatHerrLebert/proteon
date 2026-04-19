//! Per-atom Arrow schema and conversion.
//!
//! Each row is one atom. This is the primary output format for large-scale
//! feature extraction — columnar, vectorized, and zero-copy to Python.

use arrow::array::{
    ArrayBuilder, BooleanBuilder, Float64Builder, Int64Builder, RecordBatch, StringBuilder,
    UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Arrow schema for per-atom data.
///
/// Columns:
/// - `structure_id` (Utf8) — source file or PDB ID
/// - `model` (UInt32) — model number (0-indexed)
/// - `chain_id` (Utf8) — chain identifier
/// - `residue_name` (Utf8) — 3-letter residue name (ALA, GLY, …)
/// - `residue_serial` (Int64) — residue sequence number
/// - `atom_name` (Utf8) — atom name (CA, CB, N, C, O, …)
/// - `atom_serial` (Int64) — atom serial number
/// - `element` (Utf8) — element symbol (C, N, O, S, …)
/// - `x` (Float64) — x coordinate (Å)
/// - `y` (Float64) — y coordinate (Å)
/// - `z` (Float64) — z coordinate (Å)
/// - `b_factor` (Float64) — temperature factor
/// - `occupancy` (Float64) — occupancy (0.0–1.0)
/// - `is_hetero` (Boolean) — HETATM flag
/// - `is_backbone` (Boolean) — backbone atom (N, CA, C, O)
pub fn atom_schema() -> Schema {
    Schema::new(vec![
        Field::new("structure_id", DataType::Utf8, false),
        Field::new("model", DataType::UInt32, false),
        Field::new("chain_id", DataType::Utf8, false),
        Field::new("residue_name", DataType::Utf8, false),
        Field::new("residue_serial", DataType::Int64, false),
        Field::new("insertion_code", DataType::Utf8, true),
        Field::new("conformer_id", DataType::Utf8, true),
        Field::new("atom_name", DataType::Utf8, false),
        Field::new("atom_serial", DataType::Int64, false),
        Field::new("element", DataType::Utf8, true),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("b_factor", DataType::Float64, false),
        Field::new("occupancy", DataType::Float64, false),
        Field::new("is_hetero", DataType::Boolean, false),
        Field::new("is_backbone", DataType::Boolean, false),
    ])
}

/// Builder for constructing atom RecordBatches incrementally.
///
/// Accumulates atoms from one or more structures, then finalizes
/// into an Arrow RecordBatch.
pub struct AtomBatchBuilder {
    structure_id: StringBuilder,
    model: UInt32Builder,
    chain_id: StringBuilder,
    residue_name: StringBuilder,
    residue_serial: Int64Builder,
    insertion_code: StringBuilder,
    conformer_id: StringBuilder,
    atom_name: StringBuilder,
    atom_serial: Int64Builder,
    element: StringBuilder,
    x: Float64Builder,
    y: Float64Builder,
    z: Float64Builder,
    b_factor: Float64Builder,
    occupancy: Float64Builder,
    is_hetero: BooleanBuilder,
    is_backbone: BooleanBuilder,
}

impl AtomBatchBuilder {
    /// Create a new builder with the given initial capacity (number of atoms).
    pub fn new(capacity: usize) -> Self {
        Self {
            structure_id: StringBuilder::with_capacity(capacity, capacity * 6),
            model: UInt32Builder::with_capacity(capacity),
            chain_id: StringBuilder::with_capacity(capacity, capacity),
            residue_name: StringBuilder::with_capacity(capacity, capacity * 3),
            residue_serial: Int64Builder::with_capacity(capacity),
            insertion_code: StringBuilder::with_capacity(capacity, capacity),
            conformer_id: StringBuilder::with_capacity(capacity, capacity),
            atom_name: StringBuilder::with_capacity(capacity, capacity * 4),
            atom_serial: Int64Builder::with_capacity(capacity),
            element: StringBuilder::with_capacity(capacity, capacity * 2),
            x: Float64Builder::with_capacity(capacity),
            y: Float64Builder::with_capacity(capacity),
            z: Float64Builder::with_capacity(capacity),
            b_factor: Float64Builder::with_capacity(capacity),
            occupancy: Float64Builder::with_capacity(capacity),
            is_hetero: BooleanBuilder::with_capacity(capacity),
            is_backbone: BooleanBuilder::with_capacity(capacity),
        }
    }

    /// Append a single atom record.
    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &mut self,
        structure_id: &str,
        model: u32,
        chain_id: &str,
        residue_name: &str,
        residue_serial: i64,
        insertion_code: Option<&str>,
        conformer_id: Option<&str>,
        atom_name: &str,
        atom_serial: i64,
        element: Option<&str>,
        x: f64,
        y: f64,
        z: f64,
        b_factor: f64,
        occupancy: f64,
        is_hetero: bool,
        is_backbone: bool,
    ) {
        self.structure_id.append_value(structure_id);
        self.model.append_value(model);
        self.chain_id.append_value(chain_id);
        self.residue_name.append_value(residue_name);
        self.residue_serial.append_value(residue_serial);
        match insertion_code {
            Some(ic) => self.insertion_code.append_value(ic),
            None => self.insertion_code.append_null(),
        }
        match conformer_id {
            Some(cf) => self.conformer_id.append_value(cf),
            None => self.conformer_id.append_null(),
        }
        self.atom_name.append_value(atom_name);
        self.atom_serial.append_value(atom_serial);
        match element {
            Some(e) => self.element.append_value(e),
            None => self.element.append_null(),
        }
        self.x.append_value(x);
        self.y.append_value(y);
        self.z.append_value(z);
        self.b_factor.append_value(b_factor);
        self.occupancy.append_value(occupancy);
        self.is_hetero.append_value(is_hetero);
        self.is_backbone.append_value(is_backbone);
    }

    /// Number of atoms currently in the builder.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Finalize into an Arrow RecordBatch.
    pub fn finish(mut self) -> anyhow::Result<RecordBatch> {
        let batch = RecordBatch::try_new(
            Arc::new(atom_schema()),
            vec![
                Arc::new(self.structure_id.finish()),
                Arc::new(self.model.finish()),
                Arc::new(self.chain_id.finish()),
                Arc::new(self.residue_name.finish()),
                Arc::new(self.residue_serial.finish()),
                Arc::new(self.insertion_code.finish()),
                Arc::new(self.conformer_id.finish()),
                Arc::new(self.atom_name.finish()),
                Arc::new(self.atom_serial.finish()),
                Arc::new(self.element.finish()),
                Arc::new(self.x.finish()),
                Arc::new(self.y.finish()),
                Arc::new(self.z.finish()),
                Arc::new(self.b_factor.finish()),
                Arc::new(self.occupancy.finish()),
                Arc::new(self.is_hetero.finish()),
                Arc::new(self.is_backbone.finish()),
            ],
        )?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_fields() {
        let schema = atom_schema();
        assert_eq!(schema.fields().len(), 17);
        assert_eq!(schema.field(0).name(), "structure_id");
        assert_eq!(schema.field(5).name(), "insertion_code");
        assert_eq!(schema.field(6).name(), "conformer_id");
        assert_eq!(schema.field(10).name(), "x");
    }

    #[test]
    fn test_build_batch() {
        let mut builder = AtomBatchBuilder::new(2);
        builder.append(
            "1crn",
            0,
            "A",
            "THR",
            1,
            None,
            None,
            "CA",
            1,
            Some("C"),
            17.047,
            14.099,
            3.625,
            13.79,
            1.0,
            false,
            true,
        );
        builder.append(
            "1crn",
            0,
            "A",
            "THR",
            1,
            None,
            None,
            "CB",
            2,
            Some("C"),
            16.967,
            12.784,
            4.338,
            13.25,
            1.0,
            false,
            false,
        );

        let batch = builder.finish().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 17);
    }

    #[test]
    fn test_empty_batch() {
        let builder = AtomBatchBuilder::new(0);
        assert!(builder.is_empty());
        let batch = builder.finish().unwrap();
        assert_eq!(batch.num_rows(), 0);
    }
}
