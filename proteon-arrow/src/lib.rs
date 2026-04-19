//! Arrow RecordBatch output for proteon.
//!
//! Converts protein structures into columnar Arrow format for zero-copy
//! interop with Pandas, Polars, DuckDB, Spark, and Parquet files.
//!
//! # Design
//!
//! Proteon is a compute kernel — it produces data, your infrastructure
//! moves it. Arrow is the universal interchange format:
//!
//! - **Zero-copy** to Python via `pyarrow.RecordBatch`
//! - **Direct Parquet** output for large-scale pipelines
//! - **Streaming** via `RecordBatchReader` for constant-memory processing
//!
//! # Schemas
//!
//! Two schemas are provided:
//!
//! - [`atom_schema`] — per-atom records (coords, element, residue, chain, B-factor, …)
//! - [`structure_schema`] — per-structure summary (atom count, residue count, chains, …)

pub mod atom;
pub mod convert;
pub mod structure;

#[cfg(feature = "parquet")]
pub mod reader;

#[cfg(feature = "parquet")]
pub mod writer;
