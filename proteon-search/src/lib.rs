//! MMseqs2-compatible sequence search engine.
//!
//! Phased port of the MMseqs2 search path:
//!  - Phase 1: on-disk DB I/O compatible with upstream `mmseqs createdb`.
//!  - Phase 2.1: alphabets, substitution matrices, sequence encoding.

pub mod alphabet;
pub mod db;
pub mod gapped;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod kmer;
pub mod kmer_generator;
pub mod kmer_index_file;
pub mod matrix;
pub mod msa;
pub mod padded_db;
pub mod prefilter;
pub mod pssm;
pub mod reduced_alphabet;
pub mod search;
pub mod sequence;
pub mod ungapped;
