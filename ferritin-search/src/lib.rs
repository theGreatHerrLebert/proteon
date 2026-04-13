//! MMseqs2-compatible sequence search engine.
//!
//! Phased port of the MMseqs2 search path:
//!  - Phase 1: on-disk DB I/O compatible with upstream `mmseqs createdb`.
//!  - Phase 2.1: alphabets, substitution matrices, sequence encoding.

pub mod alphabet;
pub mod db;
pub mod gapped;
pub mod kmer;
pub mod kmer_generator;
pub mod matrix;
pub mod prefilter;
pub mod reduced_alphabet;
pub mod search;
pub mod sequence;
pub mod ungapped;
