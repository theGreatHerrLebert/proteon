//! MMseqs2-compatible sequence search engine.
//!
//! Phased port of the MMseqs2 search path. Phase 1: on-disk DB I/O compatible
//! with upstream `mmseqs createdb` output, byte-exact round-trip.

pub mod db;
