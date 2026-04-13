//! Byte-exact round-trip oracle test against upstream `mmseqs createdb`.
//!
//! Skipped (with a `println!`) if the upstream `mmseqs` binary is not found
//! at `../MMseqs2/build/src/mmseqs` or `../MMseqs2/oracle-bin/mmseqs/bin/mmseqs`,
//! and the input FASTA at `../MMseqs2/examples/DB.fasta`.

use std::path::PathBuf;
use std::process::Command;

use ferritin_search::db::{DBReader, DBWriter};
use tempfile::tempdir;

fn find_mmseqs() -> Option<PathBuf> {
    let candidates = [
        "../MMseqs2/build/src/mmseqs",
        "../MMseqs2/oracle-bin/mmseqs/bin/mmseqs",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p.canonicalize().ok()?);
        }
    }
    None
}

fn find_example_fasta() -> Option<PathBuf> {
    let p = PathBuf::from("../MMseqs2/examples/DB.fasta");
    if p.exists() { Some(p.canonicalize().ok()?) } else { None }
}

#[test]
fn rust_reader_matches_upstream_createdb_output() {
    let Some(mmseqs) = find_mmseqs() else {
        eprintln!("skipping: upstream mmseqs binary not found");
        return;
    };
    let Some(fasta) = find_example_fasta() else {
        eprintln!("skipping: examples/DB.fasta not found");
        return;
    };

    let dir = tempdir().unwrap();
    let upstream_prefix = dir.path().join("upstream");
    let status = Command::new(&mmseqs)
        .args([
            "createdb",
            fasta.to_str().unwrap(),
            upstream_prefix.to_str().unwrap(),
            "-v",
            "1",
        ])
        .status()
        .expect("run mmseqs createdb");
    assert!(status.success(), "mmseqs createdb failed");

    let r = DBReader::open(&upstream_prefix).expect("open upstream DB");
    assert_eq!(r.dbtype.base(), 0, "examples/DB.fasta is protein → AMINO_ACIDS");
    assert!(!r.is_empty());
    assert_eq!(r.len(), 20_000, "examples/DB.fasta has 20k sequences");
    assert!(r.source.is_some());
    assert!(r.lookup.is_some());

    // Every index entry fits within the data blob.
    for e in &r.index {
        let end = e.offset + e.length;
        assert!(end as usize <= r.data.len());
        // upstream always null-terminates
        assert_eq!(r.data[end as usize - 1], 0, "missing \\0 at end of entry {}", e.key);
    }

    // First entry: offset 0, length 1882 (from our format-spec verification).
    assert_eq!(r.index[0].key, 0);
    assert_eq!(r.index[0].offset, 0);
    assert_eq!(r.index[0].length, 1882);
}

#[test]
fn rust_writer_roundtrips_upstream_db_byte_exact() {
    let Some(mmseqs) = find_mmseqs() else {
        eprintln!("skipping: upstream mmseqs binary not found");
        return;
    };
    let Some(fasta) = find_example_fasta() else {
        eprintln!("skipping: examples/DB.fasta not found");
        return;
    };

    let dir = tempdir().unwrap();
    let upstream_prefix = dir.path().join("upstream");
    let status = Command::new(&mmseqs)
        .args([
            "createdb",
            fasta.to_str().unwrap(),
            upstream_prefix.to_str().unwrap(),
            "-v",
            "1",
        ])
        .status()
        .expect("run mmseqs createdb");
    assert!(status.success());

    // Do it for both the data DB and the header DB.
    for suffix in ["", "_h"] {
        let src_prefix = {
            let s = format!("{}{}", upstream_prefix.to_string_lossy(), suffix);
            PathBuf::from(s)
        };
        let dst_prefix = {
            let s = format!("{}{}", dir.path().join("ours").to_string_lossy(), suffix);
            PathBuf::from(s)
        };

        let reader = DBReader::open(&src_prefix).expect("open upstream");
        let mut writer = DBWriter::create(&dst_prefix, reader.dbtype).unwrap();
        for entry in &reader.index {
            writer.write_raw(entry.key, reader.get_raw(entry)).unwrap();
        }
        if let Some(lookup) = &reader.lookup {
            writer.write_lookup(lookup).unwrap();
        }
        if let Some(source) = &reader.source {
            writer.write_source(source).unwrap();
        }
        writer.finish().unwrap();

        // Byte-exact compare for every file that exists upstream.
        let checks: &[&str] = if suffix.is_empty() {
            &["", ".index", ".dbtype", ".lookup", ".source"]
        } else {
            &["", ".index", ".dbtype"]
        };
        for ext in checks {
            let a = {
                let s = format!("{}{}", src_prefix.to_string_lossy(), ext);
                PathBuf::from(s)
            };
            let b = {
                let s = format!("{}{}", dst_prefix.to_string_lossy(), ext);
                PathBuf::from(s)
            };
            if !a.exists() {
                continue; // e.g. .source/.lookup absent on header DB
            }
            let abytes = std::fs::read(&a).unwrap();
            let bbytes = std::fs::read(&b).unwrap();
            assert_eq!(
                abytes.len(),
                bbytes.len(),
                "size mismatch on {}: upstream={} ours={}",
                a.display(),
                abytes.len(),
                bbytes.len()
            );
            assert_eq!(abytes, bbytes, "byte mismatch on {}", a.display());
        }
    }
}
