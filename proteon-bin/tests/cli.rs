//! Integration tests for the `proteon` analysis CLI (Phase 1: sasa/dssp/hbond).
//!
//! These guard the CLI against three failure modes:
//!   1. Drift from the validated numbers (the SASA total is the
//!      Biopython-oracle-backed value; DSSP is the known assignment).
//!   2. Non-deterministic batch output across thread counts.
//!   3. Silent swallowing of per-file failures.
//!
//! Parity with the Python path is structural (both call the same
//! `sasa::`/`dssp::`/`hbond::` entry points), so it is not re-asserted here;
//! the Python oracle suite covers the numeric ground truth.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_proteon")
}

fn pdb(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../test-pdbs")
        .join(name)
}

fn run(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("failed to spawn proteon");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn sasa_total_matches_oracle() {
    let p = pdb("1crn.pdb");
    let (stdout, _stderr, code) = run(&["sasa", p.to_str().unwrap()]);
    assert_eq!(code, 0, "sasa should succeed on 1crn");
    // Header + one data row.
    let value: f64 = stdout
        .lines()
        .nth(1)
        .and_then(|l| l.split('\t').next_back())
        .and_then(|s| s.parse().ok())
        .expect("could not parse total SASA from output");
    // Biopython-oracle-backed total for 1crn is ~2970.15 Å².
    assert!(
        (value - 2970.15).abs() < 0.1,
        "1crn total SASA drifted: got {value}"
    );
}

#[test]
fn dssp_golden() {
    let p = pdb("1crn.pdb");
    let (stdout, _stderr, code) = run(&["dssp", p.to_str().unwrap()]);
    assert_eq!(code, 0);
    let ss = stdout
        .lines()
        .nth(1)
        .and_then(|l| l.split('\t').next_back())
        .unwrap_or("");
    assert_eq!(ss, "CEECSSHHHHHHHHHHHTTTCCHHHHHHHHSCEECSSSCCCTTSCC");
}

#[test]
fn json_format_is_well_formed() {
    let p = pdb("1crn.pdb");
    let (stdout, _stderr, code) = run(&["sasa", p.to_str().unwrap(), "--format", "json"]);
    assert_eq!(code, 0);
    let trimmed = stdout.trim();
    assert!(trimmed.starts_with('['), "json output should be an array");
    assert!(trimmed.ends_with(']'));
    assert!(trimmed.contains("\"total_sasa\""));
    assert!(trimmed.contains("\"file\""));
}

#[test]
fn batch_output_is_thread_count_independent() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-pdbs");
    let d = dir.to_str().unwrap();
    let (serial, _, c1) = run(&["sasa", d, "-j", "1"]);
    let (parallel, _, c2) = run(&["sasa", d, "-j", "4"]);
    assert_eq!(c1, 0);
    assert_eq!(c2, 0);
    assert_eq!(
        serial, parallel,
        "batch output must be identical regardless of thread count"
    );
    // Sorted input order: 1aaj before 1crn.
    let pos_aaj = serial.find("1aaj").expect("expected 1aaj in batch output");
    let pos_crn = serial
        .find("1crn.pdb")
        .expect("expected 1crn in batch output");
    assert!(pos_aaj < pos_crn, "batch output should be in sorted order");
}

#[test]
fn failure_is_isolated_and_signalled() {
    // Write a structurally-empty "PDB" that fails to load.
    let bad = std::env::temp_dir().join("proteon_cli_bad_fixture.pdb");
    std::fs::write(&bad, "this is not a pdb\n").unwrap();
    let good = pdb("1crn.pdb");

    let (stdout, stderr, code) = run(&["sasa", good.to_str().unwrap(), bad.to_str().unwrap()]);

    // The good structure still produces output...
    assert!(
        stdout.contains("1crn.pdb"),
        "good input must still be reported"
    );
    // ...the bad one is reported on stderr...
    assert!(
        stderr.contains("ERROR"),
        "bad input must be reported to stderr"
    );
    // ...and a partial failure is signalled with a nonzero exit.
    assert_eq!(code, 1, "any per-file failure must exit nonzero");

    let _ = std::fs::remove_file(&bad);
}
