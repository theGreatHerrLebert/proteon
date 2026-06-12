//! Integration tests for the `proteon` analysis CLI (sasa/dssp/hbond/energy).
//!
//! These guard the CLI against three failure modes:
//!   1. Drift from the validated numbers (the SASA total is the
//!      Biopython-oracle-backed value; DSSP is the known assignment; the
//!      energy total is the connector-native charmm19_eef1 value).
//!   2. Non-deterministic batch output across thread counts.
//!   3. Silent swallowing of per-file failures.
//!
//! Parity with the Python path is structural (both call the same
//! `sasa::`/`dssp::`/`hbond::`/`forcefield::api::` entry points), so it is not
//! re-asserted here; the Python oracle suite covers the numeric ground truth.

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

fn format_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../proteon-electrostatics/tests/fixtures/format")
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

fn energy_total(args: &[&str]) -> f64 {
    let (stdout, stderr, code) = run(args);
    assert_eq!(code, 0, "energy failed: {stderr}");
    // total is the second-to-last column (n_unassigned_atoms is last).
    let header = stdout.lines().next().unwrap();
    let total_col = header.split('\t').position(|c| c == "total").unwrap();
    stdout
        .lines()
        .nth(1)
        .unwrap()
        .split('\t')
        .nth(total_col)
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn energy_units_and_default_ff() {
    let p = pdb("1crn.pdb");
    let ps = p.to_str().unwrap();
    // Default ff is charmm19_eef1 in kJ/mol (matches proteon.compute_energy).
    let kj = energy_total(&["energy", ps]);
    let kcal = energy_total(&["energy", ps, "--units", "kcal/mol"]);
    // kJ = kcal * 4.184; the connector-native charmm19_eef1 total for 1crn is
    // ~22.27 kcal/mol (=> ~93.19 kJ/mol).
    assert!(
        (kcal - 22.2734).abs() < 0.1,
        "charmm kcal total drifted: {kcal}"
    );
    assert!(
        (kj - kcal * 4.184).abs() < 0.05,
        "kJ/kcal conversion wrong: {kj} vs {kcal}"
    );
}

#[test]
fn energy_unknown_ff_fails_cleanly() {
    let p = pdb("1crn.pdb");
    let (_stdout, stderr, code) = run(&["energy", p.to_str().unwrap(), "--ff", "bogus"]);
    assert_ne!(code, 0, "unknown ff must fail");
    assert!(
        stderr.contains("unknown force field"),
        "should name the bad ff"
    );
}

fn col(stdout: &str, name: &str) -> String {
    let header = stdout.lines().next().unwrap();
    let i = header
        .split('\t')
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("no column {name} in: {header}"));
    stdout
        .lines()
        .nth(1)
        .unwrap()
        .split('\t')
        .nth(i)
        .unwrap()
        .to_string()
}

#[test]
fn protonate_writes_structure_with_hydrogens() {
    let out = std::env::temp_dir().join("proteon_cli_protonate.pdb");
    let _ = std::fs::remove_file(&out);
    let (stdout, stderr, code) = run(&[
        "protonate",
        pdb("1crn.pdb").to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "protonate failed: {stderr}");
    assert!(
        col(&stdout, "h_added").parse::<i64>().unwrap() > 0,
        "should add H"
    );
    assert_eq!(
        col(&stdout, "minimized"),
        "false",
        "protonate must not minimize"
    );
    let written = std::fs::read_to_string(&out).expect("output not written");
    assert!(written.contains("ATOM"), "output should be a PDB");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn prepare_minimizes_and_reports() {
    let out = std::env::temp_dir().join("proteon_cli_prepare.pdb");
    let _ = std::fs::remove_file(&out);
    // Small step budget keeps the test fast while still exercising the
    // minimize branch.
    let (stdout, stderr, code) = run(&[
        "prepare",
        pdb("1crn.pdb").to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--minimize-steps",
        "2",
    ]);
    assert_eq!(code, 0, "prepare failed: {stderr}");
    assert_eq!(col(&stdout, "minimized"), "true", "prepare should minimize");
    assert_eq!(col(&stdout, "ff"), "charmm19_eef1", "default ff");
    assert!(out.exists(), "prepared structure not written");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn write_command_requires_output_destination() {
    // No -o / --out-dir → clean error, no panic.
    let (_o, stderr, code) = run(&["protonate", pdb("1crn.pdb").to_str().unwrap()]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("--out-dir") || stderr.contains("-o"),
        "should explain output is required"
    );
}

#[test]
fn write_command_rejects_unknown_ff() {
    let out = std::env::temp_dir().join("proteon_cli_badff.pdb");
    let (_o, stderr, code) = run(&[
        "prepare",
        pdb("1crn.pdb").to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--ff",
        "bogus",
    ]);
    assert_ne!(code, 0);
    assert!(stderr.contains("unknown force field"));
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

#[test]
fn electrostatics_solves_off_plus_pqr() {
    let off = format_fixture("na.off");
    let pqr = format_fixture("na.pqr");
    let phi = std::env::temp_dir().join("proteon_cli_na_phi.tsv");
    let (stdout, stderr, code) = run(&[
        "electrostatics",
        "--off",
        off.to_str().unwrap(),
        "--pqr",
        pqr.to_str().unwrap(),
        "--potential-out",
        phi.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "electrostatics should succeed: {stderr}");

    // Header + one data row; rfenergy is the first column and strongly negative
    // (Born-like solvation of a +1 ion), the solve converged, n_elements = 512.
    let rfenergy: f64 = col(&stdout, "rfenergy_kj_mol").parse().expect("rfenergy");
    assert!(rfenergy < 0.0, "solvation energy should be negative, got {rfenergy}");
    assert_eq!(col(&stdout, "converged"), "true");
    assert_eq!(col(&stdout, "n_elements"), "512");

    // The potential file carries one row per vertex (258) plus a header.
    let lines = std::fs::read_to_string(&phi).expect("potential file");
    assert_eq!(lines.lines().count(), 259);
    let _ = std::fs::remove_file(&phi);
}

#[test]
fn electrostatics_fast_summation_matches_dense() {
    let off = format_fixture("na.off");
    let pqr = format_fixture("na.pqr");
    let dense = run(&[
        "electrostatics",
        "--off",
        off.to_str().unwrap(),
        "--pqr",
        pqr.to_str().unwrap(),
    ]);
    let tc = run(&[
        "electrostatics",
        "--off",
        off.to_str().unwrap(),
        "--pqr",
        pqr.to_str().unwrap(),
        "--fast-summation",
        "--fs-order",
        "8",
        "--fs-theta",
        "0.45",
    ]);
    assert_eq!(dense.2, 0);
    assert_eq!(tc.2, 0, "fast-summation solve should succeed: {}", tc.1);
    let e_dense: f64 = col(&dense.0, "rfenergy_kj_mol").parse().unwrap();
    let e_tc: f64 = col(&tc.0, "rfenergy_kj_mol").parse().unwrap();
    let rel = (e_tc - e_dense).abs() / e_dense.abs();
    assert!(rel < 1e-3, "treecode rfenergy {e_tc} off dense {e_dense} (rel {rel:.2e})");
}

#[test]
fn electrostatics_off_without_pqr_fails() {
    let off = format_fixture("na.off");
    let (_stdout, stderr, code) = run(&["electrostatics", "--off", off.to_str().unwrap()]);
    assert_ne!(code, 0, "missing --pqr must fail");
    assert!(stderr.contains("pqr"), "error should mention the missing --pqr");
}
