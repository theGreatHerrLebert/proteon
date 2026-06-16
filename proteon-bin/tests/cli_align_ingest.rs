//! Integration tests for the `tmalign`, `usalign`, and `ingest` CLIs.
//!
//! The `proteon` analysis CLI is covered by `cli.rs`; these guard the three
//! remaining product surfaces — structure alignment and the bulk
//! structure→Parquet pipeline — against:
//!   1. Output-format drift (the `--outfmt 2` tabular contract, parsed by
//!      downstream tools; the Parquet schema + row counts).
//!   2. Numeric drift (a self-alignment is exactly TM=1.0, RMSD=0).
//!   3. Failure modes on bad / missing input (clean nonzero exit, isolated
//!      per-file failures, and — for `ingest` — an empty all-failed run
//!      signalled by a nonzero exit, not a silent success).
//!
//! Parity of the alignment numbers themselves is covered by the library tests
//! and the USAlign oracle; here we only assert the CLI plumbing.

use std::path::PathBuf;
use std::process::Command;

fn pdb(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../test-pdbs")
        .join(name)
}

fn run(bin: &str, args: &[&str]) -> (String, String, i32) {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin}: {e}"));
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

/// The first non-comment (`#`) line split on tabs — the `--outfmt 2` data row.
fn outfmt2_row(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .find(|l| !l.trim_start().starts_with('#') && l.contains('\t'))
        .unwrap_or_else(|| panic!("no tabular data row in output:\n{stdout}"))
        .split('\t')
        .collect()
}

// ---------------------------------------------------------------------------
// tmalign
// ---------------------------------------------------------------------------

#[test]
fn tmalign_self_alignment_outfmt2_is_perfect() {
    let bin = env!("CARGO_BIN_EXE_tmalign");
    let p = pdb("1crn.pdb");
    let ps = p.to_str().unwrap();
    let (stdout, stderr, code) = run(bin, &[ps, ps, "--outfmt", "2"]);
    assert_eq!(code, 0, "self-alignment should succeed: {stderr}");

    // Header is present and names the 11-column contract.
    assert!(
        stdout.contains("#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD"),
        "outfmt 2 must print the tabular header:\n{stdout}"
    );
    let row = outfmt2_row(&stdout);
    assert_eq!(row.len(), 11, "outfmt 2 row must have 11 columns: {row:?}");
    // TM1, TM2 (cols 2,3) == 1.0, RMSD (col 4) == 0 for a self-alignment.
    assert_eq!(row[2], "1.0000", "TM1 of self-alignment must be 1.0");
    assert_eq!(row[3], "1.0000", "TM2 of self-alignment must be 1.0");
    assert_eq!(row[4], "0.00", "RMSD of self-alignment must be 0");
    // L1 == L2 == Lali == 46 residues for 1crn.
    assert_eq!(row[8], "46");
    assert_eq!(row[9], "46");
    assert_eq!(row[10], "46");
}

#[test]
fn tmalign_unrelated_pair_outfmt2_parses() {
    let bin = env!("CARGO_BIN_EXE_tmalign");
    let a = pdb("1crn.pdb");
    let b = pdb("1ubq.pdb");
    let (stdout, stderr, code) = run(
        bin,
        &[a.to_str().unwrap(), b.to_str().unwrap(), "--outfmt", "2"],
    );
    assert_eq!(
        code, 0,
        "alignment of two real structures should succeed: {stderr}"
    );
    let row = outfmt2_row(&stdout);
    assert_eq!(row.len(), 11);
    // TM-scores are valid probabilities in (0, 1]; unrelated folds score low
    // but the exact value is the library's business, not the CLI's.
    let tm1: f64 = row[2].parse().expect("TM1 should parse");
    assert!(tm1 > 0.0 && tm1 <= 1.0, "TM1 out of range: {tm1}");
}

#[test]
fn tmalign_default_format_is_human_readable() {
    let bin = env!("CARGO_BIN_EXE_tmalign");
    let p = pdb("1crn.pdb");
    let ps = p.to_str().unwrap();
    let (stdout, _stderr, code) = run(bin, &[ps, ps]);
    assert_eq!(code, 0);
    // The default (outfmt 0) block is the human-readable report.
    assert!(
        stdout.contains("TM-score"),
        "default output should report TM-score"
    );
    assert!(stdout.contains("RMSD"), "default output should report RMSD");
}

#[test]
fn tmalign_missing_second_file_fails_cleanly() {
    let bin = env!("CARGO_BIN_EXE_tmalign");
    let p = pdb("1crn.pdb");
    let (_stdout, stderr, code) = run(bin, &[p.to_str().unwrap()]);
    assert_ne!(code, 0, "a single file argument must fail");
    assert!(
        stderr.to_lowercase().contains("two structure") || stderr.contains("--help"),
        "error should explain two structures are required:\n{stderr}"
    );
}

#[test]
fn tmalign_nonexistent_file_fails() {
    let bin = env!("CARGO_BIN_EXE_tmalign");
    let p = pdb("1crn.pdb");
    let (_stdout, stderr, code) = run(bin, &[p.to_str().unwrap(), "/no/such/structure.pdb"]);
    assert_ne!(code, 0, "a missing input file must fail");
    assert!(!stderr.is_empty(), "a failure must say something on stderr");
}

// ---------------------------------------------------------------------------
// usalign
// ---------------------------------------------------------------------------

#[test]
fn usalign_self_alignment_outfmt2_is_perfect() {
    let bin = env!("CARGO_BIN_EXE_usalign");
    let p = pdb("1crn.pdb");
    let ps = p.to_str().unwrap();
    let (stdout, stderr, code) = run(bin, &[ps, ps, "--outfmt", "2"]);
    assert_eq!(code, 0, "self-alignment should succeed: {stderr}");
    let row = outfmt2_row(&stdout);
    assert_eq!(row.len(), 11, "outfmt 2 row must have 11 columns: {row:?}");
    assert_eq!(row[2], "1.0000", "TM1 of self-alignment must be 1.0");
    assert_eq!(row[3], "1.0000", "TM2 of self-alignment must be 1.0");
    assert_eq!(row[4], "0.00", "RMSD of self-alignment must be 0");
}

#[test]
fn usalign_missing_second_file_fails_cleanly() {
    let bin = env!("CARGO_BIN_EXE_usalign");
    let p = pdb("1crn.pdb");
    let (_stdout, stderr, code) = run(bin, &[p.to_str().unwrap()]);
    assert_ne!(code, 0, "a single file argument must fail");
    assert!(!stderr.is_empty(), "a failure must say something on stderr");
}

// ---------------------------------------------------------------------------
// ingest
// ---------------------------------------------------------------------------

/// Read a Parquet file's metadata: (num_rows, column_names).
fn parquet_summary(path: &std::path::Path) -> (i64, Vec<String>) {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let file = std::fs::File::open(path).expect("open parquet");
    let reader = SerializedFileReader::new(file).expect("parse parquet");
    let meta = reader.metadata().file_metadata();
    let cols = meta
        .schema_descr()
        .columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    (meta.num_rows(), cols)
}

#[test]
fn ingest_single_file_writes_valid_parquet() {
    let bin = env!("CARGO_BIN_EXE_ingest");
    let out = std::env::temp_dir().join("proteon_cli_ingest_single.parquet");
    let _ = std::fs::remove_file(&out);

    let (_stdout, stderr, code) = run(
        bin,
        &[
            pdb("1crn.pdb").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "ingest should succeed: {stderr}");
    assert!(
        stderr.contains("0 failed"),
        "no failures expected:\n{stderr}"
    );
    assert!(out.exists(), "parquet not written");

    let (rows, cols) = parquet_summary(&out);
    // 1crn has 327 atoms → one per-atom row each.
    assert_eq!(rows, 327, "1crn should yield 327 atom rows");
    assert!(
        cols.iter().any(|c| c == "structure_id") && cols.iter().any(|c| c == "atom_name"),
        "per-atom schema missing expected columns: {cols:?}"
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
fn ingest_multiple_inputs_concatenate_into_one_parquet() {
    let bin = env!("CARGO_BIN_EXE_ingest");
    let out = std::env::temp_dir().join("proteon_cli_ingest_multi.parquet");
    let _ = std::fs::remove_file(&out);

    let (_stdout, stderr, code) = run(
        bin,
        &[
            pdb("1crn.pdb").to_str().unwrap(),
            pdb("1aaj.pdb").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "ingest of two files should succeed: {stderr}");
    let (rows, _cols) = parquet_summary(&out);
    // Two structures concatenate, so strictly more rows than 1crn alone (327).
    assert!(
        rows > 327,
        "two structures should exceed one structure's rows: {rows}"
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
fn ingest_per_structure_writes_one_file_each() {
    let bin = env!("CARGO_BIN_EXE_ingest");
    let dir = std::env::temp_dir().join("proteon_cli_ingest_perstruct");
    let _ = std::fs::remove_dir_all(&dir);

    let (_stdout, stderr, code) = run(
        bin,
        &[
            pdb("1crn.pdb").to_str().unwrap(),
            pdb("1aaj.pdb").to_str().unwrap(),
            "--out",
            dir.to_str().unwrap(),
            "--per-structure",
        ],
    );
    assert_eq!(code, 0, "per-structure ingest should succeed: {stderr}");
    assert!(dir.join("1crn.parquet").exists(), "missing 1crn.parquet");
    assert!(dir.join("1aaj.parquet").exists(), "missing 1aaj.parquet");
    // Each per-structure file is independently valid.
    let (rows, _) = parquet_summary(&dir.join("1crn.parquet"));
    assert_eq!(rows, 327, "per-structure 1crn should still be 327 rows");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ingest_isolates_a_bad_file_but_still_writes_the_good_one() {
    let bin = env!("CARGO_BIN_EXE_ingest");
    let bad = std::env::temp_dir().join("proteon_cli_ingest_bad.pdb");
    std::fs::write(&bad, "this is not a pdb\n").unwrap();
    let out = std::env::temp_dir().join("proteon_cli_ingest_partial.parquet");
    let _ = std::fs::remove_file(&out);

    let (_stdout, stderr, code) = run(
        bin,
        &[
            pdb("1crn.pdb").to_str().unwrap(),
            bad.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ],
    );
    // Partial success: the good structure is ingested, the bad one is reported,
    // and the run still exits 0 (the bulk pipeline tolerates a few failures).
    assert_eq!(code, 0, "partial success should exit 0: {stderr}");
    assert!(
        stderr.contains("1 failed"),
        "the bad file must be counted:\n{stderr}"
    );
    assert!(
        stderr.contains("SKIP"),
        "the bad file must be named:\n{stderr}"
    );
    let (rows, _) = parquet_summary(&out);
    assert_eq!(rows, 327, "the good structure must still be written");

    let _ = std::fs::remove_file(&bad);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn ingest_all_inputs_failed_exits_nonzero() {
    let bin = env!("CARGO_BIN_EXE_ingest");
    let bad = std::env::temp_dir().join("proteon_cli_ingest_allbad.pdb");
    std::fs::write(&bad, "this is not a pdb\n").unwrap();
    let out = std::env::temp_dir().join("proteon_cli_ingest_allbad.parquet");
    let _ = std::fs::remove_file(&out);

    let (_stdout, stderr, code) = run(
        bin,
        &[bad.to_str().unwrap(), "--out", out.to_str().unwrap()],
    );
    // Producing NOTHING is an error, not a silent success — a pipeline gating
    // on the exit code must not mistake an empty ingest for a complete one.
    assert_ne!(code, 0, "an all-failed ingest must exit nonzero:\n{stderr}");
    assert!(
        stderr.contains("no structures ingested"),
        "error should explain nothing was ingested:\n{stderr}"
    );

    let _ = std::fs::remove_file(&bad);
    let _ = std::fs::remove_file(&out);
}
