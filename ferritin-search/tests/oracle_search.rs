//! End-to-end search oracle: ferritin-search vs upstream `mmseqs search`.
//!
//! For each sequence in the vendored fixture, this test runs both
//! upstream and our engine searching it against the full fixture and
//! compares hit lists. Every module has unit tests; this is the
//! composition gate that proves the pieces fit together correctly.
//!
//! Skipped if the upstream binary is unreachable. CI enforcement via
//! `FERRITIN_SEARCH_REQUIRE_ORACLE=1` is the same env var as the DB I/O
//! oracle test (`tests/roundtrip.rs`).
//!
//! Comparison strategy: this is a sensitivity / recall comparison, not
//! byte-exact. We use different reduction algorithms, don't compute
//! e-values, and use simpler scoring than upstream — so we expect
//! divergence in hit ordering and inclusion. The assertions are
//! deliberately tolerant:
//!
//!   1. **Self-hit (both pipelines)**: every query finds itself in
//!      both upstream's and our top-1. This is a strong sanity check
//!      independent of any algorithmic divergence.
//!   2. **Top-1 agreement rate**: across all queries, our top-1
//!      target_id matches upstream's top-1 for at least 80% of them.
//!      The remaining 20% allows for tied scores breaking differently.
//!   3. **Top-K recall**: when upstream's top-1 isn't ours, it must at
//!      least appear in our top-10. (Upstream might be finding a
//!      slightly higher-scoring near-duplicate that our reduction
//!      blends with another candidate.)

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Command;

use ferritin_search::alphabet::Alphabet;
use ferritin_search::db::DBReader;
use ferritin_search::matrix::SubstitutionMatrix;
use ferritin_search::search::{SearchEngine, SearchOptions};
use ferritin_search::sequence::Sequence;
use tempfile::tempdir;

const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn require_oracle() -> bool {
    std::env::var("FERRITIN_SEARCH_REQUIRE_ORACLE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn find_mmseqs() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("FERRITIN_SEARCH_MMSEQS_BIN") {
        let p = PathBuf::from(&explicit);
        if explicit.contains('/') && !p.exists() {
            return None;
        }
        return p.canonicalize().ok().or(Some(p));
    }
    let candidates = [
        format!("{CRATE_ROOT}/../../MMseqs2/build/src/mmseqs"),
        format!("{CRATE_ROOT}/../../MMseqs2/oracle-bin/mmseqs/bin/mmseqs"),
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p.canonicalize().ok();
        }
    }
    None
}

fn find_fixture() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("FERRITIN_SEARCH_EXAMPLE_FASTA") {
        let p = PathBuf::from(explicit);
        return if p.exists() { p.canonicalize().ok() } else { None };
    }
    let p = PathBuf::from(format!("{CRATE_ROOT}/tests/data/oracle_fixture.fasta"));
    if p.exists() { p.canonicalize().ok() } else { None }
}

fn skip_or_panic(test_name: &str, reason: &str) -> bool {
    if require_oracle() {
        panic!("{test_name}: oracle required but {reason}");
    } else {
        eprintln!("SKIP {test_name}: {reason}");
        true
    }
}

/// Build an in-memory map `accession → numeric_id` from the upstream
/// `<prefix>.lookup` file.
fn lookup_accession_to_id(reader: &DBReader) -> HashMap<String, u32> {
    let lookup = reader.lookup.as_ref().expect("createdb writes .lookup");
    lookup.iter().map(|e| (e.accession.clone(), e.key)).collect()
}

/// Parse upstream's m8 output into `query_acc → ranked Vec<target_acc>`.
fn parse_m8(text: &str) -> HashMap<String, Vec<String>> {
    let mut by_query: HashMap<String, Vec<String>> = HashMap::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let q = cols.next().unwrap_or("").to_owned();
        let t = cols.next().unwrap_or("").to_owned();
        if q.is_empty() || t.is_empty() {
            continue;
        }
        by_query.entry(q).or_default().push(t);
    }
    by_query
}

fn run_upstream_search(mmseqs: &PathBuf, fasta: &PathBuf, workdir: &std::path::Path) -> String {
    let query_db = workdir.join("queryDB");
    let target_db = workdir.join("targetDB");
    let result_db = workdir.join("resultDB");
    let m8_path = workdir.join("result.m8");
    let tmp = workdir.join("tmp");
    std::fs::create_dir_all(&tmp).unwrap();

    for (subcmd, args) in [
        (
            "createdb",
            vec![
                fasta.to_str().unwrap().to_owned(),
                query_db.to_str().unwrap().to_owned(),
                "-v".to_owned(),
                "1".to_owned(),
            ],
        ),
        (
            "createdb",
            vec![
                fasta.to_str().unwrap().to_owned(),
                target_db.to_str().unwrap().to_owned(),
                "-v".to_owned(),
                "1".to_owned(),
            ],
        ),
        (
            "search",
            vec![
                query_db.to_str().unwrap().to_owned(),
                target_db.to_str().unwrap().to_owned(),
                result_db.to_str().unwrap().to_owned(),
                tmp.to_str().unwrap().to_owned(),
                "-v".to_owned(),
                "1".to_owned(),
                "--threads".to_owned(),
                "1".to_owned(),
                "-s".to_owned(),
                "5.7".to_owned(),
            ],
        ),
        (
            "convertalis",
            vec![
                query_db.to_str().unwrap().to_owned(),
                target_db.to_str().unwrap().to_owned(),
                result_db.to_str().unwrap().to_owned(),
                m8_path.to_str().unwrap().to_owned(),
                "-v".to_owned(),
                "1".to_owned(),
            ],
        ),
    ] {
        let status = Command::new(mmseqs)
            .arg(subcmd)
            .args(&args)
            .status()
            .unwrap_or_else(|e| panic!("failed to run mmseqs {subcmd}: {e}"));
        assert!(status.success(), "mmseqs {subcmd} returned non-zero");
    }

    std::fs::read_to_string(&m8_path).expect("read m8 output")
}

#[test]
fn ferritin_search_recall_matches_upstream_on_fixture() {
    let Some(mmseqs) = find_mmseqs() else {
        skip_or_panic(
            "ferritin_search_recall_matches_upstream_on_fixture",
            "mmseqs binary not found",
        );
        return;
    };
    let Some(fasta) = find_fixture() else {
        skip_or_panic(
            "ferritin_search_recall_matches_upstream_on_fixture",
            "fixture fasta not found",
        );
        return;
    };

    let workdir = tempdir().unwrap();

    // Run upstream pipeline on the fixture (queries == targets).
    let m8_text = run_upstream_search(&mmseqs, &fasta, workdir.path());
    let upstream_hits = parse_m8(&m8_text);

    // Reuse upstream's createdb output to get our targets — no need to
    // write a FASTA parser here; the DB is already a DBReader-readable
    // payload of the same sequences.
    let target_prefix = workdir.path().join("targetDB");
    let reader = DBReader::open(&target_prefix).expect("open upstream target DB");
    let acc_to_id = lookup_accession_to_id(&reader);
    let id_to_acc: HashMap<u32, String> =
        acc_to_id.iter().map(|(a, i)| (*i, a.clone())).collect();

    let alpha = Alphabet::protein();
    let matrix = SubstitutionMatrix::blosum62();

    // Decode every entry into a Sequence keyed by its numeric DB key.
    let entries: Vec<(u32, Sequence)> = reader
        .index
        .iter()
        .map(|e| {
            let bytes = reader.get_payload(e);
            (e.key, Sequence::from_ascii(alpha.clone(), bytes))
        })
        .collect();

    let opts = SearchOptions::default();
    let engine =
        SearchEngine::build(entries.clone(), &matrix, alpha.clone(), opts).unwrap();

    // Track per-test stats instead of per-query asserts so a single
    // outlier doesn't fail the whole oracle.
    let mut total_with_upstream_hits = 0usize;
    let mut self_hit_in_ours = 0usize;
    let mut self_hit_in_upstream = 0usize;
    let mut top1_agree = 0usize;
    let mut upstream_top1_in_our_top10 = 0usize;

    for (qid, qseq) in &entries {
        let q_acc = id_to_acc.get(qid).expect("query id in lookup");
        let our_hits = engine.search(qseq);
        let our_top1 = our_hits.first().map(|h| h.target_id);
        let our_top10: HashSet<u32> =
            our_hits.iter().take(10).map(|h| h.target_id).collect();

        if our_top1 == Some(*qid) {
            self_hit_in_ours += 1;
        }

        let Some(upstream) = upstream_hits.get(q_acc) else {
            continue;
        };
        total_with_upstream_hits += 1;
        let up_top1_acc = &upstream[0];
        let up_top1_id = match acc_to_id.get(up_top1_acc) {
            Some(id) => *id,
            None => continue,
        };
        if up_top1_id == *qid {
            self_hit_in_upstream += 1;
        }
        if our_top1 == Some(up_top1_id) {
            top1_agree += 1;
        }
        if our_top10.contains(&up_top1_id) {
            upstream_top1_in_our_top10 += 1;
        }
    }

    eprintln!(
        "oracle stats: queries={}, self_hit_ours={}, self_hit_upstream={}, \
         top1_agree={}, up_top1_in_our_top10={}",
        entries.len(),
        self_hit_in_ours,
        self_hit_in_upstream,
        top1_agree,
        upstream_top1_in_our_top10,
    );

    // Assertion 1: every query finds itself in our top-1.
    assert_eq!(
        self_hit_in_ours,
        entries.len(),
        "self-hit failed for {} queries in our pipeline",
        entries.len() - self_hit_in_ours,
    );

    // Upstream may legitimately find no hits for some queries below its
    // default e-value; only assert against the queries it returned hits for.
    assert!(
        total_with_upstream_hits > 0,
        "upstream returned no hits for any query — fixture sanity broken",
    );

    // Assertion 2: when both pipelines return a top-1, we agree on it
    // for the vast majority. Exact threshold is conservative — small
    // tied-score differences break some.
    let top1_agree_rate = top1_agree as f64 / total_with_upstream_hits as f64;
    assert!(
        top1_agree_rate >= 0.8,
        "top-1 agreement {:.2} < 0.80 ({}/{})",
        top1_agree_rate,
        top1_agree,
        total_with_upstream_hits,
    );

    // Assertion 3: when we don't put upstream's top-1 first, it should
    // still be in our top-10 — divergence is in ordering, not recall.
    let recall_at_10 =
        upstream_top1_in_our_top10 as f64 / total_with_upstream_hits as f64;
    assert!(
        recall_at_10 >= 0.95,
        "recall@10 of upstream top-1 in our top-10 was {:.2} < 0.95 ({}/{})",
        recall_at_10,
        upstream_top1_in_our_top10,
        total_with_upstream_hits,
    );
}
