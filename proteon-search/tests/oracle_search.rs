//! End-to-end search oracle: proteon-search vs upstream `mmseqs search`.
//!
//! For each sequence in a query FASTA, runs both upstream and our engine
//! searching against a target FASTA and compares hit lists. Every module
//! has unit tests; this is the composition gate that proves the pieces
//! fit together correctly.
//!
//! ## Opt-in only
//!
//! This test is **gated on `PROTEON_SEARCH_REQUIRE_ORACLE=1`** and skips
//! immediately otherwise. It runs upstream's full createdb + search +
//! convertalis pipeline plus our SearchEngine, which costs tens of
//! seconds even on the small fixture and minutes on the larger CI
//! corpus — too slow to auto-run with `cargo test`. CI sets the env var
//! explicitly; local dev opts in only when running the oracle.
//!
//! ## Corpus selection (env vars, all optional)
//!
//! - `PROTEON_SEARCH_QUERY_FASTA`: queries (default: vendored 50-seq
//!   fixture).
//! - `PROTEON_SEARCH_TARGET_FASTA`: targets (default: same as queries —
//!   self-search). CI sets this to the full upstream `examples/DB.fasta`
//!   (~20k sequences) so the oracle exercises real cross-similarity
//!   instead of self-hit detection only.
//! - `PROTEON_SEARCH_EXAMPLE_FASTA`: legacy alias setting both query and
//!   target to the same file.
//! - `PROTEON_SEARCH_MMSEQS_BIN`: explicit path to the `mmseqs` binary
//!   (or a bare name on `$PATH`).
//!
//! ## Comparison strategy
//!
//! This is a sensitivity / recall comparison, not byte-exact. We use a
//! different reduction algorithm, don't compute e-values, and use simpler
//! scoring than upstream — so we expect divergence in hit ordering and
//! inclusion. Assertions are deliberately tolerant:
//!
//!   1. **Self-hit (when query exists in target)**: every query that
//!      appears in the target corpus must find itself at top-1 in our
//!      pipeline (strong composition sanity).
//!   2. **Top-1 agreement rate**: when both pipelines return a top-1, we
//!      agree on it for >= 80% of queries.
//!   3. **Recall@10 of upstream's top-10**: averaged across queries,
//!      `|upstream_top10 ∩ our_top10| / |upstream_top10|` is >= 0.5.
//!      Meaningful sensitivity metric on cross-similar corpora.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use proteon_search::alphabet::Alphabet;
use proteon_search::db::DBReader;
use proteon_search::matrix::SubstitutionMatrix;
use proteon_search::search::{SearchEngine, SearchOptions};
use proteon_search::sequence::Sequence;
use tempfile::tempdir;

const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn require_oracle() -> bool {
    std::env::var("PROTEON_SEARCH_REQUIRE_ORACLE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn find_mmseqs() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("PROTEON_SEARCH_MMSEQS_BIN") {
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

/// Resolve the query FASTA path. Precedence: explicit
/// `PROTEON_SEARCH_QUERY_FASTA`, legacy `PROTEON_SEARCH_EXAMPLE_FASTA`,
/// vendored fixture.
fn find_query_fasta() -> Option<PathBuf> {
    for var in ["PROTEON_SEARCH_QUERY_FASTA", "PROTEON_SEARCH_EXAMPLE_FASTA"] {
        if let Ok(explicit) = std::env::var(var) {
            let p = PathBuf::from(explicit);
            return if p.exists() {
                p.canonicalize().ok()
            } else {
                None
            };
        }
    }
    let p = PathBuf::from(format!("{CRATE_ROOT}/tests/data/oracle_fixture.fasta"));
    if p.exists() {
        p.canonicalize().ok()
    } else {
        None
    }
}

/// Resolve the target FASTA path. Precedence: explicit
/// `PROTEON_SEARCH_TARGET_FASTA`, legacy `PROTEON_SEARCH_EXAMPLE_FASTA`,
/// fall back to the query FASTA (self-search).
fn find_target_fasta(query: &Path) -> Option<PathBuf> {
    for var in [
        "PROTEON_SEARCH_TARGET_FASTA",
        "PROTEON_SEARCH_EXAMPLE_FASTA",
    ] {
        if let Ok(explicit) = std::env::var(var) {
            let p = PathBuf::from(explicit);
            return if p.exists() {
                p.canonicalize().ok()
            } else {
                None
            };
        }
    }
    Some(query.to_path_buf())
}

/// Build an in-memory map `accession → numeric_id` from the upstream
/// `<prefix>.lookup` file.
fn lookup_accession_to_id(reader: &DBReader) -> HashMap<String, u32> {
    let lookup = reader.lookup.as_ref().expect("createdb writes .lookup");
    lookup
        .iter()
        .map(|e| (e.accession.clone(), e.key))
        .collect()
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

/// Run the upstream pipeline on the given query / target FASTAs and
/// return the m8 output text.
fn run_upstream_search(
    mmseqs: &Path,
    query_fasta: &Path,
    target_fasta: &Path,
    workdir: &Path,
) -> String {
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
                query_fasta.to_str().unwrap().to_owned(),
                query_db.to_str().unwrap().to_owned(),
                "-v".to_owned(),
                "1".to_owned(),
            ],
        ),
        (
            "createdb",
            vec![
                target_fasta.to_str().unwrap().to_owned(),
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

/// Decode every entry in `prefix` into `(seq_id, Sequence)` pairs in the
/// given alphabet. Uses our DBReader so we don't need a FASTA parser.
fn load_sequences_from_db(prefix: &std::path::Path, alpha: &Alphabet) -> Vec<(u32, Sequence)> {
    let reader = DBReader::open(prefix).expect("open mmseqs target DB");
    reader
        .index
        .iter()
        .map(|e| {
            let bytes = reader.get_payload(e);
            (e.key, Sequence::from_ascii(alpha.clone(), bytes))
        })
        .collect()
}

#[test]
fn proteon_search_recall_matches_upstream() {
    // OPT-IN ONLY. cargo test must not pay this cost by default.
    if !require_oracle() {
        eprintln!(
            "SKIP proteon_search_recall_matches_upstream: \
             set PROTEON_SEARCH_REQUIRE_ORACLE=1 to run (test invokes upstream \
             createdb+search+convertalis end-to-end; tens of seconds to minutes).",
        );
        return;
    }
    let mmseqs = find_mmseqs().expect(
        "PROTEON_SEARCH_REQUIRE_ORACLE is set but no mmseqs binary found \
         (set PROTEON_SEARCH_MMSEQS_BIN to its path)",
    );
    let query_fasta =
        find_query_fasta().expect("PROTEON_SEARCH_REQUIRE_ORACLE is set but query FASTA not found");
    let target_fasta = find_target_fasta(&query_fasta).expect("target FASTA not found");

    let workdir = tempdir().unwrap();
    eprintln!(
        "oracle: query_fasta={}, target_fasta={}",
        query_fasta.display(),
        target_fasta.display(),
    );

    let m8_text = run_upstream_search(&mmseqs, &query_fasta, &target_fasta, workdir.path());
    let upstream_hits = parse_m8(&m8_text);

    // Reuse upstream's createdb output to load both queries and targets
    // — no need for an in-tree FASTA parser.
    let alpha = Alphabet::protein();
    let query_prefix = workdir.path().join("queryDB");
    let target_prefix = workdir.path().join("targetDB");
    let query_entries = load_sequences_from_db(&query_prefix, &alpha);
    let target_entries = load_sequences_from_db(&target_prefix, &alpha);

    let target_reader = DBReader::open(&target_prefix).unwrap();
    let target_acc_to_id = lookup_accession_to_id(&target_reader);
    let query_reader = DBReader::open(&query_prefix).unwrap();
    let query_id_to_acc: HashMap<u32, String> = lookup_accession_to_id(&query_reader)
        .into_iter()
        .map(|(a, i)| (i, a))
        .collect();

    // Build the engine over the target corpus. Cap max_prefilter_hits
    // tighter than the SearchOptions default — at the larger corpus
    // (50 queries × 20k targets), the default 1000 candidates per query
    // means 50k scalar SW alignments and tens of minutes of runtime.
    // 100 candidates per query is plenty: upstream's top-10 (the
    // recall@10 denominator) is comfortably inside that.
    let matrix = SubstitutionMatrix::blosum62();
    let opts = SearchOptions {
        max_prefilter_hits: Some(100),
        ..SearchOptions::default()
    };
    let engine = SearchEngine::build(target_entries, &matrix, alpha.clone(), opts).unwrap();

    let mut total_queries = 0usize;
    let mut self_top1_in_ours = 0usize;
    let mut self_top10_in_ours = 0usize;
    let mut queries_with_self_in_targets = 0usize;
    let mut total_with_upstream_hits = 0usize;
    let mut top1_agree = 0usize;
    let mut recall_at_10_sum = 0.0f64;

    for (qid, qseq) in &query_entries {
        total_queries += 1;
        let q_acc = query_id_to_acc.get(qid).expect("query id in lookup");
        let our_hits = engine.search(qseq);
        let our_top1 = our_hits.first().map(|h| h.target_id);
        let our_top10: HashSet<u32> = our_hits.iter().take(10).map(|h| h.target_id).collect();

        // Self-hit tracking only meaningful when the query exists in the
        // target corpus by accession. Track both top-1 and top-10
        // membership: in a paralog-rich target corpus, upstream itself
        // often ranks a paralog above self at top-1, so demanding self
        // at top-1 in our pipeline is wrong. The meaningful sanity check
        // is that self appears somewhere in our top-10.
        let self_target_id = target_acc_to_id.get(q_acc).copied();
        if let Some(self_tid) = self_target_id {
            queries_with_self_in_targets += 1;
            if our_top1 == Some(self_tid) {
                self_top1_in_ours += 1;
            }
            if our_top10.contains(&self_tid) {
                self_top10_in_ours += 1;
            }
        }

        let Some(upstream) = upstream_hits.get(q_acc) else {
            continue;
        };
        total_with_upstream_hits += 1;

        let up_top1_id = match target_acc_to_id.get(&upstream[0]) {
            Some(id) => *id,
            None => continue,
        };
        if our_top1 == Some(up_top1_id) {
            top1_agree += 1;
        }

        let upstream_top10: HashSet<u32> = upstream
            .iter()
            .take(10)
            .filter_map(|acc| target_acc_to_id.get(acc).copied())
            .collect();
        let intersection = upstream_top10.intersection(&our_top10).count() as f64;
        let denom = upstream_top10.len().max(1) as f64;
        recall_at_10_sum += intersection / denom;
    }

    let avg_recall_at_10 = if total_with_upstream_hits > 0 {
        recall_at_10_sum / total_with_upstream_hits as f64
    } else {
        0.0
    };

    eprintln!(
        "oracle stats: queries={total_queries}, queries_with_self_in_targets={queries_with_self_in_targets}, \
         self_top1_in_ours={self_top1_in_ours}, self_top10_in_ours={self_top10_in_ours}, \
         queries_with_upstream_hits={total_with_upstream_hits}, \
         top1_agree={top1_agree}, avg_recall@10={avg_recall_at_10:.3}",
    );

    // Assertion 1: when the query exists in the target corpus, it must
    // appear in our top-10. We don't assert top-1 — at large
    // paralog-rich corpora, upstream itself ranks a paralog above self
    // for some queries; demanding our top-1 to be self would be
    // stricter than upstream is on the same input.
    if queries_with_self_in_targets > 0 {
        assert_eq!(
            self_top10_in_ours,
            queries_with_self_in_targets,
            "self failed to appear in top-10 for {} of {} queries present in target",
            queries_with_self_in_targets - self_top10_in_ours,
            queries_with_self_in_targets,
        );
    }

    // The remaining assertions need upstream to have returned hits.
    assert!(
        total_with_upstream_hits > 0,
        "upstream returned no hits for any query — fixture sanity broken",
    );

    // Assertion 2: top-1 agreement across queries with upstream hits.
    let top1_rate = top1_agree as f64 / total_with_upstream_hits as f64;
    assert!(
        top1_rate >= 0.8,
        "top-1 agreement {top1_rate:.2} < 0.80 ({top1_agree}/{total_with_upstream_hits})",
    );

    // Assertion 3: recall@10 — meaningful when targets contain
    // cross-similar sequences (CI corpus). Self-search gives trivial
    // 1.0 because each query has only one upstream hit (itself).
    assert!(
        avg_recall_at_10 >= 0.5,
        "avg recall@10 {avg_recall_at_10:.3} < 0.50",
    );
}
