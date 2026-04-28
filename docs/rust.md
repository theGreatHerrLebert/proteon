# Rust API

The Rust workspace crates are documented via `cargo doc`, hosted on this
same site under `/rust/`. The links in the table below jump straight into
the rustdoc tree.

We **self-host** rustdoc rather than relying on docs.rs because proteon
currently depends on `pdbtbx` via a git reference (the published 0.12.0 has
bugs blocking us). docs.rs refuses git deps, so until pdbtbx cuts a
post-fix release, rustdoc lives here.

## Crates

| Crate | Entry point |
|-------|-------------|
| `proteon-align` | <a href="../rust/proteon_align/index.html">proteon_align</a> |
| `proteon-io` | <a href="../rust/proteon_io/index.html">proteon_io</a> |
| `proteon-arrow` | <a href="../rust/proteon_arrow/index.html">proteon_arrow</a> |
| `proteon-search` | <a href="../rust/proteon_search/index.html">proteon_search</a> |
| `proteon-bin` (CLIs) | <a href="../rust/tmalign/index.html">tmalign</a>, <a href="../rust/usalign/index.html">usalign</a>, <a href="../rust/ingest/index.html">ingest</a>, <a href="../rust/build_kmi/index.html">build_kmi</a>, <a href="../rust/fasta_to_mmseqs_db/index.html">fasta_to_mmseqs_db</a> |

`proteon-connector` is a PyO3 `cdylib` and is intentionally **not** rustdoc'd
here — `cargo doc` doesn't build it cleanly without a libpython link. Its
public surface is documented at the Python-wrapper level on this site (see
[Subsystems](subsystems/index.md)).

## Building locally

```bash
cd proteon
cargo doc --workspace --no-deps --document-private-items=false
# Output: target/doc/
```

To preview alongside the mkdocs site:

```bash
mkdir -p docs/rust
rsync -a --delete target/doc/ docs/rust/
mkdocs serve
```

The CI workflow does the equivalent on every push to `main` and publishes
the combined site to GitHub Pages.

## Search

Rustdoc has its own search (`s` to focus). It is **not** unified with the
mkdocs search box at the top of these pages — the mkdocs search covers prose
and the Python API reference; the rustdoc search covers the Rust crates.
This is a deliberate tradeoff vs Sphinx; see
[architecture](architecture.md#what-gets-documented-where).
