# crates.io publish status

Per-crate readiness from `cargo publish --dry-run` (audit run 2026-04-28
on 0.1.2). Re-run via `cargo publish --dry-run -p <crate> --allow-dirty`.

| Crate | Publishable today? | Blocker |
|---|---|---|
| `proteon-align` | After one small fix | `src/search/alphabet.rs` `include!`s `../validation/alphabet_vqvae_rust.txt`, which is stripped from the package tarball |
| `proteon-search` | **Yes — clean dry-run** | none |
| `proteon-io` | No | `pdbtbx` is a `git = …` dependency without a `version = …`; crates.io rejects |
| `proteon-arrow` | No | same `pdbtbx` git-only dep |
| `proteon-bin` | No | same `pdbtbx` git-only dep |
| `proteon-connector` | No | same `pdbtbx` git-only dep |

## Workspace metadata

`Cargo.toml` `[workspace.package]` already provides `version`, `edition`,
`rust-version`, `authors`, `license`, `repository`, `homepage`, `readme`,
`keywords`, `categories`. Each crate inherits the right pieces and adds
its own `description`. No metadata gaps.

## The pdbtbx blocker

Four of the six crates depend on the patched `pdbtbx` fork
(`git = "github:theGreatHerrLebert/pdbtbx" rev = "c82e8c08…"`). crates.io
won't accept a manifest whose dep has no version requirement, so until
either:

- the patched `pdbtbx` lands on crates.io as a published version, **or**
- proteon vendors / forks pdbtbx in-tree as `proteon-pdbtbx`,

the four downstream crates stay unpublishable. This matches the long-
standing note in CLAUDE.md.

## The `proteon-align` reach-out

```rust
// proteon-align/src/search/alphabet.rs:52
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../validation/alphabet_vqvae_rust.txt"
));
```

`validation/` lives outside the crate, so `cargo package` strips it.
Two ways to fix:

1. **Copy the file into the crate** (e.g.
   `proteon-align/src/search/alphabet_vqvae_rust.txt`) and update the
   `include!` path. `validation/train_vqvae.py` accepts an `--output`
   argument, so re-training writes to the crate path going forward —
   no tooling change needed beyond a one-line invocation note.
2. Generate at build time via a `build.rs` that copies `OUT_DIR`-side.
   Heavier; not warranted for a frozen artifact.

(1) is the right call when the user wants to unblock `proteon-align` for
publish. ~3 lines of change.

## Order to publish in (when ready)

`proteon-align` → `proteon-search` → (pdbtbx unblock) → `proteon-io`
→ `proteon-arrow` → `proteon-bin` → `proteon-connector`. Path
dependencies between workspace crates need to be resolved bottom-up.
