#!/usr/bin/env bash
# Local mirror of the gates that the Tests workflow on `main` enforces.
#
# `main` is gated by ruleset 15659866: every direct push has to wait for
# all 8 Tests jobs to go green (~17 min) before it's accepted. This
# script lets you find the cheap failures locally first, so the only
# remote wait is the one you actually need.
#
# Usage:
#   scripts/preflight.sh           # fast (~30 s):  catches lint, fmt, build,
#                                                    fast-tier pytest
#   scripts/preflight.sh --full    # full  (~15 min): mirrors what CI runs
#
# Exit 0 if every step passed; the script aborts at the first failure.
#
# Known wart: if `cargo build` fails with `failed to run the Python interpreter
# at /scratch/TMAlign/ferritin/.venv/...`, the pyo3 build cache is carrying a
# stale Python path from before the ferritin→proteon rename. Run `cargo clean`
# once and re-run; this should not recur.

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="fast"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,16p' "$0"
  exit 0
fi

step() {
  local name=$1
  shift
  echo
  echo "▶ $name"
  if "$@"; then
    echo "✓ $name"
  else
    echo "✗ $name failed — aborting before any more checks."
    exit 1
  fi
}

# Tier 1: cheap checks that catch the bulk of CI failures.
step "Version Sync"   python3 tools/check_versions.py
step "cargo fmt"      cargo fmt --all -- --check
step "cargo clippy"   cargo clippy --workspace --all-targets -- -D warnings

# Activate the project venv if it exists so pytest sees `proteon`.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

if [[ "$MODE" == "fast" ]]; then
  step "cargo build"  cargo build --workspace --quiet
  step "pytest (fast tier: not slow, not oracle)" \
    pytest -q -m "not slow and not oracle"
  echo
  echo "All fast preflight checks passed."
  echo "Run scripts/preflight.sh --full before opening a PR for the full mirror."
else
  step "cargo test"   cargo test --workspace
  step "pytest (full)" pytest -q
  echo
  echo "All full preflight checks passed — should mirror CI."
fi
