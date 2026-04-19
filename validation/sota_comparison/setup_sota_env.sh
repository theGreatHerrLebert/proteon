#!/bin/bash
# Create the sota_venv and install proteon + v1 SOTA tools.
#
# Usage:
#   bash validation/sota_comparison/setup_sota_env.sh
#
# Environment variables (override to customize):
#   SOTA_VENV      — path to the new venv (default: sibling of the build venv)
#   PROTEON_ROOT  — proteon repo root (default: auto-detect from script location)
#   SKIP_PROTEON  — if set, don't reinstall proteon (faster repeat runs)
#
# Idempotent: safe to re-run. Will skip venv creation if SOTA_VENV already exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTEON_ROOT="${PROTEON_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SOTA_VENV="${SOTA_VENV:-/globalscratch/dateschn/proteon-benchmark/sota_venv}"

echo "[sota setup] proteon root: $PROTEON_ROOT"
echo "[sota setup] sota venv:     $SOTA_VENV"

# --- 1. Create the venv if it doesn't already exist ---
if [ ! -d "$SOTA_VENV" ]; then
    echo "[sota setup] creating venv..."
    python3 -m venv "$SOTA_VENV"
else
    echo "[sota setup] venv already exists, reusing"
fi

# shellcheck source=/dev/null
source "$SOTA_VENV/bin/activate"

# --- 2. Upgrade pip/wheel, install maturin ---
echo "[sota setup] upgrading pip + installing build tools..."
pip install --quiet --upgrade pip wheel
pip install --quiet "maturin>=1.5,<2"

# --- 3. Install proteon via maturin develop ---
if [ -z "${SKIP_PROTEON:-}" ]; then
    echo "[sota setup] building proteon-connector via maturin develop --release..."
    (cd "$PROTEON_ROOT/proteon-connector" && maturin develop --release)
    echo "[sota setup] installing proteon Python wrapper (editable)..."
    pip install --quiet -e "$PROTEON_ROOT/packages/proteon"
else
    echo "[sota setup] SKIP_PROTEON set, skipping proteon reinstall"
fi

# --- 4. Install SOTA requirements ---
echo "[sota setup] installing SOTA requirements..."
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# --- 5. Sanity check imports ---
echo "[sota setup] sanity check imports..."
python - <<'PY'
import sys
failures = []
for name in ("proteon", "freesasa", "openmm", "numpy"):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  ok: {name} {ver}")
    except ImportError as e:
        failures.append(f"  FAIL: {name}: {e}")
if failures:
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
PY

echo "[sota setup] done. Activate with: source $SOTA_VENV/bin/activate"
