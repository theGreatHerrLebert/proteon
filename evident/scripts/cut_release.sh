#!/usr/bin/env bash
# Cut a release: fetch artifacts from monster3, lock the bundle, print git steps.
#
# Heavy-compute artifacts (50K oracle JSONL, OBC GB CPU/GPU runs, etc.)
# are produced on user-controlled hardware (monster3 by default) and live
# at gitignored paths in the local working tree. lock_release_replays.py
# expects them to be present at the path the claim YAML names. This
# script fetches them once, then runs the lock + index regen.
#
# Per the EVIDENT-vs-VIBES rule, the heavy compute does NOT run here —
# this script only mirrors completed artifacts and freezes the bundle.
# If a referenced artifact is not on monster3 yet, the corresponding
# claim row will land as `status: missing` in the release manifest,
# which is the correct behaviour (visible coverage gap, not a silent
# fail).
#
# Usage:
#   ./evident/scripts/cut_release.sh v0.1.0
#   ./evident/scripts/cut_release.sh v0.1.0 --no-fetch     # skip ssh, lock with whatever is already local
#   ./evident/scripts/cut_release.sh v0.1.0 --host other   # use SSH host alias `other` instead of monster3
#
# Edits to the artifact list below are how new claims plug in: append a
# (remote-path, local-path) pair to the ARTIFACTS array.

set -euo pipefail

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<USAGE
Usage: $(basename "$0") <release-tag> [--no-fetch] [--host HOST]

  release-tag    e.g. v0.1.0, v0.2.0 — names the bundle directory
                 evident/reports/<release-tag>/

Options:
  --no-fetch     Skip the scp step. Use whatever artifacts are already
                 in the local working tree.
  --host HOST    SSH host alias (default: monster3).

After this script succeeds:
  git add evident/reports/<release-tag>/
  git commit -m "release: <release-tag> EVIDENT bundle"
  git tag <release-tag>
  git push origin main <release-tag>
USAGE
    exit 0
fi

RELEASE="$1"
shift || true

HOST="monster3"
FETCH=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-fetch) FETCH=0; shift ;;
        --host)     HOST="$2"; shift 2 ;;
        *)          echo "unknown arg: $1" >&2; exit 64 ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Heavy-compute artifacts: (remote-path-on-$HOST, local-path-in-repo).
# Each pair maps a claim's evidence.artifact to where the JSONL lives
# on the host that ran it. Add new entries here when a new release-tier
# claim ships an artifact that has to be mirrored.
declare -a ARTIFACTS=(
    "/globalscratch/dateschn/proteon-benchmark/proteon/validation/charmm19_eef1_ball_oracle_50k.jsonl validation/charmm19_eef1_ball_oracle_50k.jsonl"
    "/globalscratch/dateschn/proteon-benchmark/v020_amber96_sp_50k/amber96_oracle.jsonl validation/amber96_oracle_50k.jsonl"
    "/globalscratch/dateschn/proteon-benchmark/v020_dssp_50k/dssp_mkdssp_oracle.jsonl validation/dssp_mkdssp_oracle_50k.jsonl"
    "/globalscratch/dateschn/proteon-benchmark/v020_fold_50k_charmm/charmm_pair_1k.jsonl validation/fold_preservation_50k/charmm_pair_1k.jsonl"
)

if (( FETCH )); then
    echo "# fetching artifacts from $HOST"
    for pair in "${ARTIFACTS[@]}"; do
        remote="${pair%% *}"
        local="${pair##* }"
        mkdir -p "$(dirname "$local")"
        echo "  $HOST:$remote -> $local"
        scp -q "${HOST}:${remote}" "$local"
    done
    echo
fi

echo "# locking release bundle for $RELEASE"
python3 evident/scripts/lock_release_replays.py --release "$RELEASE" --clean

echo
echo "# next steps:"
echo "  git add evident/reports/${RELEASE}/"
echo "  git commit -m \"release: ${RELEASE} EVIDENT bundle\""
echo "  git tag ${RELEASE}"
echo "  git push origin main ${RELEASE}"
