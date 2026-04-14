"""SHA-256 helpers for corpus artifact integrity.

Used by the export modules to hash `tensors.npz` payloads and record
the digest in the artifact's `manifest.json`. See roadmap Section 6
(Corpus Artifact Model) — every artifact should carry a `checksum`
so release validation can detect silent corruption and so release
diffs become comparable across runs.

Kept deliberately small: one hash function, one verifier, no global
state. The intent is reuse across every NPZ-writing export module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


# 1 MiB read buffer — large enough to amortize syscall overhead on the
# compressed tensor blobs (typically 10-500 MB) without spiking RSS.
_CHUNK_BYTES = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of the file at `path`, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise `ValueError` if `path`'s SHA-256 doesn't match `expected`.

    The exception carries both hashes so test failures / release
    validation reports can point at exactly which artifact drifted.
    """
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path}: "
            f"expected sha256={expected}, actual sha256={actual}"
        )
