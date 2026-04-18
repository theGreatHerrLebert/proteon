"""Tests for `ferritin._artifact_checksum`.

Small crypto-adjacent module that hashes corpus artifacts (`tensors.npz`
payloads) so release validation can detect silent corruption. A bug
here would poison every release manifest without loud symptoms — worth
direct coverage despite the module's size.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ferritin._artifact_checksum import _CHUNK_BYTES, sha256_file, verify_sha256


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestSha256File:
    """Direct hashing — confirm the streamed implementation matches
    hashlib's single-shot output byte-for-byte."""

    def test_empty_file(self, tmp_path: Path):
        p = _write(tmp_path, "empty.bin", b"")
        # Known SHA-256 of the empty input.
        assert sha256_file(p) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_short_payload_matches_hashlib(self, tmp_path: Path):
        payload = b"ferritin checksum smoke"
        p = _write(tmp_path, "short.bin", payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert sha256_file(p) == expected

    def test_exact_chunk_boundary(self, tmp_path: Path):
        # One full 1 MiB chunk — exercises the "chunk read then EOF
        # on the next read" branch in the streaming loop.
        payload = os.urandom(_CHUNK_BYTES)
        p = _write(tmp_path, "exact.bin", payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert sha256_file(p) == expected

    def test_multi_chunk_spans_boundary(self, tmp_path: Path):
        # 2.5 MiB — three reads, last one short. Catches any bug that
        # only surfaces when read sizes vary across iterations.
        payload = os.urandom(int(_CHUNK_BYTES * 2.5))
        p = _write(tmp_path, "multi.bin", payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert sha256_file(p) == expected

    def test_accepts_str_and_pathlib(self, tmp_path: Path):
        payload = b"same content"
        p = _write(tmp_path, "dual.bin", payload)
        assert sha256_file(str(p)) == sha256_file(p)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            sha256_file(tmp_path / "does_not_exist.bin")


class TestVerifySha256:
    """Round-trip and tamper detection for the verifier."""

    def test_matches_roundtrip(self, tmp_path: Path):
        p = _write(tmp_path, "ok.bin", b"hello release manifest")
        digest = sha256_file(p)
        # Must not raise.
        verify_sha256(p, digest)

    def test_mismatch_raises_value_error_with_both_hashes(self, tmp_path: Path):
        p = _write(tmp_path, "tampered.bin", b"content A")
        wrong = sha256_file(p)  # hash of A
        # Overwrite with different content — hash on disk no longer matches.
        p.write_bytes(b"content B")
        with pytest.raises(ValueError) as exc:
            verify_sha256(p, wrong)
        message = str(exc.value)
        # The exception is a diagnostic surface for release validation;
        # both hashes must show up so a failed CI run points at the
        # exact artifact that drifted.
        assert "expected sha256=" in message
        assert "actual sha256=" in message
        assert wrong in message

    def test_case_sensitive(self, tmp_path: Path):
        """hexdigest is lowercase; uppercase expected should mismatch."""
        p = _write(tmp_path, "case.bin", b"case sensitivity matters")
        digest = sha256_file(p)
        with pytest.raises(ValueError):
            verify_sha256(p, digest.upper())
