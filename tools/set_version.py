#!/usr/bin/env python3
"""Update all release-version call sites from the repo-root VERSION file."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"


def _read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def _validate_version(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(
            f"invalid version {version!r}; expected semantic version like 0.1.2"
        )


def _replace_one(text: str, pattern: str, replacement: str, *, path: Path) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"failed to update {path}: pattern {pattern!r} not found exactly once")
    return updated


def _write_if_changed(path: Path, text: str) -> None:
    current = path.read_text(encoding="utf-8")
    if current != text:
        path.write_text(text, encoding="utf-8")


def _sync_targets(version: str) -> None:
    cargo = ROOT / "Cargo.toml"
    cargo_text = cargo.read_text(encoding="utf-8")
    cargo_text = _replace_one(
        cargo_text,
        r'(^version\s*=\s*")([^"]+)(")$',
        rf"\g<1>{version}\3",
        path=cargo,
    )
    _write_if_changed(cargo, cargo_text)

    proteon_pyproject = ROOT / "packages" / "proteon" / "pyproject.toml"
    proteon_text = proteon_pyproject.read_text(encoding="utf-8")
    proteon_text = _replace_one(
        proteon_text,
        r'(^version\s*=\s*")([^"]+)(")$',
        rf"\g<1>{version}\3",
        path=proteon_pyproject,
    )
    proteon_text = _replace_one(
        proteon_text,
        r'("proteon-connector==)([^"]+)(")',
        rf"\g<1>{version}\3",
        path=proteon_pyproject,
    )
    _write_if_changed(proteon_pyproject, proteon_text)

    connector_pyproject = ROOT / "proteon-connector" / "pyproject.toml"
    connector_text = connector_pyproject.read_text(encoding="utf-8")
    connector_text = _replace_one(
        connector_text,
        r'(^version\s*=\s*")([^"]+)(")$',
        rf"\g<1>{version}\3",
        path=connector_pyproject,
    )
    _write_if_changed(connector_pyproject, connector_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set the repo release version from one root VERSION file."
    )
    parser.add_argument(
        "version",
        help="Semantic version to write to VERSION and synchronize into build metadata.",
    )
    args = parser.parse_args()

    try:
        _validate_version(args.version)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    VERSION_FILE.write_text(f"{args.version}\n", encoding="utf-8")
    _sync_targets(args.version)
    print(f"OK: synchronized release version {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
