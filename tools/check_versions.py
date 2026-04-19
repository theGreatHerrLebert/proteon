#!/usr/bin/env python3
"""Ensure release versions stay synchronized across the repo.

This is intentionally conservative: the build metadata still carries explicit
versions in multiple places, so CI should fail as soon as they drift from the
repo-root VERSION file.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"


def _load_toml(path: Path) -> dict:
    if tomllib is None:
        raise RuntimeError("tomllib unavailable")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_section(text: str, section: str) -> str:
    match = re.search(
        rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    if not match:
        raise ValueError(f"missing TOML section [{section}]")
    return match.group(1)


def _extract_string(body: str, key: str) -> str:
    match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"', body)
    if not match:
        raise ValueError(f"missing string key {key!r}")
    return match.group(1)


def _extract_dependencies(body: str) -> list[str]:
    match = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)\]", body)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def main() -> int:
    source_version = VERSION_FILE.read_text(encoding="utf-8").strip()

    try:
        cargo = _load_toml(ROOT / "Cargo.toml")
        proteon = _load_toml(ROOT / "packages" / "proteon" / "pyproject.toml")
        connector = _load_toml(ROOT / "proteon-connector" / "pyproject.toml")

        workspace_version = cargo["workspace"]["package"]["version"]
        proteon_version = proteon["project"]["version"]
        connector_version = connector["project"]["version"]
        deps = proteon["project"].get("dependencies", [])
    except RuntimeError:
        cargo_body = _extract_section(_read_text(ROOT / "Cargo.toml"), "workspace.package")
        proteon_body = _extract_section(
            _read_text(ROOT / "packages" / "proteon" / "pyproject.toml"),
            "project",
        )
        connector_body = _extract_section(
            _read_text(ROOT / "proteon-connector" / "pyproject.toml"),
            "project",
        )
        workspace_version = _extract_string(cargo_body, "version")
        proteon_version = _extract_string(proteon_body, "version")
        connector_version = _extract_string(connector_body, "version")
        deps = _extract_dependencies(proteon_body)

    errors: list[str] = []
    versions = {
        "VERSION": source_version,
        "workspace": workspace_version,
        "proteon": proteon_version,
        "proteon-connector": connector_version,
    }
    if len(set(versions.values())) != 1:
        errors.append(
            "Version mismatch: "
            + ", ".join(f"{name}={version}" for name, version in versions.items())
        )

    expected_connector_pin = f"proteon-connector=={connector_version}"
    actual_connector_dep = next(
        (dep for dep in deps if dep.startswith("proteon-connector")),
        None,
    )
    if actual_connector_dep != expected_connector_pin:
        errors.append(
            "packages/proteon dependency mismatch: "
            f"expected '{expected_connector_pin}', got '{actual_connector_dep}'"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: synchronized release version {workspace_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
