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


def _workspace_member_names(cargo_text: str) -> list[str]:
    """Workspace member crate names, read from [workspace] members paths.

    Members are declared as directory paths; the crate name is the final path
    component for every member in this repo. Falling back to the directory
    name keeps this free of a second TOML parse per crate.
    """
    body = _extract_section(cargo_text, "workspace")
    match = re.search(r"(?ms)^members\s*=\s*\[(.*?)\]", body)
    if not match:
        return []
    return [entry.rstrip("/").split("/")[-1]
            for entry in re.findall(r'"([^"]+)"', match.group(1))]


def _lockfile_versions(lock_text: str, names: set[str]) -> dict[str, str]:
    """Map crate name -> version for the requested crates in Cargo.lock."""
    found: dict[str, str] = {}
    for block in re.split(r"(?m)^\[\[package\]\]\s*$", lock_text):
        name_match = re.search(r'(?m)^name\s*=\s*"([^"]+)"', block)
        if not name_match or name_match.group(1) not in names:
            continue
        version_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', block)
        if version_match:
            found[name_match.group(1)] = version_match.group(1)
    return found


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

    # Cargo.lock is generated, but it is COMMITTED, so it drifts silently:
    # bumping the workspace version in Cargo.toml does not rewrite the lock
    # until someone runs cargo, and whoever does gets a dirty tree they did
    # not create. The v0.4.0 bump left all nine workspace crates pinned at
    # 0.3.0 here, which this check did not catch because it only ever read
    # the declared versions.
    cargo_text = _read_text(ROOT / "Cargo.toml")
    member_names = set(_workspace_member_names(cargo_text))
    if member_names:
        lock_path = ROOT / "Cargo.lock"
        if not lock_path.exists():
            errors.append("Cargo.lock is missing")
        else:
            locked = _lockfile_versions(_read_text(lock_path), member_names)
            missing = sorted(member_names - set(locked))
            if missing:
                errors.append(
                    "Cargo.lock has no [[package]] entry for workspace "
                    f"member(s): {', '.join(missing)}"
                )
            stale = sorted(
                f"{name}={version}"
                for name, version in locked.items()
                if version != source_version
            )
            if stale:
                errors.append(
                    f"Cargo.lock out of sync with VERSION ({source_version}): "
                    + ", ".join(stale)
                    + ". Run `cargo metadata >/dev/null` and commit Cargo.lock."
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
