#!/usr/bin/env python3
"""Validate the EVIDENT manifest schema.

See workflow/SCHEMA.md for the full schema. This module checks structure
and vocabulary consistency, not scientific truth. Domain-specific claims
still need their own oracle or benchmark commands.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - only hit in incomplete images
    raise SystemExit("Missing dependency: PyYAML") from exc


BASE_VOCABULARIES: dict[str, set[str]] = {
    "tolerance_metric": {
        "relative_error",
        "median_relative_error",
        "absolute_error",
        "pass_rate",
        "recall",
        "precision",
        "f1",
        "drift",
    },
    "tolerance_op": {"<", "<=", ">=", ">", "=="},
    "input_class": {
        "single-chain",
        "multi-chain",
        "random-sample",
        "synthetic",
        "fixture",
    },
    "subsystem": set(),
    "oracle": set(),
    "capability": set(),
}

VALID_TIERS = {"ci", "release", "research"}
VALID_TRUST_STRATEGIES = {"understanding", "validation", "proof"}
VALID_KINDS = {"measurement", "policy", "reference"}
VALID_PROVENANCE = {"automatic", "human", "peer-reviewed"}
# Mirror of claim_scoring.VALID_FORMATS / VALID_AGGREGATES. Kept as local
# literals so this validator carries no hard import dependency on the scorer
# (it doubles as the framework-level workflow/validate_manifest.py).
VALID_SCORING_FORMATS = {"jsonl", "json"}
VALID_SCORING_AGGREGATES = {"median", "mean", "max", "min", "value", "fraction"}
PLACEHOLDER_PREFIXES = ("PENDING-", "TODO", "TBD")

REQUIRED_FIELDS_ALL = {
    "id",
    "title",
    "case",
    "source",
    "tier",
    "trust_strategy",
    "claim",
    "evidence",
    "assumptions",
    "failure_modes",
}
REQUIRED_FIELDS_MEASUREMENT = {
    "subsystem",
    "inputs",
    "pinned_versions",
    "tolerances",
}


def fail(message: str) -> None:
    raise ValueError(message)


def require_non_empty_string(value: Any, field: str, claim_id: str) -> None:
    if not isinstance(value, str) or not value.strip():
        fail(f"claim {claim_id}: {field} must be a non-empty string")


def is_placeholder_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip().upper()
    return any(stripped.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def require_not_placeholder(value: Any, field: str, claim_id: str) -> None:
    if is_placeholder_string(value):
        fail(f"claim {claim_id}: {field} must not be a placeholder")


def require_string_list(value: Any, field: str, claim_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        fail(f"claim {claim_id}: {field} must be a non-empty list")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            fail(f"claim {claim_id}: {field} must contain only non-empty strings")
    return value


def require_in_vocab(
    value: str,
    vocab_name: str,
    vocabularies: dict[str, set[str]],
    field: str,
    claim_id: str,
) -> None:
    vocab = vocabularies[vocab_name]
    if not vocab:
        fail(
            f"claim {claim_id}: vocabulary {vocab_name!r} is empty; "
            f"declare it in the manifest's vocabularies: block before using {field}"
        )
    if value not in vocab:
        fail(
            f"claim {claim_id}: {field} {value!r} not in vocabulary "
            f"{vocab_name}; allowed: {sorted(vocab)}"
        )


def validate_existing_path(
    root: pathlib.Path, value: Any, field: str, claim_id: str
) -> None:
    require_non_empty_string(value, field, claim_id)
    path = root / value
    if "#" in str(path):
        path = pathlib.Path(str(path).split("#", 1)[0])
    if not path.exists():
        fail(f"claim {claim_id}: {field} path does not exist: {value}")


def validate_evidence(
    value: Any, claim_id: str, vocabularies: dict[str, set[str]]
) -> list[str]:
    if not isinstance(value, dict):
        fail(f"claim {claim_id}: evidence must be a mapping")
    for field in ("oracle", "command", "artifact"):
        if field not in value:
            fail(f"claim {claim_id}: evidence.{field} is required")
    if "tolerance" in value:
        fail(
            f"claim {claim_id}: evidence.tolerance is no longer supported; "
            f"move tolerance text into the top-level tolerances: list "
            f"(see workflow/SCHEMA.md)"
        )
    oracles = require_string_list(value["oracle"], "evidence.oracle", claim_id)
    for oracle_name in oracles:
        require_in_vocab(
            oracle_name, "oracle", vocabularies, "evidence.oracle[]", claim_id
        )
    require_non_empty_string(value["command"], "evidence.command", claim_id)
    require_non_empty_string(value["artifact"], "evidence.artifact", claim_id)
    return oracles


def validate_tolerances(
    value: Any, claim_id: str, vocabularies: dict[str, set[str]]
) -> None:
    if not isinstance(value, list) or not value:
        fail(f"claim {claim_id}: tolerances must be a non-empty list")
    for i, entry in enumerate(value):
        if not isinstance(entry, dict):
            fail(f"claim {claim_id}: tolerances[{i}] must be a mapping")
        prose = entry.get("prose")
        if not isinstance(prose, str) or not prose.strip():
            fail(f"claim {claim_id}: tolerances[{i}].prose is required")
        has_metric = "metric" in entry
        has_op = "op" in entry
        has_value = "value" in entry
        if any([has_metric, has_op, has_value]) and not all(
            [has_metric, has_op, has_value]
        ):
            fail(
                f"claim {claim_id}: tolerances[{i}] metric/op/value are "
                f"all-or-nothing"
            )
        if has_metric:
            require_in_vocab(
                entry["metric"],
                "tolerance_metric",
                vocabularies,
                f"tolerances[{i}].metric",
                claim_id,
            )
            require_in_vocab(
                entry["op"],
                "tolerance_op",
                vocabularies,
                f"tolerances[{i}].op",
                claim_id,
            )
            if not isinstance(entry["value"], (int, float)) or isinstance(
                entry["value"], bool
            ):
                fail(f"claim {claim_id}: tolerances[{i}].value must be numeric")
        if "scoring" in entry:
            validate_scoring(entry, i, claim_id)


def validate_scoring(entry: dict, i: int, claim_id: str) -> None:
    """Schema-check an optional ``scoring:`` block on a tolerance.

    Structural only — confirms the scorer can be pointed at the artifact.
    Whether the recorded band is actually met is decided by claim_scoring
    at score/replay/lock time, not here.
    """
    scoring = entry["scoring"]
    if not isinstance(scoring, dict):
        fail(f"claim {claim_id}: tolerances[{i}].scoring must be a mapping")
    # A scorable tolerance must carry the metric/op/value it scores against.
    for key in ("metric", "op", "value"):
        if key not in entry:
            fail(
                f"claim {claim_id}: tolerances[{i}] has scoring: but no {key} "
                f"to score against"
            )
    artifact = scoring.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        fail(f"claim {claim_id}: tolerances[{i}].scoring.artifact must be a non-empty string")
    fmt = scoring.get("format")
    if fmt not in VALID_SCORING_FORMATS:
        fail(
            f"claim {claim_id}: tolerances[{i}].scoring.format {fmt!r} invalid; "
            f"allowed: {sorted(VALID_SCORING_FORMATS)}"
        )
    if "aggregate" in scoring and scoring["aggregate"] not in VALID_SCORING_AGGREGATES:
        fail(
            f"claim {claim_id}: tolerances[{i}].scoring.aggregate "
            f"{scoring['aggregate']!r} invalid; allowed: {sorted(VALID_SCORING_AGGREGATES)}"
        )
    # fraction (the pass_rate aggregate) needs a per-record numerator predicate.
    is_fraction = scoring.get("aggregate") == "fraction" or (
        "aggregate" not in scoring and entry.get("metric") == "pass_rate"
    )
    if is_fraction and not isinstance(scoring.get("pass_when"), dict):
        fail(
            f"claim {claim_id}: tolerances[{i}].scoring needs a pass_when mapping "
            f"for fraction/pass_rate scoring"
        )
    if "select" in scoring:
        select = scoring["select"]
        if not isinstance(select, dict) or not isinstance(select.get("list"), str):
            fail(
                f"claim {claim_id}: tolerances[{i}].scoring.select must be a mapping "
                f"with a 'list' field name"
            )


def validate_inputs(
    value: Any,
    claim_id: str,
    vocabularies: dict[str, set[str]],
    tier: str,
    *,
    strict_release_pins: bool,
) -> None:
    if not isinstance(value, dict):
        fail(f"claim {claim_id}: inputs must be a mapping")
    if "class" in value:
        require_in_vocab(
            value["class"], "input_class", vocabularies, "inputs.class", claim_id
        )
    n = value.get("n", 0) or 0
    if not isinstance(n, int):
        fail(f"claim {claim_id}: inputs.n must be an integer")
    if tier == "release" and n > 1 and not value.get("corpus_sha"):
        fail(
            f"claim {claim_id}: inputs.corpus_sha is required for "
            f"tier=release with n>1"
        )
    if strict_release_pins and tier == "release":
        require_not_placeholder(
            value.get("corpus_sha"), "inputs.corpus_sha", claim_id
        )


def validate_pinned_versions(
    value: Any,
    oracle_names: list[str],
    project: str,
    claim_id: str,
    *,
    tier: str,
    strict_release_pins: bool,
) -> None:
    if not isinstance(value, dict) or not value:
        fail(f"claim {claim_id}: pinned_versions must be a non-empty mapping")
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            fail(f"claim {claim_id}: pinned_versions keys must be non-empty strings")
        if not isinstance(v, str) or not v.strip():
            fail(
                f"claim {claim_id}: pinned_versions[{k!r}] must be a non-empty "
                f"string (quote numeric versions like \"1.83\")"
            )
        if strict_release_pins and tier == "release":
            require_not_placeholder(v, f"pinned_versions[{k!r}]", claim_id)
    if project not in value:
        fail(
            f"claim {claim_id}: pinned_versions must include the project under "
            f"test ({project!r})"
        )
    missing_oracles = [o for o in oracle_names if o not in value]
    if missing_oracles:
        fail(
            f"claim {claim_id}: pinned_versions must include every oracle "
            f"named in evidence.oracle; missing: {missing_oracles}"
        )


def validate_outputs(value: Any, claim_id: str) -> None:
    if not isinstance(value, dict) or not value:
        fail(f"claim {claim_id}: outputs must be a non-empty mapping")
    for name, body in value.items():
        if not isinstance(name, str) or not name.strip():
            fail(f"claim {claim_id}: outputs keys must be non-empty strings")
        if not isinstance(body, dict):
            fail(f"claim {claim_id}: outputs[{name!r}] must be a mapping")


def validate_provenance_and_reviewers(claim: dict, claim_id: str) -> None:
    provenance = claim.get("provenance", "automatic")
    if provenance not in VALID_PROVENANCE:
        fail(
            f"claim {claim_id}: invalid provenance {provenance!r}; "
            f"allowed: {sorted(VALID_PROVENANCE)}"
        )
    reviewers = claim.get("reviewers")
    if provenance == "peer-reviewed":
        if not isinstance(reviewers, list) or not reviewers:
            fail(
                f"claim {claim_id}: provenance=peer-reviewed requires a "
                f"non-empty reviewers list"
            )
    else:
        if reviewers is not None:
            fail(
                f"claim {claim_id}: reviewers may only be set when "
                f"provenance=peer-reviewed (got provenance={provenance!r})"
            )
        return
    for i, entry in enumerate(reviewers):
        if not isinstance(entry, dict):
            fail(f"claim {claim_id}: reviewers[{i}] must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            fail(f"claim {claim_id}: reviewers[{i}].name is required")
        for opt in ("orcid", "affiliation", "date"):
            if opt in entry and entry[opt] is not None:
                if not isinstance(entry[opt], str) or not entry[opt].strip():
                    fail(
                        f"claim {claim_id}: reviewers[{i}].{opt} must be a "
                        f"non-empty string when present"
                    )
        unknown = set(entry) - {"name", "orcid", "affiliation", "date"}
        if unknown:
            fail(
                f"claim {claim_id}: reviewers[{i}] has unknown keys: "
                f"{sorted(unknown)}"
            )


def validate_last_verified(
    value: Any,
    claim_id: str,
    *,
    tier: str,
    strict_release_pins: bool,
) -> None:
    if not isinstance(value, dict):
        fail(f"claim {claim_id}: last_verified must be a mapping")
    for key in ("commit", "date", "value", "corpus_sha"):
        if key not in value or value[key] is None:
            continue
        if key == "value":
            if not isinstance(value[key], (int, float)) or isinstance(
                value[key], bool
            ):
                fail(f"claim {claim_id}: last_verified.value must be numeric or null")
        elif not isinstance(value[key], str):
            fail(f"claim {claim_id}: last_verified.{key} must be a string or null")
    if strict_release_pins and tier == "release":
        for key in ("commit", "corpus_sha"):
            if key not in value or value[key] is None:
                fail(f"claim {claim_id}: last_verified.{key} is required")
            require_not_placeholder(value[key], f"last_verified.{key}", claim_id)


def merge_vocabularies(declared: Any) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {k: set(v) for k, v in BASE_VOCABULARIES.items()}
    if declared is None:
        return merged
    if not isinstance(declared, dict):
        fail("vocabularies must be a mapping")
    for axis, items in declared.items():
        if axis not in merged:
            fail(
                f"unknown vocabulary axis: {axis!r} "
                f"(allowed: {sorted(merged)})"
            )
        if not isinstance(items, list):
            fail(f"vocabularies.{axis} must be a list of strings")
        for item in items:
            if not isinstance(item, str) or not item.strip():
                fail(f"vocabularies.{axis} must contain only non-empty strings")
        merged[axis].update(items)
    return merged


def _load_yaml_mapping(path: pathlib.Path, label: str) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"{label} must be a mapping: {path}")
    return data


def _collect(
    top_path: pathlib.Path,
) -> tuple[str, dict[str, set[str]], list[Any]]:
    """Return (project, vocabularies, claims) from the manifest + includes.

    Included files are loaded from paths resolved relative to the TOP
    manifest's directory and may contain their own `claims:` list. They do
    not chain further `include:` directives — keep the index flat and
    explicit.
    """
    root = top_path.parent
    data = _load_yaml_mapping(top_path, "manifest")
    if data.get("version") is None:
        fail("version is required")
    project = data.get("project")
    if not isinstance(project, str) or not project.strip():
        fail("project is required (top-level non-empty string)")

    vocabularies = merge_vocabularies(data.get("vocabularies"))

    claims: list[Any] = list(data.get("claims") or [])
    includes = data.get("include") or []
    if not isinstance(includes, list):
        fail("include must be a list of paths")

    for raw in includes:
        if not isinstance(raw, str) or not raw.strip():
            fail("include entries must be non-empty strings")
        included_path = root / raw
        if not included_path.exists():
            fail(f"include path does not exist: {raw}")
        included = _load_yaml_mapping(included_path, f"included manifest {raw}")
        included_claims = included.get("claims")
        if not isinstance(included_claims, list) or not included_claims:
            fail(f"include {raw}: claims must be a non-empty list")
        claims.extend(included_claims)

    if not claims:
        fail("claims must be a non-empty list (inline or via include:)")
    return project, vocabularies, claims


def _collect_claims(top_path: pathlib.Path) -> list[Any]:
    """Backward-compatible helper for callers (e.g. evident.py CLI)."""
    _, _, claims = _collect(top_path)
    return claims


def validate_manifest(path: pathlib.Path, *, strict_release_pins: bool = False) -> None:
    root = path.parent
    project, vocabularies, claims = _collect(path)

    seen_ids: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            fail(f"claim at index {index} must be a mapping")
        claim_id = str(claim.get("id", f"<index:{index}>"))

        missing_common = sorted(REQUIRED_FIELDS_ALL - claim.keys())
        if missing_common:
            fail(
                f"claim {claim_id}: missing required fields: "
                f"{', '.join(missing_common)}"
            )
        require_non_empty_string(claim["id"], "id", claim_id)
        if claim["id"] in seen_ids:
            fail(f"duplicate claim id: {claim['id']}")
        seen_ids.add(claim["id"])

        require_non_empty_string(claim["title"], "title", claim_id)
        require_non_empty_string(claim["claim"], "claim", claim_id)
        validate_existing_path(root, claim["case"], "case", claim_id)
        validate_existing_path(root, claim["source"], "source", claim_id)
        if "pattern" in claim:
            validate_existing_path(root, claim["pattern"], "pattern", claim_id)

        kind = claim.get("kind", "measurement")
        if kind not in VALID_KINDS:
            fail(f"claim {claim_id}: invalid kind {kind!r}")

        require_non_empty_string(claim["tier"], "tier", claim_id)
        if claim["tier"] not in VALID_TIERS:
            fail(f"claim {claim_id}: invalid tier {claim['tier']!r}")

        strategies = require_string_list(
            claim["trust_strategy"], "trust_strategy", claim_id
        )
        invalid = sorted(set(strategies) - VALID_TRUST_STRATEGIES)
        if invalid:
            fail(
                f"claim {claim_id}: invalid trust strategies: {', '.join(invalid)}"
            )

        oracles = validate_evidence(claim["evidence"], claim_id, vocabularies)
        require_string_list(claim["assumptions"], "assumptions", claim_id)
        require_string_list(claim["failure_modes"], "failure_modes", claim_id)

        if "capabilities" in claim:
            caps = require_string_list(
                claim["capabilities"], "capabilities", claim_id
            )
            for cap in caps:
                require_in_vocab(
                    cap, "capability", vocabularies, "capabilities[]", claim_id
                )

        if "outputs" in claim:
            validate_outputs(claim["outputs"], claim_id)
        if "last_verified" in claim:
            validate_last_verified(
                claim["last_verified"],
                claim_id,
                tier=claim["tier"],
                strict_release_pins=strict_release_pins,
            )
        validate_provenance_and_reviewers(claim, claim_id)

        if kind == "measurement":
            missing_meas = sorted(REQUIRED_FIELDS_MEASUREMENT - claim.keys())
            if missing_meas:
                fail(
                    f"claim {claim_id}: kind=measurement missing required "
                    f"fields: {', '.join(missing_meas)}"
                )
            require_non_empty_string(claim["subsystem"], "subsystem", claim_id)
            require_in_vocab(
                claim["subsystem"], "subsystem", vocabularies, "subsystem", claim_id
            )
            validate_inputs(
                claim["inputs"],
                claim_id,
                vocabularies,
                claim["tier"],
                strict_release_pins=strict_release_pins,
            )
            validate_tolerances(claim["tolerances"], claim_id, vocabularies)
            validate_pinned_versions(
                claim["pinned_versions"],
                oracles,
                project,
                claim_id,
                tier=claim["tier"],
                strict_release_pins=strict_release_pins,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="evident.yaml")
    parser.add_argument(
        "--strict-release-pins",
        action="store_true",
        help=(
            "reject placeholder corpus hashes, pinned versions, and "
            "last_verified release pins"
        ),
    )
    args = parser.parse_args()

    try:
        validate_manifest(
            pathlib.Path(args.manifest),
            strict_release_pins=args.strict_release_pins,
        )
    except Exception as exc:
        print(f"manifest invalid: {exc}", file=sys.stderr)
        return 1
    print(f"manifest valid: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
