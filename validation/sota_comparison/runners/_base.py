"""Base types and registry for the SOTA comparison runners.

Every runner is a function `(pdb_path: str) -> RunnerResult`. Runners register
themselves on a per-op basis via the `@register` decorator. The aggregator
pairs `proteon` against every other registered impl per op.

Design decisions (locked, see plan file):
- Per-op payload schemas. No free-form dicts.
- Units normalized inside the runner: SASA in Å², energy in kJ/mol, coords in Å.
- Per-residue results keyed by `(chain, resi, icode)` tuples — never positional.
- "Tool doesn't apply" → don't register, no sentinel None.
- `status="skip"` is reserved for runtime skips (e.g. CA-only structure mkdssp
  can't process). "Doesn't apply" structures don't show up at all.
"""

from __future__ import annotations

import dataclasses
import subprocess
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional


# Global registry: op_name -> list of (impl_name, runner_callable).
OPS: Dict[str, List[tuple]] = defaultdict(list)

# Optional batch runners keyed by (op, impl). When present, the driver
# prefers the batch runner over the per-structure one to unlock in-Rust
# parallelism across the full PDB set. Batch runner signature:
#   def batch_fn(pdb_paths: List[str]) -> List[RunnerResult]
BATCH_RUNNERS: Dict[tuple, Callable] = {}

# Modules that failed to import are recorded here so the driver can report
# them without crashing the whole pipeline (e.g., freesasa not installed).
IMPORT_FAILURES: Dict[str, str] = {}


@dataclasses.dataclass
class RunnerResult:
    """Uniform envelope for every runner result.

    Attributes:
        op: The operation being measured ("sasa", "energy", "dssp", ...).
        impl: The implementation that produced this result ("proteon",
            "freesasa", "openmm", ...).
        impl_version: Tool version string (e.g. "freesasa 2.2.1"). Empty if
            unknown — log a warning so the user knows to fix it.
        pdb_id: The PDB ID this result is for (e.g. "1crn").
        pdb_path: Absolute path to the input file the runner was called with.
        elapsed_s: Wall time of the runner call only (not import / warmup).
        status: "ok" — payload is valid; "skip" — runtime skip with a reason
            in `error`; "error" — runner crashed with the exception in `error`.
        error: Failure reason if status != "ok", else None.
        payload: Op-specific dict matching the schema for that op. See the
            module docstrings of `runners/sasa.py`, `runners/energy.py`, etc.
    """

    op: str
    impl: str
    impl_version: str
    pdb_id: str
    pdb_path: str
    elapsed_s: float
    status: str  # "ok" | "skip" | "error"
    error: Optional[str]
    payload: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def register(op: str, impl: str) -> Callable:
    """Decorator that registers a runner for `(op, impl)` in the global registry.

    Usage:
        @register("sasa", "freesasa")
        def freesasa(pdb_path: str) -> RunnerResult:
            ...

    The decorated function is left unchanged so it can be called directly in
    tests. Registry order is insertion order; `proteon` should be registered
    first in each op module so it appears as the baseline column in reports.
    """

    def decorator(fn: Callable[[str], RunnerResult]) -> Callable[[str], RunnerResult]:
        OPS[op].append((impl, fn))
        return fn

    return decorator


def register_batch(op: str, impl: str) -> Callable:
    """Decorator that registers a batched runner for `(op, impl)`.

    A batched runner takes a list of PDB paths and returns a list of
    RunnerResult envelopes (same length). The driver prefers the batched
    runner when one is registered, because it unlocks in-Rust parallelism
    across the full PDB set (e.g. batch_prepare processing 200 structures
    at once on 120 cores is ~100x faster than 200 single-structure calls).

    Usage:
        @register_batch("energy", "proteon")
        def proteon_batch(paths: List[str]) -> List[RunnerResult]:
            ...

    The corresponding non-batched runner should still be registered via
    `@register`, because the driver's determinism check and warmup pass
    use the single-structure path.
    """

    def decorator(fn: Callable) -> Callable:
        BATCH_RUNNERS[(op, impl)] = fn
        return fn

    return decorator


def subprocess_version(cmd: str, *args: str, timeout: float = 5.0) -> str:
    """Best-effort version detection for an external CLI tool.

    Tries the given args (typically `--version` or `-V`) and returns the first
    non-empty line of stdout/stderr. Returns "unknown" on any failure.
    """
    try:
        proc = subprocess.run(
            [cmd, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or proc.stderr or "").strip().split("\n")[0].strip()
        return out or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"


def time_call(fn: Callable, *args, **kwargs):
    """Run `fn` and return `(result, elapsed_s)`. Wall time of the call only."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def make_error_result(
    op: str,
    impl: str,
    impl_version: str,
    pdb_id: str,
    pdb_path: str,
    elapsed_s: float,
    error: str,
) -> RunnerResult:
    """Construct a RunnerResult for a failed runner call.

    Use this in `except` blocks so error envelopes are uniformly shaped.
    """
    return RunnerResult(
        op=op,
        impl=impl,
        impl_version=impl_version,
        pdb_id=pdb_id,
        pdb_path=pdb_path,
        elapsed_s=elapsed_s,
        status="error",
        error=error,
        payload={},
    )


def make_skip_result(
    op: str,
    impl: str,
    impl_version: str,
    pdb_id: str,
    pdb_path: str,
    reason: str,
) -> RunnerResult:
    """Construct a RunnerResult for a runtime skip (e.g. CA-only structure)."""
    return RunnerResult(
        op=op,
        impl=impl,
        impl_version=impl_version,
        pdb_id=pdb_id,
        pdb_path=pdb_path,
        elapsed_s=0.0,
        status="skip",
        error=reason,
        payload={},
    )
