"""Pythonic wrappers for structure I/O.

Includes batch_load for parallel loading with rayon (GIL released).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import List, Optional, Sequence, Set, Tuple, Union

from .loader_failure_analysis import KNOWN_BUCKETS, LoaderFailureBucket, bucket_loader_failure
from .structure import Structure

try:
    import ferritin_connector

    _io = ferritin_connector.py_io
except ImportError:  # pragma: no cover
    _io = None


@dataclass(frozen=True)
class LoadRescueResult:
    """Result of an explicit loader-rescue attempt."""

    structure: Structure
    path: Path
    rescued: bool
    rescue_bucket: Optional[LoaderFailureBucket] = None
    rescue_steps: Tuple[str, ...] = ()
    original_error: Optional[str] = None


def _load_ptr(path: str, *, force_format: Optional[str] = None):
    if force_format == "pdb":
        return _io.load_pdb(path)
    if force_format == "mmcif":
        return _io.load_mmcif(path)
    return _io.load(path)


def _load_structure(path: Union[str, Path], *, force_format: Optional[str] = None) -> Structure:
    ptr = _load_ptr(str(path), force_format=force_format)
    return Structure.from_py_ptr(ptr)


def _is_pdb_like(path: Path, force_format: Optional[str]) -> bool:
    if force_format == "pdb":
        return True
    if force_format == "mmcif":
        return False
    suffixes = [s.lower() for s in path.suffixes]
    return ".pdb" in suffixes or ".ent" in suffixes


def _drop_records(text: str, prefixes: Sequence[str]) -> Tuple[str, bool]:
    prefix_set = tuple(prefixes)
    kept_lines = []
    changed = False
    for line in text.splitlines(keepends=True):
        if line.startswith(prefix_set):
            changed = True
            continue
        kept_lines.append(line)
    return "".join(kept_lines), changed


def _rewrite_zero_charge_suffix(text: str) -> Tuple[str, bool]:
    out_lines = []
    changed = False
    for line in text.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            newline = "\n" if line.endswith("\n") else ""
            body = line[:-1] if newline else line
            padded = body.ljust(80)
            if padded[78:80].strip() == "0":
                padded = padded[:78] + "  " + padded[80:]
                changed = True
            body = padded.rstrip()
            line = body + newline
        out_lines.append(line)
    return "".join(out_lines), changed


def _apply_rescue_transform(
    text: str,
    bucket: LoaderFailureBucket,
) -> Tuple[Optional[str], Tuple[str, ...]]:
    if bucket.code == "seqadv_invalid_field":
        repaired, changed = _drop_records(text, ("SEQADV",))
        return (repaired, ("drop_seqadv",)) if changed else (None, ())
    if bucket.code == "solitary_dbref1_definition":
        repaired, changed = _drop_records(text, ("DBREF1", "DBREF2"))
        return (repaired, ("drop_dbref12",)) if changed else (None, ())
    if bucket.code in {"seqres_total_invalid", "seqres_multiple_residues"}:
        repaired, changed = _drop_records(text, ("SEQRES", "DBREF ", "DBREF1", "DBREF2"))
        return (repaired, ("drop_seqres_dbref_metadata",)) if changed else (None, ())
    if bucket.code == "ssbond_missing_partner":
        repaired, changed = _drop_records(text, ("SSBOND",))
        return (repaired, ("drop_ssbond",)) if changed else (None, ())
    if bucket.code == "atom_charge_suffix":
        repaired, changed = _rewrite_zero_charge_suffix(text)
        return (repaired, ("clear_zero_charge_suffix",)) if changed else (None, ())
    return None, ()


def load_with_rescue(
    path: Union[str, Path],
    *,
    allow: Optional[Sequence[str]] = None,
    force_format: Optional[str] = None,
    max_passes: int = 3,
) -> LoadRescueResult:
    """Load one structure, retrying a narrow set of deterministic rescue fixes.

    Rescue is explicit and opt-in: the default `load()` path remains unchanged.
    Only PDB-like files are currently eligible for rescue because the initial
    repair pass operates on fixed-width PDB records.
    """
    path_obj = Path(path)
    try:
        structure = _load_structure(path_obj, force_format=force_format)
        return LoadRescueResult(structure=structure, path=path_obj, rescued=False)
    except Exception as exc:
        if not _is_pdb_like(path_obj, force_format):
            raise

        allowed = set(allow) if allow is not None else {
            code for code, item in KNOWN_BUCKETS.items() if item.rescueable
        }
        current_error = exc
        current_text = path_obj.read_text(encoding="utf-8")
        rescue_steps: List[str] = []
        seen_buckets: Set[str] = set()
        primary_bucket: Optional[LoaderFailureBucket] = None
        structure: Optional[Structure] = None

        for _ in range(max_passes):
            bucket = bucket_loader_failure(str(current_error))
            if primary_bucket is None:
                primary_bucket = bucket
            if (
                not bucket.rescueable
                or bucket.code not in allowed
                or bucket.code in seen_buckets
            ):
                raise current_error
            seen_buckets.add(bucket.code)

            repaired_text, new_steps = _apply_rescue_transform(current_text, bucket)
            if repaired_text is None:
                raise current_error
            current_text = repaired_text
            rescue_steps.extend(new_steps)

            suffix = path_obj.suffix or ".pdb"
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=suffix,
                encoding="utf-8",
                delete=False,
            ) as handle:
                handle.write(current_text)
                tmp_path = Path(handle.name)
            try:
                structure = _load_structure(tmp_path, force_format="pdb")
                break
            except Exception as retry_exc:
                current_error = retry_exc
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            raise current_error

        return LoadRescueResult(
            structure=structure,
            path=path_obj,
            rescued=True,
            rescue_bucket=primary_bucket,
            rescue_steps=tuple(rescue_steps),
            original_error=str(exc),
        )


def batch_load_tolerant_with_rescue(
    paths: Sequence[Union[str, Path]],
    *,
    n_threads: Optional[int] = None,
    allow: Optional[Sequence[str]] = None,
    force_format: Optional[str] = None,
) -> List[Tuple[int, LoadRescueResult]]:
    """Load many structures, skipping failures and retrying deterministic rescues.

    This path is intentionally serial today because rescue decisions depend on the
    exact per-file exception text and may require Python-side text rewriting.
    """
    del n_threads
    results: List[Tuple[int, LoadRescueResult]] = []
    for i, path in enumerate(paths):
        try:
            result = load_with_rescue(path, allow=allow, force_format=force_format)
        except Exception:
            continue
        results.append((i, result))
    return results


def load(path: Union[str, Path]) -> Structure:
    """Load a structure from a PDB or mmCIF file.

    Format is auto-detected from the file extension.

    Args:
        path: Path to the file (.pdb, .cif, .mmcif).

    Returns:
        Structure: The parsed structure.

    Agent Notes:
        PREFER: batch_load() for multiple files. Do not loop in Python.
        WATCH: residue_count includes water and ligands. Amino acid count
            may be lower.
    """
    return _load_structure(path)


def load_pdb(path: Union[str, Path]) -> Structure:
    """Load a structure, forcing PDB format."""
    return _load_structure(path, force_format="pdb")


def load_mmcif(path: Union[str, Path]) -> Structure:
    """Load a structure, forcing mmCIF format."""
    return _load_structure(path, force_format="mmcif")


def save(structure: Structure, path: Union[str, Path]) -> None:
    """Save a structure to a PDB or mmCIF file.

    Format is auto-detected from the file extension.

    Args:
        structure: The structure to save.
        path: Output file path (.pdb or .cif/.mmcif).
    """
    _io.save(structure.get_py_ptr(), str(path))


def save_pdb(structure: Structure, path: Union[str, Path]) -> None:
    """Save a structure, forcing PDB format."""
    _io.save_pdb(structure.get_py_ptr(), str(path))


def save_mmcif(structure: Structure, path: Union[str, Path]) -> None:
    """Save a structure, forcing mmCIF format."""
    _io.save_mmcif(structure.get_py_ptr(), str(path))


def batch_load(
    paths: Sequence[Union[str, Path]],
    *,
    n_threads: Optional[int] = None,
) -> List[Structure]:
    """Load many structures in parallel using rayon (GIL released).

    All file I/O, parsing, and PDB construction happens in Rust
    across multiple threads. Much faster than a Python loop.

    Args:
        paths: List of file paths (.pdb, .cif, .mmcif).
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of Structure objects (same order as paths).

    Raises:
        IOError: If any file fails to load.

    Examples:
        >>> structures = ferritin.batch_load(glob.glob("pdbs/*.pdb"), n_threads=-1)
    """
    str_paths = [str(p) for p in paths]
    ptrs = _io.batch_load(str_paths, n_threads)
    return [Structure.from_py_ptr(ptr) for ptr in ptrs]


def batch_load_tolerant(
    paths: Sequence[Union[str, Path]],
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[int, Structure]]:
    """Load many structures in parallel, skipping failures.

    Same as batch_load but doesn't raise on individual failures.

    Args:
        paths: List of file paths.
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of (index, Structure) tuples for files that loaded successfully.
        The index refers to the position in the original paths list.

    Examples:
        >>> results = ferritin.batch_load_tolerant(all_files, n_threads=-1)
        >>> print(f"{len(results)}/{len(all_files)} loaded")

    Agent Notes:
        WATCH: Failures are skipped silently at the return-value level. Always use
            the returned indices to map loaded structures back to the original list.
        PREFER: Use this for archive-scale ingestion where partial success is acceptable.
    """
    str_paths = [str(p) for p in paths]
    pairs = _io.batch_load_tolerant(str_paths, n_threads)
    return [(i, Structure.from_py_ptr(ptr)) for i, ptr in pairs]
