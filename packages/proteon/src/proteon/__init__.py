"""Proteon — Rust-first structural bioinformatics toolkit.

Fast structure loading, alignment, analysis, search, and preparation from Python.

The top-level ``proteon`` namespace is a curated convenience surface for the
most common workflows. More specialized or format-specific APIs remain available
from their submodules, e.g. ``proteon.sequence_export`` or
``proteon.supervision_export``. Underscore-prefixed names and non-exported
internals are not part of the stable top-level contract.

    >>> import proteon
    >>> s = proteon.load("1crn.pdb")
    >>> s.coords.shape
    (327, 3)
    >>> phi, psi, omega = proteon.backbone_dihedrals(s)
    >>> cm = proteon.contact_map(proteon.extract_ca_coords(s), cutoff=8.0)
    >>> df = proteon.to_dataframe(s)
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import warnings

try:
    __version__ = version("proteon")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .arrow import (
    from_arrow,
    from_parquet,
    to_arrow,
    to_parquet,
    to_structure_arrow,
)
from .align import (
    AlignResult,
    ChainPairResult,
    FlexAlignResult,
    MMAlignResult,
    SoiAlignResult,
    flex_align,
    flex_align_many_to_many,
    flex_align_one_to_many,
    mm_align,
    mm_align_many_to_many,
    mm_align_one_to_many,
    soi_align,
    soi_align_many_to_many,
    soi_align_one_to_many,
    tm_align,
    tm_align_many_to_many,
    tm_align_one_to_many,
)
from .analysis import (
    backbone_dihedrals,
    batch_contact_maps,
    batch_dihedrals,
    batch_distance_matrices,
    batch_extract_ca,
    batch_radius_of_gyration,
    centroid,
    contact_map,
    dihedral_angle,
    distance_matrix,
    extract_ca_coords,
    load_and_analyze,
    load_and_contact_maps,
    load_and_extract_ca,
    radius_of_gyration,
    to_dataframe,
)
from .core import RustWrapperObject
from .forcefield import (
    batch_minimize_hydrogens,
    compute_energy,
    gpu_available,
    gpu_info,
    load_and_minimize_hydrogens,
    minimize_hydrogens,
    minimize_structure,
    run_md,
)
from .hbond import (
    backbone_hbonds,
    batch_backbone_hbonds,
    geometric_hbonds,
    hbond_count,
)
from .hydrogens import (
    batch_place_peptide_hydrogens,
    place_all_hydrogens,
    place_general_hydrogens,
    place_peptide_hydrogens,
    reconstruct_fragments,
)
from .prepare import PrepReport, batch_prepare, load_and_prepare, prepare
from .dssp import (
    batch_dssp,
    dssp,
    dssp_array,
    load_and_dssp,
)
from .geometry import (
    apply_transform,
    assign_secondary_structure,
    kabsch_superpose,
    rmsd,
    rmsd_no_super,
    tm_score,
)
from .sasa import (
    atom_sasa,
    batch_total_sasa,
    load_and_sasa,
    relative_sasa,
    residue_sasa,
    total_sasa,
)
from .select import select
from .search import (
    SearchDB,
    SearchEntry,
    SearchHit,
    batch_encode_alphabet,
    build_search_db,
    compile_search_db,
    encode_alphabet,
    load_search_db,
    save_search_db,
    search,
    warm_search_db,
)
from .msa import MsaSearch
from .msa_backend import (
    batch_build_sequence_examples_with_msa,
    build_search_engine,
    build_search_engine_from_mmseqs_db,
    build_sequence_example_with_msa,
    open_search_engine_from_mmseqs_db_with_kmi,
    rust_msa_available,
    search_and_build_msa,
)
from .templates import TEMPLATE_GAP_INDEX, TemplateFeatures, build_template_features
from .sequence_example import (
    SequenceExample,
    batch_build_sequence_examples,
    build_sequence_example,
)
from .msa_io import load_msas_from_dir, parse_a3m_file, parse_a3m_text
from .io import (
    LoadRescueResult,
    batch_load,
    batch_load_tolerant,
    batch_load_tolerant_with_rescue,
    load,
    load_mmcif,
    load_pdb,
    load_with_rescue,
    save,
    save_mmcif,
    save_pdb,
)
from .structure import Atom, Chain, Model, Residue, Structure
from .supervision import (
    StructureQualityMetadata,
    StructureSupervisionExample,
    batch_build_structure_supervision_examples,
    build_structure_supervision_example,
)
from .failure_taxonomy import (
    ALL_FAILURE_CLASSES,
    classify_exception,
)
from .loader_failure_analysis import (
    LoaderFailureBucket,
    LoaderFailureSummary,
    bucket_loader_failure,
    load_failure_rows,
    summarize_loader_failures,
    summaries_to_markdown,
)
# Explicitly govern the top-level ``proteon`` namespace. New exports should be
# added deliberately here instead of leaking in implicitly via imports.
_ARROW_API = (
    "from_arrow",
    "from_parquet",
    "to_arrow",
    "to_parquet",
    "to_structure_arrow",
)

_ALIGN_API = (
    "AlignResult",
    "ChainPairResult",
    "FlexAlignResult",
    "MMAlignResult",
    "SoiAlignResult",
    "flex_align",
    "flex_align_many_to_many",
    "flex_align_one_to_many",
    "mm_align",
    "mm_align_many_to_many",
    "mm_align_one_to_many",
    "soi_align",
    "soi_align_many_to_many",
    "soi_align_one_to_many",
    "tm_align",
    "tm_align_many_to_many",
    "tm_align_one_to_many",
)

_ANALYSIS_API = (
    "backbone_dihedrals",
    "batch_contact_maps",
    "batch_dihedrals",
    "batch_distance_matrices",
    "batch_extract_ca",
    "batch_radius_of_gyration",
    "centroid",
    "contact_map",
    "dihedral_angle",
    "distance_matrix",
    "extract_ca_coords",
    "load_and_analyze",
    "load_and_contact_maps",
    "load_and_extract_ca",
    "radius_of_gyration",
    "to_dataframe",
)

_CORE_API = ("RustWrapperObject",)

_FORCEFIELD_API = (
    "batch_minimize_hydrogens",
    "compute_energy",
    "gpu_available",
    "gpu_info",
    "load_and_minimize_hydrogens",
    "minimize_hydrogens",
    "minimize_structure",
    "run_md",
)

_HBOND_API = (
    "backbone_hbonds",
    "batch_backbone_hbonds",
    "geometric_hbonds",
    "hbond_count",
)

_HYDROGEN_API = (
    "batch_place_peptide_hydrogens",
    "place_all_hydrogens",
    "place_general_hydrogens",
    "place_peptide_hydrogens",
    "reconstruct_fragments",
)

_PREPARE_API = (
    "PrepReport",
    "batch_prepare",
    "load_and_prepare",
    "prepare",
)

_DSSP_API = (
    "batch_dssp",
    "dssp",
    "dssp_array",
    "load_and_dssp",
)

_GEOMETRY_API = (
    "apply_transform",
    "assign_secondary_structure",
    "kabsch_superpose",
    "rmsd",
    "rmsd_no_super",
    "tm_score",
)

_SASA_API = (
    "atom_sasa",
    "batch_total_sasa",
    "load_and_sasa",
    "relative_sasa",
    "residue_sasa",
    "total_sasa",
)

_SELECT_API = ("select",)

_SEARCH_API = (
    "SearchDB",
    "SearchEntry",
    "SearchHit",
    "batch_encode_alphabet",
    "build_search_db",
    "compile_search_db",
    "encode_alphabet",
    "load_search_db",
    "save_search_db",
    "search",
    "warm_search_db",
)

_MSA_API = (
    "MsaSearch",
    "batch_build_sequence_examples_with_msa",
    "build_search_engine",
    "build_search_engine_from_mmseqs_db",
    "build_sequence_example_with_msa",
    "open_search_engine_from_mmseqs_db_with_kmi",
    "rust_msa_available",
    "search_and_build_msa",
)

_TEMPLATE_API = (
    "TEMPLATE_GAP_INDEX",
    "TemplateFeatures",
    "build_template_features",
)

_SEQUENCE_API = (
    "SequenceExample",
    "batch_build_sequence_examples",
    "build_sequence_example",
    "load_msas_from_dir",
    "parse_a3m_file",
    "parse_a3m_text",
)

_IO_API = (
    "LoadRescueResult",
    "batch_load",
    "batch_load_tolerant",
    "batch_load_tolerant_with_rescue",
    "load",
    "load_mmcif",
    "load_pdb",
    "load_with_rescue",
    "save",
    "save_mmcif",
    "save_pdb",
)

_STRUCTURE_API = (
    "Atom",
    "Chain",
    "Model",
    "Residue",
    "Structure",
)

_SUPERVISION_API = (
    "StructureQualityMetadata",
    "StructureSupervisionExample",
    "batch_build_structure_supervision_examples",
    "build_structure_supervision_example",
)

_FAILURE_API = (
    "ALL_FAILURE_CLASSES",
    "classify_exception",
    "LoaderFailureBucket",
    "LoaderFailureSummary",
    "bucket_loader_failure",
    "load_failure_rows",
    "summarize_loader_failures",
    "summaries_to_markdown",
)

__all__ = (
    "__version__",
    *_ARROW_API,
    *_ALIGN_API,
    *_ANALYSIS_API,
    *_CORE_API,
    *_FORCEFIELD_API,
    *_HBOND_API,
    *_HYDROGEN_API,
    *_PREPARE_API,
    *_DSSP_API,
    *_GEOMETRY_API,
    *_SASA_API,
    *_SELECT_API,
    *_SEARCH_API,
    *_MSA_API,
    *_TEMPLATE_API,
    *_SEQUENCE_API,
    *_IO_API,
    *_STRUCTURE_API,
    *_SUPERVISION_API,
    *_FAILURE_API,
)

_ADVANCED_DATASET_ALIASES = {
    "PreparedStructureRecord": (".prepared_manifest", "PreparedStructureRecord"),
    "build_prepared_structure_records": (
        ".prepared_manifest",
        "build_prepared_structure_records",
    ),
    "load_prepared_structure_manifest": (
        ".prepared_manifest",
        "load_prepared_structure_manifest",
    ),
    "write_prepared_structure_manifest": (
        ".prepared_manifest",
        "write_prepared_structure_manifest",
    ),
    "SEQUENCE_EXPORT_FORMAT": (".sequence_export", "SEQUENCE_EXPORT_FORMAT"),
    "SEQUENCE_PARQUET_SCHEMA_VERSION": (
        ".sequence_export",
        "SEQUENCE_PARQUET_SCHEMA_VERSION",
    ),
    "SequenceParquetWriter": (".sequence_export", "SequenceParquetWriter"),
    "export_sequence_examples": (".sequence_export", "export_sequence_examples"),
    "iter_sequence_examples": (".sequence_export", "iter_sequence_examples"),
    "load_sequence_examples": (".sequence_export", "load_sequence_examples"),
    "SequenceReleaseManifest": (".sequence_release", "SequenceReleaseManifest"),
    "build_sequence_dataset": (".sequence_release", "build_sequence_dataset"),
    "build_sequence_release": (".sequence_release", "build_sequence_release"),
    "TRAINING_EXPORT_FORMAT": (".training_example", "TRAINING_EXPORT_FORMAT"),
    "iter_training_examples": (".training_example", "iter_training_examples"),
    "load_training_examples": (".training_example", "load_training_examples"),
    "TrainingExample": (".training_example", "TrainingExample"),
    "TrainingReleaseManifest": (".training_example", "TrainingReleaseManifest"),
    "build_training_release": (".training_example", "build_training_release"),
    "join_training_examples": (".training_example", "join_training_examples"),
    "CorpusReleaseManifest": (".corpus_release", "CorpusReleaseManifest"),
    "build_corpus_release_manifest": (
        ".corpus_release",
        "build_corpus_release_manifest",
    ),
    "load_corpus_release_manifest": (
        ".corpus_release",
        "load_corpus_release_manifest",
    ),
    "CorpusValidationReport": (
        ".corpus_validation",
        "CorpusValidationReport",
    ),
    "ValidationIssue": (".corpus_validation", "ValidationIssue"),
    "validate_corpus_release": (
        ".corpus_validation",
        "validate_corpus_release",
    ),
    "build_local_corpus_smoke_release": (
        ".corpus_smoke",
        "build_local_corpus_smoke_release",
    ),
    "SUPERVISION_EXPORT_FORMAT": (
        ".supervision_export",
        "SUPERVISION_EXPORT_FORMAT",
    ),
    "export_structure_supervision_examples": (
        ".supervision_export",
        "export_structure_supervision_examples",
    ),
    "load_structure_supervision_examples": (
        ".supervision_export",
        "load_structure_supervision_examples",
    ),
    "FailureRecord": (".supervision_release", "FailureRecord"),
    "StructureSupervisionReleaseManifest": (
        ".supervision_release",
        "StructureSupervisionReleaseManifest",
    ),
    "build_structure_supervision_release": (
        ".supervision_release",
        "build_structure_supervision_release",
    ),
    "load_failure_records": (".supervision_release", "load_failure_records"),
    "build_structure_supervision_dataset": (
        ".supervision_dataset",
        "build_structure_supervision_dataset",
    ),
    "build_structure_supervision_dataset_from_prepared": (
        ".supervision_dataset",
        "build_structure_supervision_dataset_from_prepared",
    ),
}


def __getattr__(name):
    target = _ADVANCED_DATASET_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    warnings.warn(
        f"`proteon.{name}` is an advanced dataset/release API and is no longer "
        f"part of the curated top-level surface; import it from "
        f"`proteon{module_name}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    globals()[name] = value
    return value
