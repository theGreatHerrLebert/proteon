"""Proteon — Rust-first structural bioinformatics toolkit.

Fast structure loading, alignment, analysis, and preparation from Python, plus
an experimental structural search stack and a NumPy/Parquet-first data layer
for downstream geometric deep learning.

The top-level ``proteon`` namespace is a curated convenience surface for the
most common workflows, including the canonical "prepare and export a
structure-supervision corpus" path. More specialized or format-specific APIs
remain available from their submodules. Underscore-prefixed names and
non-exported internals are not part of the stable top-level contract.
Search-related APIs are available here, but should currently be treated as
experimental.

    >>> import proteon
    >>> s = proteon.load("1crn.pdb")
    >>> s.coords.shape
    (327, 3)
    >>> phi, psi, omega = proteon.backbone_dihedrals(s)
    >>> cm = proteon.contact_map(proteon.extract_ca_coords(s), cutoff=8.0)
    >>> df = proteon.to_dataframe(s)

DL prep, three lines from PDBs to a supervision release directory:

    >>> import proteon as p
    >>> structures = [p.load(path) for path in pdb_paths]
    >>> prep_reports = p.batch_prepare(structures)  # mutates in place
    >>> p.build_structure_supervision_dataset_from_prepared(
    ...     structures, prep_reports, out_dir="out/release", release_id="v1",
    ... )

See ``examples/10_corpus_release_smoke.py`` for the full pipeline and
``devdocs/STRUCTURE_SUPERVISION_SCHEMA.md`` for the NumPy + Parquet contract.
Framework-specific integration (PyTorch / PyG / DGL / JAX) lives in
satellite packages such as ``proteon-graphein`` and ``proteon-pyg``, never in
this core package.
"""

from importlib.metadata import PackageNotFoundError, version

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
    batch_compute_energy,
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
    batch_hbond_count,
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
from .prepare import (
    PrepReport,
    batch_prepare,
    load_and_prepare,
    normalize_histidine_tautomers,
    prepare,
)
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
    batch_atom_sasa,
    batch_relative_sasa,
    batch_residue_sasa,
    batch_total_sasa,
    load_and_sasa,
    relative_sasa,
    residue_sasa,
    total_sasa,
)
from .electrostatics import (
    born_energy,
    read_hmo,
    read_msms,
    read_off,
    read_pqr,
    surface_potential,
    write_off,
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
from .supervision_export import (
    SUPERVISION_EXPORT_FORMAT,
    SUPERVISION_PARQUET_SCHEMA_VERSION,
    SupervisionParquetWriter,
    export_structure_supervision_examples,
    iter_structure_supervision_examples,
    load_structure_supervision_examples,
)
from .prepared_manifest import (
    PreparedStructureRecord,
    build_prepared_structure_records,
    load_prepared_structure_manifest,
    write_prepared_structure_manifest,
)
from .supervision_release import (
    FailureRecord,
    StructureSupervisionReleaseManifest,
    build_structure_supervision_release,
    load_failure_records,
)
from .supervision_dataset import (
    build_structure_supervision_dataset,
    build_structure_supervision_dataset_from_prepared,
)
from .corpus_release import (
    CorpusReleaseManifest,
    build_corpus_release_manifest,
    load_corpus_release_manifest,
)
from .corpus_smoke import build_local_corpus_smoke_release
from .corpus_validation import (
    ClusterLeakageReport,
    CorpusValidationReport,
    ValidationIssue,
    validate_corpus_release,
)
from .cluster_assignments import (
    ALL_CLUSTER_NAMESPACES,
    ALL_INPUT_DIGEST_KINDS,
    CLUSTER_ASSIGNMENTS_FORMAT,
    CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION,
    DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE,
    DIGEST_KIND_CUSTOM,
    DIGEST_KIND_FASTA,
    DIGEST_KIND_RECORD_ID_LIST,
    NAMESPACE_CUSTOM,
    NAMESPACE_PREPARED_RECORD_ID,
    NAMESPACE_RAW_PDB_ID,
    NAMESPACE_UNIPROT_ID,
    ClusterAssignmentRow,
    ClusterAssignments,
    ClusterAssignmentsManifest,
    ClusterAwareSplitResult,
    ClusterCoverageError,
    ClusterCoverageReport,
    ClusterValidationReport,
    build_cluster_assignments_release,
    build_cluster_assignments_schema,
    cluster_aware_split,
    compute_record_id_digest,
    iter_cluster_assignments,
    load_cluster_assignments,
    load_cluster_assignments_jsonl,
    load_cluster_assignments_parquet,
    validate_cluster_coverage,
    validate_cluster_namespace,
    validate_cluster_record_id_uniqueness,
    validate_cluster_representative_consistency,
    validate_cluster_size_consistency,
    validate_manifest_consistency,
    write_cluster_assignments_jsonl,
    write_cluster_assignments_parquet,
)
from .sequence_export import (
    SEQUENCE_EXPORT_FORMAT,
    SEQUENCE_PARQUET_SCHEMA_VERSION,
    SequenceParquetWriter,
    export_sequence_examples,
    iter_sequence_examples,
    load_sequence_examples,
)
from .sequence_release import (
    SequenceReleaseManifest,
    build_sequence_dataset,
    build_sequence_release,
)
from .training_example import (
    TRAINING_EXPORT_FORMAT,
    TRAINING_PARQUET_SCHEMA_VERSION,
    TrainingExample,
    TrainingReleaseManifest,
    build_training_release,
    iter_training_examples,
    join_training_examples,
    load_training_examples,
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
    "batch_compute_energy",
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
    "batch_hbond_count",
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
    "normalize_histidine_tautomers",
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
    "batch_atom_sasa",
    "batch_relative_sasa",
    "batch_residue_sasa",
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

# The NumPy + Parquet-first export surface for structure-supervision
# corpora. Framework-specific adapters (PyTorch / PyG / DGL / JAX) live in
# satellite packages such as proteon-graphein and proteon-pyg — never here.
_SUPERVISION_EXPORT_API = (
    "SUPERVISION_EXPORT_FORMAT",
    "SUPERVISION_PARQUET_SCHEMA_VERSION",
    "SupervisionParquetWriter",
    "export_structure_supervision_examples",
    "iter_structure_supervision_examples",
    "load_structure_supervision_examples",
)

# Manifest + release-builder API. The dataset builders
# (build_structure_supervision_dataset[_from_prepared]) are the canonical
# high-level path from prepared structures to an on-disk supervision release;
# the lower-level pieces are exposed for callers that need finer control.
_CORPUS_RELEASE_API = (
    "CorpusReleaseManifest",
    "FailureRecord",
    "PreparedStructureRecord",
    "StructureSupervisionReleaseManifest",
    "build_corpus_release_manifest",
    "build_local_corpus_smoke_release",
    "build_prepared_structure_records",
    "build_structure_supervision_dataset",
    "build_structure_supervision_dataset_from_prepared",
    "build_structure_supervision_release",
    "load_corpus_release_manifest",
    "load_failure_records",
    "load_prepared_structure_manifest",
    "write_prepared_structure_manifest",
)

# Sequence-side export surface, parallel to _SUPERVISION_EXPORT_API. The
# canonical "structures + MSA -> sequence release" entry point is
# build_sequence_dataset; the lower-level writer/iter/load primitives are
# exposed for callers that need finer control.
_SEQUENCE_EXPORT_API = (
    "SEQUENCE_EXPORT_FORMAT",
    "SEQUENCE_PARQUET_SCHEMA_VERSION",
    "SequenceParquetWriter",
    "SequenceReleaseManifest",
    "build_sequence_dataset",
    "build_sequence_release",
    "export_sequence_examples",
    "iter_sequence_examples",
    "load_sequence_examples",
)

# Training-example surface: joins a sequence release with a structure
# supervision release into the model-facing training_example layer.
# TrainingExample is deliberately a thin join (per
# devdocs/STRUCTURE_SUPERVISION_SCHEMA.md and
# devdocs/GEOMETRIC_DL_INFRA_ROADMAP.md §10) — crop / sample / curriculum
# logic stays in model code, not in core proteon.
_TRAINING_API = (
    "TRAINING_EXPORT_FORMAT",
    "TRAINING_PARQUET_SCHEMA_VERSION",
    "TrainingExample",
    "TrainingReleaseManifest",
    "build_training_release",
    "iter_training_examples",
    "join_training_examples",
    "load_training_examples",
)

# Release-time validation. validate_corpus_release walks a release
# directory and checks count consistency, split-count consistency,
# duplicate joined record_ids, and tensor completeness across the four
# release layers (prepared / sequence / structure / training).
_CORPUS_VALIDATION_API = (
    "ClusterLeakageReport",
    "CorpusValidationReport",
    "ValidationIssue",
    "validate_corpus_release",
)

# Cluster-assignments artifact contract (v0.3.0 Phase B0). Consumed by
# Phase C (cluster_aware_split) and Phase D (cluster-leakage check in
# validate_corpus_release). The contract is deliberately consume-only —
# proteon does not own the clustering algorithm itself; upstream tools
# (MMseqs2, foldseek, …) produce the rows and proteon validates +
# joins them via the canonical post-chain-expansion record_id namespace.
_CLUSTER_ASSIGNMENTS_API = (
    # Dataclasses
    "ClusterAssignmentRow",
    "ClusterAssignments",
    "ClusterAssignmentsManifest",
    "ClusterAwareSplitResult",
    "ClusterCoverageReport",
    "ClusterValidationReport",
    "ClusterCoverageError",
    # Format + schema constants
    "CLUSTER_ASSIGNMENTS_FORMAT",
    "CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION",
    "DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE",
    # Namespace enum
    "ALL_CLUSTER_NAMESPACES",
    "NAMESPACE_PREPARED_RECORD_ID",
    "NAMESPACE_RAW_PDB_ID",
    "NAMESPACE_UNIPROT_ID",
    "NAMESPACE_CUSTOM",
    # Digest-kind enum
    "ALL_INPUT_DIGEST_KINDS",
    "DIGEST_KIND_FASTA",
    "DIGEST_KIND_RECORD_ID_LIST",
    "DIGEST_KIND_CUSTOM",
    # I/O
    "build_cluster_assignments_release",
    "build_cluster_assignments_schema",
    "compute_record_id_digest",
    "load_cluster_assignments",
    "iter_cluster_assignments",
    "write_cluster_assignments_jsonl",
    "load_cluster_assignments_jsonl",
    "write_cluster_assignments_parquet",
    "load_cluster_assignments_parquet",
    # Validators + leakage-controlled split
    "validate_cluster_namespace",
    "validate_cluster_coverage",
    "validate_cluster_representative_consistency",
    "validate_cluster_size_consistency",
    "validate_cluster_record_id_uniqueness",
    "validate_manifest_consistency",
    "cluster_aware_split",
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
    "born_energy",
    "surface_potential",
    "read_off",
    "read_pqr",
    "read_hmo",
    "read_msms",
    "write_off",
    *_SELECT_API,
    *_SEARCH_API,
    *_MSA_API,
    *_TEMPLATE_API,
    *_SEQUENCE_API,
    *_IO_API,
    *_STRUCTURE_API,
    *_SUPERVISION_API,
    *_SUPERVISION_EXPORT_API,
    *_CORPUS_RELEASE_API,
    *_SEQUENCE_EXPORT_API,
    *_TRAINING_API,
    *_CORPUS_VALIDATION_API,
    *_CLUSTER_ASSIGNMENTS_API,
    *_FAILURE_API,
)
