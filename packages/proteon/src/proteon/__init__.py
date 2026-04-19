"""Proteon — Rust-first structural bioinformatics toolkit.

Fast structure loading, alignment, analysis, and preparation from Python, plus
an experimental structural search stack.

The top-level ``proteon`` namespace is a curated convenience surface for the
most common workflows. More specialized or format-specific APIs remain available
from their submodules, e.g. ``proteon.sequence_export`` or
``proteon.supervision_export``. Underscore-prefixed names and non-exported
internals are not part of the stable top-level contract. Search-related APIs
are available here, but should currently be treated as experimental.

    >>> import proteon
    >>> s = proteon.load("1crn.pdb")
    >>> s.coords.shape
    (327, 3)
    >>> phi, psi, omega = proteon.backbone_dihedrals(s)
    >>> cm = proteon.contact_map(proteon.extract_ca_coords(s), cutoff=8.0)
    >>> df = proteon.to_dataframe(s)
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
