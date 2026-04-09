"""Ferritin — Structural bioinformatics toolkit powered by Rust.

Fast structure loading, alignment, and analysis with a clean Python API.

    >>> import ferritin
    >>> s = ferritin.load("1crn.pdb")
    >>> s.coords.shape
    (327, 3)
    >>> phi, psi, omega = ferritin.backbone_dihedrals(s)
    >>> cm = ferritin.contact_map(ferritin.extract_ca_coords(s), cutoff=8.0)
    >>> df = ferritin.to_dataframe(s)
"""

__version__ = "0.1.0"

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
from .prepare import (
    PrepReport,
    batch_prepare,
    load_and_prepare,
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
from .io import (
    batch_load,
    batch_load_tolerant,
    load,
    load_mmcif,
    load_pdb,
    save,
    save_mmcif,
    save_pdb,
)
from .structure import Atom, Chain, Model, Residue, Structure
