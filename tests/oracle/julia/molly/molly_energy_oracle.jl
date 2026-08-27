#!/usr/bin/env julia
# Molly.jl oracle: per-component AMBER96 (+ OBC GB) energies on a PDB structure.
# Outputs JSON with an energy breakdown for comparison with proteon and OpenMM.
#
# Molly is a *third* independent implementation of the same force field, and it
# parses the same OpenMM `amber96.xml` that `validation/amber96_oracle.py` feeds
# to OpenMM. Where proteon and OpenMM disagree, Molly breaks the tie.
#
# Usage:
#   julia --project=<this dir> molly_energy_oracle.jl \
#         --ff /path/to/amber96.xml [--solvent obc1|obc2|gbn2|none] \
#         structure.pdb [structure2.pdb ...]
#
# Notes on comparison hygiene (see devdocs/ORACLE.md):
#   * `nonbonded_method=:none` disables the cutoff. Proteon's production default
#     is a 15 Å cutoff; oracle-grade comparison must isolate the force-field
#     math from the cutoff approximation.
#   * `dist_cutoff` must ALSO be widened. `nonbonded_method=:none` does not
#     imply an infinite interaction radius: Molly still builds a neighbour list
#     at `dist_cutoff`, which defaults to 1.0 nm. Leaving it at the default
#     truncates long-range Coulomb and cost ~2600 kJ/mol on crambin (an
#     apparent 129% "disagreement" that was purely a setup error). The box and
#     cutoff are sized from the structure's own extent below.
#   * A large cubic box is imposed so the minimum-image convention can never
#     fold a long-range pair back onto itself. The PDB's own CRYST1 box for a
#     small protein is far too small for an effectively-infinite cutoff.
#   * Molly ignores `GBSAOBCForce` when parsing OpenMM XML (it warns) and
#     instead sources OBC radii/screens from its own element table. The GB term
#     is therefore an *independently parameterised* second opinion, not a
#     re-read of the same numbers.

using Molly
using Unitful
using JSON

const KJ = u"kJ * mol^-1"

"""Energy of a system restricted to one interaction subset, in kJ/mol.

Uses the `System(sys; ...)` copy constructor to null out every interaction
except the requested one — the same idiom Molly's own OpenMM regression tests
use (`test/protein.jl`). The copy carries `neighbor_finder` over unchanged, so
1-2 / 1-3 exclusions and 1-4 scaling stay correct in the subset.
"""
function subset_energy(sys, neighbors; pairwise=(), specific=(), general=())
    part = System(sys; pairwise_inters=pairwise, specific_inter_lists=specific,
                  general_inters=general, strictness=:nowarn)
    return ustrip(uconvert(KJ, potential_energy(part, neighbors; n_threads=1)))
end

"""Box side and neighbour-list cutoff that make the cutoff effectively infinite.

Sized from the structure itself rather than hardcoded, so this stays correct
for inputs much larger than crambin. The cutoff must exceed the longest
intramolecular distance (or long-range Coulomb is truncated) while staying
below half the box side (or the minimum-image convention folds pairs back).
"""
function box_and_cutoff(pdb_path::String, ff::MolecularForceField)
    # The probe needs a box large enough to pass Molly's unit-cell check
    # regardless of what CRYST1 says; its only job is to expose coordinates.
    probe = System(pdb_path, ff; boundary=CubicBoundary(1000.0u"nm"),
                   nonbonded_method=:none, dist_cutoff=1.0u"nm",
                   center_coords=true, strictness=:nowarn)
    coords = probe.coords
    lo = reduce((a, b) -> min.(a, b), coords)
    hi = reduce((a, b) -> max.(a, b), coords)
    extent = maximum(ustrip.(u"nm", hi .- lo))          # nm
    cutoff = max(5.0, 2.0 * extent)                      # comfortably > any pair
    box    = 4.0 * cutoff                                # cutoff < box/2 by 2x
    return box * u"nm", cutoff * u"nm"
end

function compute_energy(pdb_path::String, ff::MolecularForceField, solvent::Symbol)
    box, cutoff = box_and_cutoff(pdb_path, ff)
    sys = System(pdb_path, ff;
                 boundary=CubicBoundary(box),
                 implicit_solvent=solvent,
                 nonbonded_method=:none,
                 dist_cutoff=cutoff,
                 center_coords=true,
                 strictness=:nowarn)

    sils = sys.specific_inter_lists
    pins = sys.pairwise_inters
    gens = sys.general_inters

    # Molly's setup.jl builds these in a fixed order: bonds, angles, proper
    # torsions, improper torsions. Guard the assumption rather than trust it.
    length(sils) == 4 || error("expected 4 specific interaction lists, got $(length(sils))")
    length(pins) == 2 || error("expected 2 pairwise interactions (LJ, Coulomb), got $(length(pins))")

    neighbors = find_neighbors(sys; n_threads=1)

    result = Dict{String, Any}(
        "file"             => pdb_path,
        "n_atoms"          => length(sys),
        "solvent"          => String(solvent),
        "dist_cutoff_nm"   => ustrip(u"nm", cutoff),
        "box_side_nm"      => ustrip(u"nm", box),
        "bond_stretch"     => subset_energy(sys, neighbors; specific=sils[1:1]),
        "angle_bend"       => subset_energy(sys, neighbors; specific=sils[2:2]),
        "torsion"          => subset_energy(sys, neighbors; specific=sils[3:3]),
        "improper_torsion" => subset_energy(sys, neighbors; specific=sils[4:4]),
        "vdw"              => subset_energy(sys, neighbors; pairwise=pins[1:1]),
        "electrostatic"    => subset_energy(sys, neighbors; pairwise=pins[2:2]),
        "total"            => ustrip(uconvert(KJ, potential_energy(sys, neighbors; n_threads=1))),
        # Counts are diagnostic: the BALL oracle's improper gap showed up as a
        # count mismatch (10 vs 125) long before the energy delta was explained.
        "n_bonds"          => length(sils[1].is),
        "n_angles"         => length(sils[2].is),
        "n_torsions"       => length(sils[3].is),
        "n_impropers"      => length(sils[4].is),
    )

    result["solvation"] = isempty(gens) ? 0.0 :
        subset_energy(sys, neighbors; general=gens[1:1])

    return result
end

function main()
    ff_files = String[]
    pdbs = String[]
    solvent = :obc1

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--ff"
            i += 1; push!(ff_files, ARGS[i])
        elseif a == "--solvent"
            i += 1; solvent = Symbol(ARGS[i])
        else
            push!(pdbs, a)
        end
        i += 1
    end

    if isempty(ff_files) || isempty(pdbs)
        println(stderr, "Usage: molly_energy_oracle.jl --ff amber96.xml " *
                        "[--solvent obc1] structure.pdb [...]")
        exit(1)
    end

    ff = MolecularForceField(ff_files...)

    results = []
    for path in pdbs
        try
            r = compute_energy(path, ff, solvent)
            push!(results, r)
            println(stderr, "OK: $(basename(path)) total=$(round(r["total"], digits=2))")
        catch e
            println(stderr, "FAIL: $(basename(path)): $e")
            push!(results, Dict("file" => path, "error" => string(e)))
        end
    end

    println(JSON.json(results, 2))
end

main()
