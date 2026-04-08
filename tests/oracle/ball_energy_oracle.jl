#!/usr/bin/env julia
# BALL Julia oracle: compute AMBER96 energy components on a PDB structure.
# Outputs JSON with energy breakdown for comparison with ferritin.
#
# Usage: julia --project=BALL_PATH ball_energy_oracle.jl structure.pdb

using BiochemicalAlgorithms
using JSON

function compute_energy(pdb_path::String)
    # Load structure
    sys = load_pdb(pdb_path)

    # Build bonds and normalize names
    fdb = FragmentDB()
    normalize_names!(sys, fdb)
    reconstruct_fragments!(sys, fdb)
    build_bonds!(sys, fdb)

    # Set up AMBER96 force field
    ff = AmberFF(sys, amberversion=Symbol("amber96"))

    # Compute energy
    energy = compute_energy!(ff)

    # Extract components
    result = Dict{String, Any}(
        "file" => pdb_path,
        "total" => energy,
    )

    # Get individual component energies
    for (name, value) in ff.energy
        result[name] = value
    end

    # Also get component details from each component
    for comp in ff.components
        comp_energy = compute_energy!(comp)
        for (name, value) in comp.energy
            result["$(comp.name)::$name"] = value
        end
    end

    return result
end

if length(ARGS) < 1
    println(stderr, "Usage: julia ball_energy_oracle.jl structure.pdb [structure2.pdb ...]")
    exit(1)
end

results = []
for path in ARGS
    try
        r = compute_energy(path)
        push!(results, r)
        println(stderr, "OK: $(basename(path)) total=$(round(r["total"], digits=2))")
    catch e
        println(stderr, "FAIL: $(basename(path)): $e")
        push!(results, Dict("file" => path, "error" => string(e)))
    end
end

println(JSON.json(results, 2))
