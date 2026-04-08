#!/usr/bin/env julia
# BALL Julia oracle: compute AMBER96 energy on RAW PDB (no reconstruction).
# This isolates the energy computation from the preparation pipeline.

using BiochemicalAlgorithms
using JSON

function compute_energy_raw(pdb_path::String)
    sys = load_pdb(pdb_path)

    fdb = FragmentDB()
    normalize_names!(sys, fdb)
    # NO reconstruct_fragments! - use structure as-is
    build_bonds!(sys, fdb)

    ff = AmberFF(sys)

    energy = compute_energy!(ff)

    result = Dict{String, Any}(
        "file" => pdb_path,
        "n_atoms" => length(atoms(sys)),
        "n_bonds" => length(bonds(sys)),
        "total" => energy,
    )

    for comp in ff.components
        compute_energy!(comp)
        for (name, value) in comp.energy
            result["$(comp.name)::$name"] = value
        end
    end

    return result
end

for path in ARGS
    try
        r = compute_energy_raw(path)
        println(stderr, "OK: $(basename(path)) atoms=$(r["n_atoms"]) bonds=$(r["n_bonds"]) total=$(round(r["total"], digits=2))")
        println(JSON.json(r, 2))
    catch e
        println(stderr, "FAIL: $(basename(path)): $e")
    end
end
