# NESSie.jl oracle harness for proteon-electrostatics.
#
# Offline fixture generator (TO_ELECTROSTATICS.md §3, phase P0). Dumps deterministic
# JSON the Rust tests gate against. NOT a proteon build/runtime dependency — it runs
# in its own pinned Julia project (`NESSie@1.5`) and the small fixtures are checked
# into the proteon-electrostatics test tree.
#
# Six dumps (§3):
#   collocation_dump  L1  Rjasanow analytic Laplace collocation matrices (SL + DL)
#   yukawa_dump       L2  Radon regular-Yukawa collocation matrices (SL + DL)
#   solve_dump        L3  Cauchy data u,q,[w] + umol,qmol  (:gmres and :blas)
#   system_dump       L3  kernel building-block matrices (PARTIAL assembly oracle —
#                         the assembled 2/3-block system + RHS + dielectric factors
#                         + jump terms land at P0.5, once §1b pins the formulation)
#   post_dump         L4  rfenergy + espotential sampled across domains :Ω/:Σ/:Γ
#   analytic_dump     L4  closed-form Born/Xie energies — independent ground truth
#
# Plus: the canonical convergence mesh is proteon's `icosphere` emitted as OFF (so
# both sides consume the *same* mesh); this harness reads it via `readoff`.
#
# Usage:
#   julia --project=tools/oracle/nessie tools/oracle/nessie/harness.jl <out_dir>
# (run `Pkg.instantiate()` in this project once first — see README.md)

module NESSieOracle

using JSON
using NESSie
using NESSie: ε0
using NESSie.Format: readoff, readpqr
using NESSie.BEM: solve
using NESSie.Rjasanow: laplacecoll!
using NESSie.Radon: regularyukawacoll!
using Pkg

const PTYPES = (SingleLayer, DoubleLayer)
const PTYPE_NAME = Dict(SingleLayer => "single", DoubleLayer => "double")

# --- provenance ----------------------------------------------------------------

"Version + tree-hash stamp pinned into every dump for reproducibility. The
 tree_hash is the real pin (a registered version can be re-tagged); commit the
 generated Manifest.toml alongside for a fully reproducible resolve."
function provenance()
    nessie_ver, nessie_hash = try
        info = Pkg.dependencies()[Base.UUID("f00e83c4-bd01-11e9-272c-ff1b96fb444d")]
        (string(info.version), string(something(info.tree_hash, "unknown")))
    catch
        ("unknown", "unknown")
    end
    Dict(
        "nessie_version"   => nessie_ver,
        "nessie_tree_hash" => nessie_hash,
        "julia_version"    => string(VERSION),
    )
end

"Dielectric/nonlocal parameters as plain fields (so fixtures are self-contained)."
params_json(p) = Dict("eps_omega" => p.εΩ, "eps_sigma" => p.εΣ,
                      "eps_inf" => p.ε∞, "lambda" => p.λ)

"Point charges as plain fields."
charges_json(model) = [Dict("pos" => c.pos, "val" => c.val) for c in model.charges]

# --- model loading + geometry serialization ------------------------------------

"Load a NESSie model from an OFF mesh + PQR charges, with explicit parameters."
function load_model(off_path::AbstractString, pqr_path::AbstractString;
                    εΩ = 1.0, εΣ = 78.0, ε∞ = 1.8, λ = 20.0)
    model = readoff(off_path)
    model.charges = readpqr(pqr_path)
    model.params.εΩ = εΩ
    model.params.εΣ = εΣ
    model.params.ε∞ = ε∞
    model.params.λ  = λ
    model
end

"Triangle elements → plain arrays (so the Rust side is fully determined)."
function elements_json(model)
    [Dict(
        "v1"     => e.v1,  "v2" => e.v2, "v3" => e.v3,
        "center" => e.center, "normal" => e.normal, "area" => e.area,
    ) for e in model.elements]
end

obspoints(model) = [e.center for e in model.elements]   # collocation points Ξ

"Collocation matrix (|Ξ| × |elements|) for a kernel-matrix `!` routine."
function _coll_matrix(fill!, model)
    elements = model.elements
    Ξ = obspoints(model)
    dest = zeros(Float64, length(Ξ), length(elements))
    fill!(dest)
    [collect(row) for row in eachrow(dest)]   # row-major nested arrays for JSON
end

# --- the six dumps --------------------------------------------------------------

"L1: analytic Laplace single/double-layer collocation matrices."
function collocation_dump(model)
    elements, Ξ = model.elements, obspoints(model)
    mats = Dict(PTYPE_NAME[p] =>
        _coll_matrix(d -> laplacecoll!(p, d, elements, Ξ), model) for p in PTYPES)
    Dict("kind" => "collocation", "provenance" => provenance(),
         "observation_points" => Ξ, "elements" => elements_json(model),
         "matrices" => mats)
end

"L2: regular-Yukawa single/double-layer collocation matrices at a given yukawa."
function yukawa_dump(model, yukawa::Float64)
    elements, Ξ = model.elements, obspoints(model)
    mats = Dict(PTYPE_NAME[p] =>
        _coll_matrix(d -> regularyukawacoll!(p, d, elements, Ξ, yukawa), model) for p in PTYPES)
    Dict("kind" => "yukawa", "provenance" => provenance(), "yukawa" => yukawa,
         "observation_points" => Ξ, "elements" => elements_json(model),
         "matrices" => mats)
end

"L3: Cauchy data for a locality + method (`:gmres` / `:blas`). Self-contained:
 carries the ordered elements, observation points, charges, and params so the Rust
 side reproduces nothing implicitly and reordered geometry is detectable."
function solve_dump(model, locality::Type{<:LocalityType}, method::Symbol)
    bem = solve(locality, model; method = method)
    out = Dict("kind" => "solve", "provenance" => provenance(),
               "locality" => string(locality), "method" => string(method),
               "num_elements" => length(model.elements),
               "elements" => elements_json(model),
               "observation_points" => obspoints(model),
               "charges" => charges_json(model), "params" => params_json(model.params),
               "u" => collect(bem.u), "q" => collect(bem.q),
               "umol" => collect(bem.umol), "qmol" => collect(bem.qmol))
    locality === NonlocalES && (out["w"] = collect(bem.w))
    out
end

"L3 assembly oracle (PARTIAL): the kernel building-block matrices NESSie's `:blas`
 path assembles from. It does NOT yet emit the assembled 2/3-block system, RHS,
 dielectric factors, or jump terms — those are pinned by the §1b formulation spec
 (P0.5). Until then this gates kernel-block parity, not full assembly; finish it
 once the blocks are written down."
function assembly_kernels_dump(model, locality::Type{<:LocalityType})
    yuk = NESSie.yukawa(model.params)
    out = Dict("kind" => "assembly_kernels", "provenance" => provenance(),
               "locality" => string(locality), "yukawa" => yuk,
               "params" => params_json(model.params),
               "elements" => elements_json(model),
               "observation_points" => obspoints(model),
               "laplace" => collocation_dump(model)["matrices"])
    locality === NonlocalES && (out["regular_yukawa"] = yukawa_dump(model, yuk)["matrices"])
    out["_todo"] = "P0.5: emit the assembled block system + RHS once formulation is pinned"
    out
end

"L4: reaction-field energy + potentials, each domain sampled at points that
 actually lie in it. Centroids lie on Γ, so Ω/Σ must use distinct interior/exterior
 sets — otherwise the interior/exterior formulas are evaluated on-surface. The
 default sets assume an origin-centred surface (Born/Xie); pass explicit sets for
 a general molecule."
function post_dump(model, locality::Type{<:LocalityType};
                   omega = nothing, sigma = nothing, gamma = nothing)
    bem = solve(locality, model)
    surf = obspoints(model)
    Ω = omega === nothing ? [0.5 .* c for c in surf] : omega   # pulled inward
    Σ = sigma === nothing ? [1.5 .* c for c in surf] : sigma   # pushed outward
    Γ = gamma === nothing ? surf : gamma                       # on the surface
    samp = Dict("Ω" => Ω, "Σ" => Σ, "Γ" => Γ)
    pot = Dict(string(d) => espotential(d, samp[string(d)], bem) for d in (:Ω, :Σ, :Γ))
    Dict("kind" => "post", "provenance" => provenance(),
         "locality" => string(locality), "rfenergy" => rfenergy(bem),
         "sample_points" => samp, "espotential" => pot)
end

"L4 independent ground truth: closed-form Born ion energies (local + nonlocal).
 Each entry carries the ion's charge, radius, and params so the fixture is
 self-contained (no reliance on NESSie's built-in `bornion` table)."
function analytic_dump(; ions = ["li", "na", "k", "rb", "cs", "mg", "ca", "sr", "ba"])
    born = Dict(name => begin
            ion = bornion(name)
            Dict("charge" => ion.charge, "radius" => ion.radius,
                 "params" => params_json(ion.params),
                 "local"    => rfenergy(LocalES, ion),
                 "nonlocal" => rfenergy(NonlocalES, ion))
        end for name in ions)
    Dict("kind" => "analytic", "provenance" => provenance(), "born" => born)
end

# --- driver ---------------------------------------------------------------------

writejson(dir, name, obj) = open(joinpath(dir, name), "w") do io
    JSON.print(io, obj)
end

"""
Generate the P0 fixture set from NESSie's bundled Born data into `out_dir`.

This is a starter corpus (one Born ion). Extend with Xie spheres, an
analytic-sphere ladder (proteon `icosphere` → OFF), and one small protein as the
layers come online. Keep dense convergence runs regenerate-on-demand, not checked
in (plan Q5).
"""
function generate(out_dir::AbstractString)
    mkpath(out_dir)
    data = NESSie.nessie_data_path("born")            # NESSie's bundled fixtures
    model = load_model(joinpath(data, "na.off"), joinpath(data, "na.pqr"))
    yuk = NESSie.yukawa(model.params)

    writejson(out_dir, "collocation_na.json", collocation_dump(model))
    writejson(out_dir, "yukawa_na.json",      yukawa_dump(model, yuk))
    writejson(out_dir, "assembly_kernels_local_na.json",    assembly_kernels_dump(model, LocalES))
    writejson(out_dir, "assembly_kernels_nonlocal_na.json", assembly_kernels_dump(model, NonlocalES))
    for loc in (LocalES, NonlocalES), m in (:gmres, :blas)
        writejson(out_dir, "solve_$(loc)_$(m)_na.json", solve_dump(model, loc, m))
    end
    writejson(out_dir, "post_local_na.json",    post_dump(model, LocalES))
    writejson(out_dir, "post_nonlocal_na.json", post_dump(model, NonlocalES))
    writejson(out_dir, "analytic.json",         analytic_dump())
    @info "Wrote NESSie oracle fixtures" out_dir
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) >= 1 ? ARGS[1] : "fixtures"
    NESSieOracle.generate(out)
end
