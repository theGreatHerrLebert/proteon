# Xie analytic test-model reference dump — gold-standard fixtures for the Rust port.
# Run: julia --project=tools/oracle/nessie tools/oracle/nessie/xie_dump.jl
# Writes proteon-electrostatics/tests/fixtures/nessie/xie_dump.json
using NESSie
using NESSie.TestModel
import JSON

const OUT = joinpath(@__DIR__, "..", "..", "..",
    "proteon-electrostatics", "tests", "fixtures", "nessie", "xie_dump.json")

# A small spread of off-centre observation points (inside Ω and outside Σ).
obs_points(a) = [
    [0.0, 0.0, 0.0],
    [1.0, 0.5, -0.3],
    [a * 0.6, 0.0, 0.0],          # inside
    [a * 1.5, 0.0, 0.0],          # outside
    [0.0, a * 2.0, a * 1.0],      # outside
]

function model_block(M, sph, len, a)
    m = M(sph, len)
    pts = obs_points(a)
    Dict(
        "rfenergy"    => rfenergy(m),
        "espotential" => [espotential(p, m) for p in pts],
        "molpotential"=> [molpotential(p, m) for p in pts],
    )
end

function case(charges, params, radius, len)
    sph = XieSphere(radius, charges, params; compat = true)
    Dict(
        "input_charges" => [Dict("pos" => c.pos, "val" => c.val) for c in charges],
        "params"        => Dict("eps_omega" => params.εΩ, "eps_sigma" => params.εΣ,
                                "eps_inf" => params.ε∞, "lambda" => params.λ),
        "radius"        => radius,
        "len"           => len,
        "scaled_charges"=> [Dict("pos" => c.pos, "val" => c.val) for c in sph.charges],
        "obs_points"    => obs_points(radius),
        "local"    => model_block(LocalXieModel, sph, len, radius),
        "nonlocal1"=> model_block(NonlocalXieModel1, sph, len, radius),
        "nonlocal2"=> model_block(NonlocalXieModel2, sph, len, radius),
    )
end

opt = Option(2.0, 78.0, 1.8, 20.0)
cases = [
    # single central charge (reduces to Born-like)
    case([Charge([0.0, 0.0, 0.0], 1.0)], opt, 5.0, 30),
    # two charges (off-centre, scaled)
    case([Charge([1.0, 0.0, 0.0], 1.0), Charge([-0.5, 0.5, 0.0], -1.0)], opt, 5.0, 30),
    # three charges, different params
    case([Charge([0.3, 0.0, 0.0], 1.0), Charge([0.0, 0.4, 0.0], 0.5),
          Charge([0.0, 0.0, -0.6], -0.8)], Option(4.0, 80.0, 2.0, 15.0), 6.0, 30),
]

open(OUT, "w") do io
    JSON.print(io, Dict("nessie_version" => string(pkgversion(NESSie)), "cases" => cases), 2)
end
println("wrote ", OUT, " (", length(cases), " cases)")
