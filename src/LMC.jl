module LMC

using Distributions
using ForwardDiff
using LinearAlgebra
using Random

function _argcheck(Δt)
    Δt < 0 && throw(DomainError(Δt, "Δt must be nonnegative."))
end

"""
    lmc(logΠ, x, Δt; rng = Random.GLOBAL_RNG)

Langevin Monte Carlo sampler for an arbitrary distribution `exp(logΠ)`.

# Arguments

  - `Δt`: time step of discrete Langevin dynamics.
"""
function lmc(logΠ, x, Δt; rng = Random.GLOBAL_RNG)
    _argcheck(Δt)
    d = length(x)
    dist = MvNormal(zeros(d), 2Δt * I)
    ΔR = rand(rng, dist)
    work = similar(x)
    cfg = ForwardDiff.GradientConfig(logΠ, x)
    xₙ = x + Δt * ForwardDiff.gradient!(work, logΠ, x, cfg) + ΔR
    ΔRᵢ = x - xₙ - Δt * ForwardDiff.gradient!(work, logΠ, xₙ, cfg)
    θ = logΠ(xₙ) - logΠ(x) + logpdf(dist, ΔRᵢ) - logpdf(dist, ΔR)
    if θ ≥ 0 || rand(rng) < exp(θ)
        xₙ
    else
        x
    end
end

lmc(logΠ, x::Real, Δt; rng = Random.GLOBAL_RNG) =
    lmc(logΠ, Float64[x], Δt; rng = rng)[begin]

end