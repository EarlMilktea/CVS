module LMC

using Distributions
using ForwardDiff
using LinearAlgebra
using Random

function _argcheck(Δt)
    Δt < 0 && throw(DomainError(Δt, "Δt must be nonnegative."))
end

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

lmc(logΠ, x::U, Δt; rng = Random.GLOBAL_RNG) where {U<:Real} =
    lmc(logΠ, U[x], Δt; rng = rng)[begin]

end