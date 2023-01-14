module LMC

using Random
using LinearAlgebra
using Distributions
using ForwardDiff

function _argcheck(Δt)
    Δt < 0 && throw(DomainError(Δt, "Δt must be nonnegative."))
end

function lmc(logΠ, x, Δt; rng = Random.GLOBAL_RNG)
    _argcheck(Δt)
    d = length(x)
    dist = MvNormal(2Δt * Matrix{Float64}(I, d, d))
    ΔR = rand(rng, dist)
    xₙ = x + Δt * ForwardDiff.gradient(logΠ, x) + ΔR
    ΔRᵢ = x - xₙ - Δt * ForwardDiff.gradient(logΠ, xₙ)
    θ = logΠ(xₙ) - logΠ(x) + logpdf(dist, ΔRᵢ) - logpdf(dist, ΔR)
    if θ ≥ 0 || rand(rng) < exp(θ)
        xₙ
    else
        x
    end
end

function lmc(logΠ, x::Real, Δt; rng = Random.GLOBAL_RNG)
    _argcheck(Δt)
    dist = Normal(0, 2Δt)
    ΔR = rand(rng, dist)
    xₙ = x + Δt * ForwardDiff.derivative(logΠ, x) + ΔR
    ΔRᵢ = x - xₙ - Δt * ForwardDiff.derivative(logΠ, xₙ)
    θ = logΠ(xₙ) - logΠ(x) + logpdf(dist, ΔRᵢ) - logpdf(dist, ΔR)
    if θ ≥ 0 || rand(rng) < exp(θ)
        xₙ
    else
        x
    end
end

end