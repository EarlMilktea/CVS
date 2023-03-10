module HMC

using ForwardDiff
using Random

function _sqnorm(x)
    sum(xi -> xi^2, x)
end

function _hmc_kernel!(logΠ, Δt, nstep, q, p)
    work = similar(q)
    cfg = ForwardDiff.GradientConfig(logΠ, q)
    for _ ∈ 1:nstep
        q .+= p * Δt / 2
        p .+= ForwardDiff.gradient!(work, logΠ, q, cfg) * Δt
        q .+= p * Δt / 2
    end
end

function _argcheck(T, Δt)
    T < 0 && throw(DomainError("T must be nonnegative."))
    Δt < 0 && throw(DomainError("Δt must be nonnegative."))
end

function hmc(logΠ, x, Δt, T; rng = Random.GLOBAL_RNG)
    _argcheck(T, Δt)
    q = copy(x)
    p = randn(rng, length(q))
    θ = _sqnorm(p) / 2 - logΠ(q)
    _hmc_kernel!(logΠ, Δt, ceil(T / Δt), q, p)
    θ -= _sqnorm(p) / 2 - logΠ(q)
    # Just discard p
    if θ ≥ 0 || rand(rng) < exp(θ)
        q
    else
        x
    end
end

hmc(logΠ, x::U, Δt, T; rng = Random.GLOBAL_RNG) where {U<:Real} =
    hmc(logΠ, U[x], Δt, T; rng = rng)[begin]

end