module HMC

using ForwardDiff
using Random

function _sqnorm(x)
    sum(xi -> xi^2, x)
end

function _hmc_kernel(logΠ, Δt, nstep, q, p)
    work = similar(q)
    for _ ∈ 1:nstep
        q += p * Δt / 2
        p += ForwardDiff.gradient!(work, logΠ, q) * Δt
        q += p * Δt / 2
    end
    q, p
end

function _hmc_kernel(logΠ, Δt, nstep, q::Real, p::Real)
    for _ ∈ 1:nstep
        q += p * Δt / 2
        p += ForwardDiff.derivative(logΠ, q) * Δt
        q += p * Δt / 2
    end
    q, p
end

function _argcheck(T, Δt)
    T < 0 && throw(DomainError("T must be nonnegative."))
    Δt < 0 && throw(DomainError("Δt must be nonnegative."))
end

"""
    hmc(logΠ, x, T; Δt = 0.1, rng = Random.GLOBAL_RNG)

Hamiltonian Monte Carlo sampler for an arbitrary distribution `exp(logΠ)`.

# Arguments

  - `T`: total integration time.
  - `Δt = 0.1`: time step of the leap frog method.
"""
function hmc(logΠ, x, T; Δt = 0.1, rng = Random.GLOBAL_RNG)
    _argcheck(T, Δt)
    q = copy(x)
    p = randn(rng, length(q))
    θ = _sqnorm(p) / 2 - logΠ(q)
    q, p = _hmc_kernel(logΠ, Δt, ceil(T / Δt), q, p)
    θ -= _sqnorm(p) / 2 - logΠ(q)
    # Just discard p
    if θ ≥ 0 || rand(rng) < exp(θ)
        q
    else
        x
    end
end

function hmc(logΠ, x::Real, T; Δt = 0.1, rng = Random.GLOBAL_RNG)
    _argcheck(T, Δt)
    q = x
    p = randn(rng)
    θ = _sqnorm(p) / 2 - logΠ(q)
    q, p = _hmc_kernel(logΠ, Δt, ceil(T / Δt), q, p)
    θ -= _sqnorm(p) / 2 - logΠ(q)
    # Just discard p
    if θ ≥ 0 || rand(rng) < exp(θ)
        q
    else
        x
    end
end

end