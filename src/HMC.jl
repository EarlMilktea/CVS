using ForwardDiff
using LinearAlgebra
using Random

const _N_INT_MIN = 10::Integer
@assert _N_INT_MIN ≥ 1

function _hmc_kernel!(logΠ, Δt, nstep, q, p)
    work = similar(q)
    for _ ∈ 1:nstep
        q += p * Δt / 2
        p += ForwardDiff.gradient!(work, logΠ, q) * Δt
        q += p * Δt / 2
    end
end

function hmc(logΠ, x, T; Δt = 0.1, rng = Random.GLOBAL_RNG)
    T < 0 && throw(DomainError("T must be nonnegative."))
    Δt < 0 && throw(DomainError("Δt must be nonnegative."))
    T < _N_INT_MIN * Δt && throw(ArgumentError("T must be larger than $(_N_INT_MIN) * Δt."))
    q = copy(x)
    p = randn!(rng, length(q))
    θ = LinearAlgebra.norm2(p) / 2 - logΠ(q)
    _hmc_kernel!(logΠ, Δt, ceil(T / Δt), q, p)
    θ -= LinearAlgebra.norm2(p) / 2 - logΠ(q)
    # Just discard p
    if θ ≥ 0 || rand(rng) < exp(θ)
        q
    else
        x
    end
end