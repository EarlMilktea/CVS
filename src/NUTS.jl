module NUTS

using ForwardDiff
using Random
using LinearAlgebra

function _sqnorm(x)
    sum(xi -> xi^2, x)
end

function _lf_kernel(logΠ, Δt, q, p, work, cfg)
    q += p * Δt / 2
    p += ForwardDiff.gradient!(work, logΠ, q, cfg) * Δt
    q += p * Δt / 2
    q, p
end

mutable struct _SubTree
    q₊::Vector{Float64}
    p₊::Vector{Float64}
    q₋::Vector{Float64}
    p₋::Vector{Float64}
    w::Int64
    pred::Bool
    q::Vector{Float64}
    p::Vector{Float64}
end

function _mergetree(t₊, t₋, rng)
    w = t₊.w + t₋.w
    ans = _SubTree(t₊.q₊, t₊.p₊, t₋.q₋, t₋.p₋, w, t₊.pred && t₋.pred, t₊.q, t₊.p)
    if w ≠ 0 && rand(rng) < t₋.w / w
        ans.q = t₋.q
        ans.p = t₋.p
    end
    ans
end

function _gentree_root(logΠ, Δt, qp, pp, uθ, work, cfg)
    q, p = _lf_kernel(logΠ, Δt, qp, pp, work, cfg)
    w = ifelse(uθ ≤ -_sqnorm(p) / 2 + logΠ(q), 1, 0)
    _SubTree(q, p, q, p, w, true, q, p)
end

function _gentree(logΠ, Δt, qp, pp, uθ, l, work, cfg, rng)
    if l == 1
        _gentree_root(logΠ, Δt, qp, pp, uθ, work, cfg)
    else
        t1 = _gentree(logΠ, Δt, qp, pp, uθ, l ÷ 2, work, cfg, rng)
        t = if Δt > 0
            t2 = _gentree(logΠ, Δt, t1.q₊, t1.p₊, uθ, l ÷ 2, work, cfg, rng)
            _mergetree(t2, t1, rng)
        else
            t2 = _gentree(logΠ, Δt, t1.q₋, t1.p₋, uθ, l ÷ 2, work, cfg, rng)
            _mergetree(t1, t2, rng)
        end
        let Δq = t.q₊ - t.q₋
            t.pred = t.pred && t.p₊ ⋅ Δq ≥ 0 && t.p₋ ⋅ Δq ≥ 0
        end
        t
    end
end

function _argcheck(Δt)
    Δt < 0 && throw(DomainError(Δt, "Δt must be nonnegative."))
end

"""
    nuts(logΠ, x, Δt; rng = Random.GLOBAL_RNG)

No-U-Turn Sampler for an arbitrary distribution `exp(logΠ)`.

# Arguments

  - `Δt`: time step of the leap frog method.
"""
function nuts(logΠ, x, Δt; rng = Random.GLOBAL_RNG)
    _argcheck(Δt)
    q = copy(x)
    p = randn(rng, length(q))
    uθ = -_sqnorm(p) / 2 + logΠ(q) + log(rand(rng))
    t = _SubTree(q, p, q, p, 1, true, q, p)
    work = similar(q)
    cfg = ForwardDiff.GradientConfig(logΠ, q)
    let l = 1
        while true
            if rand(rng, (-1, 1)) == 1
                tn = _gentree(logΠ, Δt, t.q₊, t.p₊, uθ, l, work, cfg, rng)
                tn.pred || break
                t = _mergetree(tn, t, rng)
            else
                tn = _gentree(logΠ, -Δt, t.q₋, t.p₋, uθ, l, work, cfg, rng)
                tn.pred || break
                t = _mergetree(t, tn, rng)
            end
            let Δq = t.q₊ - t.q₋
                if t.p₊ ⋅ Δq < 0 || t.p₋ ⋅ Δq < 0
                    break
                end
            end
            l *= 2
        end
    end
    t.q
end

nuts(logΠ, x::Real, Δt; rng = Random.GLOBAL_RNG) =
    nuts(logΠ, Float64[x], Δt; rng = rng)[begin]
end