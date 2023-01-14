module NUTS

using ForwardDiff
using Random
using LinearAlgebra

function _sqnorm(x)
    sum(xi -> xi^2, x)
end

function _lf_kernel(logΠ, Δt, q, p)
    q += p * Δt / 2
    p += ForwardDiff.gradient(logΠ, q) * Δt
    q += p * Δt / 2
    q, p
end

mutable struct _SubTree{T}
    q₊::T
    p₊::T
    q₋::T
    p₋::T
    w::Int64
    pred::Bool
    q::T
    p::T
end

function _mergetree(t₊::_SubTree{T}, t₋::_SubTree{T}, rng) where {T}
    w = t₊.w + t₋.w
    ans = _SubTree{T}(t₊.q₊, t₊.p₊, t₋.q₋, t₋.p₋, w, t₊.pred && t₋.pred, t₊.q, t₊.p)
    if w ≠ 0 && rand(rng) < t₋.w / w
        ans.q = t₋.q
        ans.p = t₋.p
    end
    ans
end

function _gentree_root(logΠ, Δt, qp, pp, uθ)
    q, p = _lf_kernel(logΠ, Δt, qp, pp)
    w = ifelse(uθ ≤ -_sqnorm(p) / 2 + logΠ(q), 1, 0)
    _SubTree{Vector{Float64}}(q, p, q, p, w, true, q, p)
end

function _gentree(logΠ, Δt, qp, pp, uθ, l, rng)
    if l == 1
        _gentree_root(logΠ, Δt, qp, pp, uθ)
    else
        t1 = _gentree(logΠ, Δt, qp, pp, uθ, l ÷ 2, rng)
        t = if Δt > 0
            t2 = _gentree(logΠ, Δt, t1.q₊, t1.p₊, uθ, l ÷ 2, rng)
            _mergetree(t2, t1, rng)
        else
            t2 = _gentree(logΠ, Δt, t1.q₋, t1.p₋, uθ, l ÷ 2, rng)
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

function nuts(logΠ, x, Δt; rng = Random.GLOBAL_RNG)
    _argcheck(Δt)
    q = copy(x)
    p = randn(rng, length(q))
    uθ = -_sqnorm(p) / 2 + logΠ(q) + log(rand(rng))
    t = _SubTree{Vector{Float64}}(q, p, q, p, 1, true, q, p)
    let l = 1
        while true
            if rand(rng, (-1, 1)) == 1
                tn = _gentree(logΠ, Δt, t.q₊, t.p₊, uθ, l, rng)
                tn.pred || break
                t = _mergetree(tn, t, rng)
            else
                tn = _gentree(logΠ, -Δt, t.q₋, t.p₋, uθ, l, rng)
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
end