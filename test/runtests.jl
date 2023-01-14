using CVS
using Random
using Test
using Statistics
using LinearAlgebra

function _logΠ_gauss(x)
    -sum(xi -> xi^2, x) / 2
end

function _test_lmc_scalar(ns)
    rng = MersenneTwister(0)
    x = 0
    for _ ∈ 1:10000
        x = lmc(_logΠ_gauss, x, 0.1; rng = rng)
    end
    work = Vector{Float64}(undef, ns)
    for i ∈ 1:ns
        x = lmc(_logΠ_gauss, x, 0.1; rng = rng)
        work[i] = x
    end
    m = mean(work)
    v = var(work)
    @show m, v
    isapprox(m, 0; atol = 0.01) && isapprox(v, 1; atol = 0.01)
end

function _test_lmc(d, ns)
    rng = MersenneTwister(0)
    x = zeros(d)
    for _ ∈ 1:10000
        x = lmc(_logΠ_gauss, x, 0.1; rng = rng)
    end
    work = Matrix{Float64}(undef, d, ns)
    for i ∈ 1:ns
        x = lmc(_logΠ_gauss, x, 0.1; rng = rng)
        work[:, i] = x
    end
    m = mean(work; dims = 2) |> vec
    v = cov(work; dims = 2)
    @show m, v
    isapprox(m, zeros(d); atol = 0.01) && isapprox(v, I; atol = 0.01)
end

@testset "LMC" begin
    @test _test_lmc_scalar(5000000)
    @test _test_lmc(1, 5000000)
    @test _test_lmc(2, 5000000)
    @test _test_lmc(5, 5000000)
end
