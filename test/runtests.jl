using CVS
using LinearAlgebra
using Random
using Statistics
using Test

function _logΠ_gauss(x)
    -sum(xi -> xi^2, x) / 2
end

function _test_sampler(kernel, d, ns; ninit = 10000, atol = 0.025)
    x = d ≠ 1 ? zeros(d) : 0.0
    for _ ∈ 1:ninit
        x = kernel(x)
    end
    work = Matrix{Float64}(undef, d, ns)
    for i ∈ 1:ns
        x = kernel(x)
        work[:, i] .= x
    end
    m = mean(work; dims = 2) |> vec
    v = cov(work; dims = 2)
    isapprox(m, zeros(d); atol = atol) && isapprox(v, I; atol = atol)
end

@testset "LMC" begin
    let rng = MersenneTwister(0)
        @test _test_sampler(x -> lmc(_logΠ_gauss, x, 0.1; rng = rng), 1, 1000000)
        @test _test_sampler(x -> lmc(_logΠ_gauss, x, 0.1; rng = rng), 2, 1000000)
        @test _test_sampler(x -> lmc(_logΠ_gauss, x, 0.1; rng = rng), 5, 1000000)
    end
end

@testset "HMC" begin
    let rng = MersenneTwister(0)
        @test _test_sampler(x -> hmc(_logΠ_gauss, x, 0.1, 1; rng = rng), 1, 500000)
        @test _test_sampler(x -> hmc(_logΠ_gauss, x, 0.1, 1; rng = rng), 2, 500000)
        @test _test_sampler(x -> hmc(_logΠ_gauss, x, 0.1, 1; rng = rng), 5, 500000)
    end
end

@testset "NUTS" begin
    let rng = MersenneTwister(0)
        @test _test_sampler(x -> nuts(_logΠ_gauss, x, 0.1; rng = rng), 1, 300000)
        @test _test_sampler(x -> nuts(_logΠ_gauss, x, 0.1; rng = rng), 2, 300000)
        @test _test_sampler(x -> nuts(_logΠ_gauss, x, 0.1; rng = rng), 5, 300000)
    end
end