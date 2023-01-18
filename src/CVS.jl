module CVS

include("LMC.jl")
using .LMC

"""
    lmc(logΠ, x, Δt; rng = Random.GLOBAL_RNG)

Langevin Monte Carlo sampler for an arbitrary distribution `exp(logΠ)`.

# Arguments

  - `Δt`: time step of discrete Langevin dynamics.
"""
lmc = LMC.lmc
export lmc

include("HMC.jl")
using .HMC

"""
    hmc(logΠ, x, Δt, T; rng = Random.GLOBAL_RNG)

Hamiltonian Monte Carlo sampler for an arbitrary distribution `exp(logΠ)`.

# Arguments

  - `Δt`: time step of the leap frog method.
  - `T`: total integration time.
"""
hmc = HMC.hmc
export hmc

include("NUTS.jl")
using .NUTS

"""
    nuts(logΠ, x, Δt; rng = Random.GLOBAL_RNG)

No-U-Turn Sampler for an arbitrary distribution `exp(logΠ)`.

# Arguments

  - `Δt`: time step of the leap frog method.
"""
nuts = NUTS.nuts
export nuts

end
