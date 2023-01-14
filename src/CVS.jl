module CVS

include("LMC.jl")
using .LMC
lmc = LMC.lmc
export lmc

include("HMC.jl")
using .HMC
hmc = HMC.hmc
export lmc, hmc

end
