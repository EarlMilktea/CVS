module CVS

include("LMC.jl")
using .LMC
lmc = LMC.lmc
export lmc

include("HMC.jl")
using .HMC
hmc = HMC.hmc
export hmc

include("NUTS.jl")
using .NUTS
nuts = NUTS.nuts
export nuts

end
