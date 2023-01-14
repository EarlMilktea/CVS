using CVS
using LinearAlgebra

function logΠ(x)
    -sum(xi -> xi^2, x) / 2
end

let d = 2, n_sample = 100000, Δt = 0.1
    x = zeros(d)

    for _ ∈ 1:n_sample
        x = lmc(logΠ, x, Δt)
        join(x, " ") |> println
    end
end
