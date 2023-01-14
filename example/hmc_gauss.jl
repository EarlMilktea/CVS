using CVS
using LinearAlgebra

function logΠ(x)
    -sum(xi -> xi^2, x) / 2
end

let d = 2, n_sample = 100000, T = 5
    x = zeros(d)

    for _ ∈ 1:n_sample
        x = hmc(logΠ, x, T)
        join(x, " ") |> println
    end
end
