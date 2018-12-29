using Example
if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# @test hello("Julia") == "Hello, Julia"
# @test domath(2.0) ≈ 7.0

### SDM
K = 5
α = ones(K)
β = 100.0
d = SparseDirMix(α, β)
rand(d)

### SBM
a, b = shape_Dir2GenDir(α)

nsim = 10e3
xD = hcat([rDirichlet(α) for i = 1:nsim ]...)
xGD = hcat([rGenDirichlet(a,b)[1] for i = 1:nsim ]...)

using Statistics
sum(abs.(mean(xD, dims=2) .- mean(xGD, dims=2)))

η = 1.0e3
π_small = 0.5
π_large = 0.1
γ = 1.5
δ = 1.5
p = SBMprior(K, η, π_small, π_large, γ, δ)

rand(p)

x = [0, 0, 2, 1, 0]

pp = SBM_multinom_post(p, x)

rand(pp)
