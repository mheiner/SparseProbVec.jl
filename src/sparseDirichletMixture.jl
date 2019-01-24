# sparseDirichletMixture.jl

export SparseDirMix, logSDMweights,
    logMarginal, rand;

struct SparseDirMix
  α::Union{Float64, Vector{Float64}}
  β::Float64
  SparseDirMix(α, β) = all(α .> 0.0) && β > 1.0 ? new(α, β) :
    error("Invalid parameter values.")
end


"""
    logSDMweights(d::SparseDirMix)

  Calculate the log of mixture weights for the SDM prior.
### Example
```julia
    α = exp.(rand(5))
    β = 2.0
    d = SparseDirMix(α, β)
    logSDMweights(d)
```
"""
function logSDMweights(d::SparseDirMix)
  K = length(d.α)
  X = reshape(repeat(d.α, inner=K), (K,K)) + Diagonal(d.β*ones(K))
  lgX = lgamma.(X)
  lpg = reshape(sum(lgX, dims=2), K)
  lpg_denom = logsumexp(lpg)

  lw = lpg .- lpg_denom
  lw
end

"""
    logMarginal(x, d::SparseDirMix)

Calculate the log of the SDM-multinomial compound (marginal) probability mass function.

"""
function logMarginal(prior::SparseDirMix, x::Vector{Int})
    lwSDM = logSDMweights(prior)

    K = length(prior.α)
    A = reshape(repeat(prior.α, inner=K), (K,K)) + Diagonal(prior.β*ones(K))
    AX = A + reshape(repeat(x, inner=K), (K,K))

    lnum = [ lmvbeta(AX[k,:]) for k in 1:K ]
    ldenom = [ lmvbeta(A[k,:]) for k in 1:K ]

    lv = lwSDM .+ lnum .- ldenom

    logsumexp( lv )
end


"""
    rand(d::SparseDirMix[, logscale=FALSE])

  Draw from sparse Dirichlet mixture: p(Θ) ∝ Dir(α)⋅∑Θ^β

"""
function Base.rand(d::SparseDirMix; logout::Bool=false)
  K = length(d.α)
  X = reshape(repeat(d.α, inner=K), (K,K)) + Diagonal(d.β*ones(K))
  lgX = lgamma.(X)
  lpg = reshape(sum(lgX, dims=2), K)
  lpg_denom = logsumexp(lpg)

  lw = lpg .- lpg_denom
  z = StatsBase.sample(Weights( exp.(lw) ))

  rDirichlet(X[z,:], logout=logout)
end
