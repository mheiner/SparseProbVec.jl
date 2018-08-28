# sparseProbVec.jl

export SparseDirMixPrior, logSDMweights,
    logSDMmarginal, rSparseDirMix;

struct SparseDirMixPrior
  α::Union{Float64, Vector{Float64}}
  β::Float64
end


### Sparse Dirichlet Mixture

"""
    logSDMweights(α, β)

  Calculate the log of mixture weights for the SDM prior.
### Example
```julia
    α = exp.(rand(5))
    β = 2.0
    logSDMweights(α, β)
```
"""
function logSDMweights(α::Vector{Float64}, β::Float64)
  @assert(β > 1.0)
  K = length(α)
  X = reshape(repeat(α, inner=K), (K,K)) + Diagonal(β*ones(K))
  lgX = lgamma.(X)
  lpg = reshape(sum(lgX, dims=2), K)
  lpg_denom = logsumexp(lpg)

  lw = lpg .- lpg_denom
  lw
end

"""
    logSDMmarginal(x, α, β)

  Calculate the log of the SDM prior predictive probability mass function.

"""
function logSDMmarginal(x::Vector{Int}, α::Vector{Float64}, β::Float64)
    lwSDM = logSDMweights(α, β)

    K = length(α)
    A = reshape(repeat(α, inner=K), (K,K)) + Diagonal(β*ones(K))
    AX = A + reshape(repeat(x, inner=K), (K,K))

    lnum = [ lmvbeta(AX[k,:]) for k in 1:K ]
    ldenom = [ lmvbeta(A[k,:]) for k in 1:K ]

    lv = lwSDM .+ lnum .- ldenom

    logsumexp( lv )
end


"""
    rSparseDirMix(α, β[, logscale=FALSE])

  Draw from sparse Dirichlet mixture: p(Θ) ∝ Dir(α)⋅∑Θ^β

"""
function rSparseDirMix(α::Vector{Float64}, β::Float64, logscale=false)
  @assert(β > 1.0)
  K = length(α)
  X = reshape(repeat(α, inner=K), (K,K)) + Diagonal(β*ones(K))
  lgX = lgamma.(X)
  lpg = reshape(sum(lgX, dims=2), K)
  lpg_denom = logsumexp(lpg)

  lw = lpg .- lpg_denom
  z = StatsBase.sample(Weights( exp.(lw) ))

  rDirichlet(X[z,:], logscale)
end
