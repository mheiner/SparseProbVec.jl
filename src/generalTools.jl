# generalTools.jl

export logsumexp, rDirichlet, lmvbeta;

"""
    logsumexp(x[, usemax])

Computes `log(sum(exp(x)))` in a stable manner.

### Example
```julia
  x = rand(5)
  logsumexp(x)
  log(sum(exp.(x)))
```
"""
function logsumexp(x::Array{Float64}, usemax::Bool=true)
  if usemax
    m = maximum(x)
  else
    m = minimum(x)
  end

  m + log(sum(exp.(x .- m)))
end

"""
    logsumexp(x, region[, usemax])

Computes `log(sum(exp(x)))` in a stable manner along dimensions specified.

### Example
```julia
  x = reshape(collect(1:24)*1.0, (2,3,4))
  logsumexp(x, 2)
```
"""
function logsumexp(x::Array{Float64}, region, usemax::Bool=true)
  if usemax
    ms = maximum(x, dims=region)
  else
    ms = minimum(x, dims=region)
  end
  bc_xminusms = broadcast(-, x, ms)

  expxx = exp.(bc_xminusms)
  sumexpxx = sum(expxx, dims=region)

  log.(sumexpxx) .+ ms
end


"""
    rDirichlet(α[, logscale])

  Single draw from Dirichlet distribution, option for log scale.

  ### Example
  ```julia
  rDirichlet(ones(5),true)
  ```
"""
function rDirichlet(α::Array{Float64, 1}, logscale::Bool=false)
  @assert( all(α .> 0.0) )

  k = length(α)
  xx = Vector{Float64}(undef, k) # allows changes to elements
  s = 0.0

  if logscale
    for i in 1:k
      xx[i] = log(rand(Gamma(α[i], 1.0)))
    end
    s = logsumexp(xx)
    out = xx .- s

  else
    for i in 1:k
      xx[i] = rand(Gamma(α[i], 1.0))
      s += xx[i]
    end
    out = xx / s

  end

out
end


"""
    lmvbeta(x)

Computes the natural log of ``∏(Γ(x)) / Γ(sum(x))``.

### Example
```julia
  x = rand(5)
  lmvbeta(x)
```
"""
function lmvbeta(x::Array{Float64})
    sum(lgamma.(x)) - lgamma(sum(x))
end
