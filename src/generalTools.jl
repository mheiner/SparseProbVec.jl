# generalTools.jl

export logsumexp, rDirichlet, rGenDirichlet, lmvbeta, shape_Dir2GenDir;

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

  return m + log(sum(exp.(x .- m)))
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

  return log.(sumexpxx) .+ ms
end


"""
    rDirichlet(α[; logout=false, logcompute=true])

  Single draw from Dirichlet distribution, option for log scale, computation on log scale.

  ### Example
  ```julia
  rDirichlet(ones(5),true)
  ```
"""
function rDirichlet(α::Array{Float64, 1}; logout::Bool=false, logcompute::Bool=true)
  all(α .> 0.0) || error("All shape parameters must be positive.")

  k = length(α)
  xx = Vector{Float64}(undef, k) # allows changes to elements
  s = 0.0

  if logcompute
    for i in 1:k
      xx[i] = log(rand(Gamma(α[i], 1.0)))
    end
    s = logsumexp(xx)
    out = xx .- s
    if !logout
        out = exp.(out)
    end
  else
    for i in 1:k
      xx[i] = rand(Gamma(α[i], 1.0))
      s += xx[i]
    end
    out = xx / s
    if logout
        out = log.(out)
    end
  end

  return out
end


"""
    rGenDirichlet(a, b[, logout])

  Single draw from the generalized Dirichlet distribution (Connor and Mosimann '69), option for log scale.

  ### Example
  ```julia
  a = ones(5)
  b = ones(5)
  rGenDirichlet(a, b)
  ```
"""
function rGenDirichlet(a::Vector{Float64}, b::Vector{Float64}; logout::Bool=false)
    n = length(a)
    K = n + 1
    length(b) == n || error("Dimension mismatch between a and b.")
    all(a .> 0.0) || error("All elements of a must be positive.")
    all(b .> 0.0) || error("All elements of b must be positive.")

    lz = Vector{Float64}(undef, n)
    loneminusz = Vector{Float64}(undef, n)
    for i = 1:n
        lx1 = log( rand( Distributions.Gamma( a[i] ) ) )
        lx2 = log( rand( Distributions.Gamma( b[i] ) ) )
        lxm = max(lx1, lx2)
        lxsum = lxm + log( exp(lx1 - lxm) + exp(lx2 - lxm) ) # logsumexp
        lz[i] = lx1 - lxsum
        loneminusz[i] = lx2 - lxsum
    end

    ## break the Stick
    lw = Vector{Float64}(undef, K)
    lwhatsleft = 0.0

    for i in 1:n
        lw[i] = lz[i] + lwhatsleft
        # lwhatsleft += log( 1.0 - exp(lw[i] - lwhatsleft) ) # logsumexp (not numerically stable)
        lwhatsleft += loneminusz[i]
    end
    lw[K] = copy(lwhatsleft)

    if logout
        out = (lw, lz)
    else
        out = (exp.(lw), exp.(lz))
    end
    return out
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
    return sum(lgamma.(x)) - lgamma(sum(x))
end


"""
    shape_Dir2GenDir(α)

Converts Dirichlet shape parameter vector to set of shape parameters for latent beta variables in the ggeneralized Dirichlet distribution (Connor and Mosimann '69).

### Example
```julia
  α = ones(5)
  shape_Dir2GenDir(α)
```
"""
function shape_Dir2GenDir(α::Vector{Float64})
    all(α .> 0.0) || error("All shape parameters must be positive.")

    K = length(α)
    n = K - 1
    rcrα = cumsum( α[range(K, step=-1, length=n)] )[range(n, step=-1, length=n)]

    return (α[1:n], rcrα)
end
