# stickBreakingMixture.jl
export SBMprior, SBM_multinom_post, logMarginal, rand, del_correctionSBM;

struct SBMprior
    K::Int
    η::Float64
    π_small::Union{Float64, Vector{Float64}}
    π_large::Union{Float64, Vector{Float64}}
    γ::Union{Float64, Vector{Float64}}
    δ::Union{Float64, Vector{Float64}}
    SBMprior(K, η, π_small, π_large, γ, δ) =
        K > 0 && η > 1.0 &&
        all(π_small .>= 0.0) && all(π_large .>= 0.0) && all(π_small .+ π_large .<= 1.0) &&
        all(γ .> 0.0) && all(δ .> 0.0) &&
        ( (length(γ) == length(δ) == 1) || (length(γ) == length(δ) == (K-1)) ) ?
        new(K, η, π_small, π_large, γ, δ) : error("Invalid parameter values.")
end

struct SBM_multinom_post
    prior::SBMprior
    x::Vector{Int} # vector of count data
    a_small::Vector{Float64}
    b_small::Vector{Float64}
    a_med::Vector{Float64}
    b_med::Vector{Float64}
    a_large::Vector{Float64}
    b_large::Vector{Float64}
    lW::Matrix{Float64}
    lmarg::Vector{Float64}
end

function SBM_multinom_post(prior::SBMprior, x::Vector{Int})
    all(x .>= 0) || error("All data counts must be non-negative.")
    length(x) == prior.K || error("Data vector and prior length mismatch.")

    n = prior.K - 1
    rcrx = cumsum( x[range(prior.K, step=-1, length=n)] )[range(n, step=-1, length=n)]  # ∑_{k+1}^K x_k

    ## mixture component 1 update
    a_small = 1.0 .+ x[1:n]
    b_small = prior.η .+ rcrx

    ## mixture component 2 update
    a_med = prior.γ .+ x[1:n]
    b_med = prior.δ .+ rcrx

    ## mixture component 3 update
    a_large = prior.η .+ x[1:n]
    b_large = 1.0 .+ rcrx

    ## calculate weights
    π_med = 1.0 .- prior.π_small .- prior.π_large

    ## calculate posterior mixture weights
    lgwt1 = log.(prior.π_small) .- lbeta.(1.0, prior.η) .+ lbeta.(a_small, b_small) # small
    lgwt2 = log.(π_med) .- lbeta.(prior.γ, prior.δ) .+ lbeta.(a_med, b_med) # medium
    lgwt3 = log.(prior.π_large) .- lbeta.(prior.η, 1.0) .+ lbeta.(a_large, b_large) # large
    lmarg = [ logsumexp( [lgwt1[i], lgwt2[i], lgwt3[i]] ) for i in 1:n ]

    lW = hcat( lgwt1 .- lmarg, lgwt2 .- lmarg, lgwt3 .- lmarg )

    return SBM_multinom_post( prior, x,
        a_small, b_small,
        a_med, b_med,
        a_large, b_large,
        lW, lmarg )
end


"""
    logMarginal(d::SBM_multinom_post, x::Vector{Int})

Calculate the log of the SBM-multinomial compound (marginal) probability mass function.

"""
function logMarginal(prior::SBMprior, x::Vector{Int})
    d = SBM_multinom_post(prior, x)
    return sum( d.lmarg )
end


function Base.rand(d::SBMprior; logout::Bool=false, zξout::Bool=false)

    π_med = 1.0 .- d.π_small .- d.π_large
    n = d.K - 1

    if length(d.π_small) == 1
        π_small = fill(d.π_small, n)
    else
        π_small = deepcopy(d.π_small)
    end

    if length(d.π_large) == 1
        π_large = fill(d.π_large, n)
    else
        π_large = deepcopy(d.π_large)
    end

    ## draw ξ
    w = hcat(π_small, π_med, π_large)
    ξ = [ StatsBase.sample( Weights( w[i,:] ) ) for i = 1:n ]

    ## allocate generalized Dirichlet parameters
    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)

    if length(d.γ) == 1
        γ = fill(d.γ, n)
    else
        γ = deepcopy(d.γ)
    end

    if length(d.δ) == 1
        δ = fill(d.δ, n)
    else
        δ = deepcopy(d.δ)
    end

    for i in 1:n
        if ξ[i] == 1
            a[i] = 1.0
            b[i] = copy(d.η)
        elseif ξ[i] == 2
            a[i] = copy(γ[i])
            b[i] = copy(δ[i])
        elseif ξ[i] == 3
            a[i] = copy(d.η)
            b[i] = 1.0
        else
            error("Illegal mixture allocation.")
        end
    end

    ## draw lw, lz
    w, z = rGenDirichlet(a, b, logout=logout)

    if zξout
        return (w, z, ξ)
    else
        return w
    end

end

function Base.rand(d::SBM_multinom_post; logout::Bool=false, zξout::Bool=false)

    n = d.prior.K - 1

    ## draw ξ
    ξ = [ StatsBase.sample( Weights( exp.( d.lW[i,:] ) )) for i = 1:n ]

    ## allocate generalized Dirichlet parameters
    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)

    for i in 1:n
        if ξ[i] == 1
            a[i] = copy(d.a_small[i])
            b[i] = copy(d.b_small[i])
        elseif ξ[i] == 2
            a[i] = copy(d.a_med[i])
            b[i] = copy(d.b_med[i])
        elseif ξ[i] == 3
            a[i] = copy(d.a_large[i])
            b[i] = copy(d.b_large[i])
        else
            error("Illegal mixture allocation.")
        end
    end

    ## draw lw, lz
    w, z = rGenDirichlet(a, b, logout=logout)

    if zξout
        return (w, z, ξ)
    else
        return w
    end
end

"""
    del_correctionSBM(π_small, π_large, K)

Computes a SBM correction factor for δ when γ and δ are used to mimic the Dirichlet distribution. The factor is the prior expected proportion of non-'zero' entries in the final probability vector.

### Example
```julia
    del_correctionSBM(0.5, 0.1, 10)
```
"""
function del_correctionSBM(π_small::Float64, π_large::Float64, K::Int)
    (π_small >= 0.0 && π_large >= 0.0 && π_small + π_large <= 1.0) || error("Valid probabilities required.")
    K > 1 || error("K must be greater than 1.")

    if π_large == 0.0
        out = ( (1.0 - π_small)*float(K-1) + 1.0 ) / float(K)
    else
        aa = 1.0 - π_large
        bb = 1.0 - aa^K
        out = ( (1.0 - π_small)*bb/π_large + π_small ) / float(K)
    end

    return out
end
