# # sparseProbVec.jl
#
# export SparseSBPrior, SparseSBPriorP, SparseSBPriorFull,
#   rpost_sparseStickBreak, logSBMmarginal;
#
# struct SparseSBPrior
#   η::Float64
#   p1::Float64
#   μ::Float64
#   M::Float64
# end
#
# mutable struct SparseSBPriorP
#   η::Float64
#   μ::Float64
#   M::Float64
#   a_p1::Float64
#   b_p1::Float64
#   p1_now::Float64
# end
#
# mutable struct SparseSBPriorFull
#   η::Float64
#   M::Float64
#   a_p1::Float64
#   b_p1::Float64
#   a_μ::Float64
#   b_μ::Float64
#   p1_now::Float64
#   μ_now::Float64
# end
#
#
# ### Sparse Stick Breaking mixture
# function h_slice_mu(z2::Vector{Float64}, n2::Int, M::Float64, μ::Float64,
#   logscale::Bool=false)
#
#   a = -n2*(loggamma(M*μ) + loggamma(M*(1.0 - μ)))
#   b = M*μ*sum( log(z2) .- log(1.0 .- z2))
#   out = a + b
#   if (!logscale)
#     out = exp(out)
#   end
#   out
# end
#
#
# function slice_mu(z::Vector{Float64}, ξ::Vector{Int}, μ_old::Float64, M::Float64,
#   a_μ::Float64, b_μ::Float64)
#
#   # assumes xi=1 for group 1, xi=2 for group 2
#
#   n2 = sum( ξ.==2 )
#   betadist = Beta(a_μ, b_μ)
#   if n2 == 0
#     μ_out = rand( betadist )
#   else
#     z2 = copy( z[ findall(ξ.==2) ] )
#     b_u = h_slice_mu(z2, n2, M, μ_old)
#     u = rand( Uniform(0.0, b_u) )
#     lu = log(u)
#
#     ii = 0
#     keepgoing = true
#     μ_cand = 0.0
#     while keepgoing
#       if ii > 1000
#         throw("too many slices!")
#       end
#
#       μ_cand = rand( betadist )
#       lh = h_slice_mu(z2, n2, M, μ_cand, true)
#
#       keepgoing = lu > lh
#       ii += 1
#     end
#
#     μ_out = copy(μ_cand)
#   end
#
#   μ_out
# end
#
#
#
# function rpost_sparseStickBreak(x::Vector{Int}, p1::Float64, η::Float64,
#   μ::Float64, M::Float64, logout::Bool=false)
#
#   K = length(x)
#   n = K - 1
#   rcrx = cumsum( x[range(K, step=-1, length=n)] )[range(n, step=-1, length=n)]  # ∑_{k+1}^K x_k
#   γ = M * μ
#   δ = M * (1.0 - μ)
#
#   ## mixture component 1 update
#   a1 = 1.0 .+ x[1:n]
#   b1 = η .+ rcrx
#
#   ## mixture component 2 update
#   a2 = γ .+ x[1:n]
#   b2 = δ .+ rcrx
#
#   ## calculate posterior mixture weights
#   lgwt1 = log(p1) .- logbeta.(1.0, η) .+ logbeta.(a1, b1)
#   lgwt2 = log(1.0 - p1) .- logbeta.(γ, δ) .+ logbeta.(a2, b2)
#   ldenom = [ logsumexp( [lgwt1[i], lgwt2[i]] ) for i in 1:n ]
#
#   post_p1 = exp.( lgwt1 .- ldenom )
#   is1 = rand(n) .< post_p1
#   ξ = -1*(is1 .* 1 .- 1) .+ 1
#
#   lz = Vector{Float64}(undef, n)
#   for i in 1:n
#     if is1[i]
#       lx1 = log( rand( Distributions.Gamma( a1[i] ) ) )
#       lx2 = log( rand( Distributions.Gamma( b1[i] ) ) )
#       lxm = max(lx1, lx2)
#       lxsum = lxm + log( exp(lx1 - lxm) + exp(lx2 - lxm) ) # logsumexp
#       lz[i] = lx1 - lxsum
#       # z[i] = rand( Beta(a1[i], b1[i]) )
#     else
#       lx1 = log( rand( Distributions.Gamma( a2[i] ) ) )
#       lx2 = log( rand( Distributions.Gamma( b2[i] ) ) )
#       lxm = max(lx1, lx2)
#       lxsum = lxm + log( exp(lx1 - lxm) + exp(lx2 - lxm) ) # logsumexp
#       lz[i] = lx1 - lxsum
#       # z[i] = rand( Beta(a2[i], b2[i]) )
#     end
#   end
#
#   ## break the Stick
#   lw = Vector{Float64}(undef, K)
#   lwhatsleft = 0.0
#
#   for i in 1:n
#     lw[i] = lz[i] + lwhatsleft
#     lwhatsleft += log( 1.0 - exp(lw[i] - lwhatsleft) ) # logsumexp
#   end
#   lw[K] = copy(lwhatsleft)
#
#   z = exp.(lz)
#   w = exp.(lw)
#
#   if logout
#       (lw, lz, ξ)
#     else
#       (w, z, ξ)
#   end
# end
# function rpost_sparseStickBreak(x::Vector{Int}, p1_old::Float64, η::Float64,
#   μ::Float64, M::Float64, a_p1::Float64, b_p1::Float64, logout::Bool=false)
#
#   ## inference for p1 only
#
#   w, z, ξ = rpost_sparseStickBreak(x, p1_old, η, μ, M, logout)
#   p1_now = rand( Beta(a_p1 + sum(ξ==1), b_p1 + sum(ξ==2) ) )
#
#   (w, z, ξ, p1_now) # w and z may be log-valued depending on logout
# end
# function rpost_sparseStickBreak(x::Vector{Int}, p1_old::Float64, η::Float64,
#   μ_old::Float64, M::Float64, a_p1::Float64, b_p1::Float64, a_μ::Float64, b_μ::Float64, logout::Bool=false)
#
#   ## inference for p1 and μ as well
#
#   w, z, ξ = rpost_sparseStickBreak(x, p1_old, η, μ_old, M, logout)
#   if logout
#       μ_now = slice_mu(exp.(z), ξ, μ_old, M, a_μ, b_μ)
#   else
#       μ_now = slice_mu(z, ξ, μ_old, M, a_μ, b_μ)
#   end
#
#   p1_now = rand( Beta(a_p1 + sum(ξ==1), b_p1 + sum(ξ==2) ) )
#
#   (w, z, ξ, μ_now, p1_now) # w and z may be log-valued depending on logout
# end
#
#
#
# """
#     logSBMmarginal(x, p1, η, μ, M)
#
#   Calculate the log of the SBM prior predictive probability mass function.
#
# """
# function logSBMmarginal(x::Vector{Int}, p1::Float64, η::Float64,
#   μ::Float64, M::Float64)
#
#       K = length(x)
#       n = K - 1
#       rcrx = cumsum( x[range(K, step=-1, length=n)] )[range(n, step=-1, length=n)]  # ∑_{k+1}^K x_k
#       γ = M * μ
#       δ = M * (1.0 - μ)
#
#       ## mixture component 1 update
#       a1 = 1.0 .+ x[1:n]
#       b1 = η .+ rcrx
#
#       ## mixture component 2 update
#       a2 = γ .+ x[1:n]
#       b2 = δ .+ rcrx
#
#       ## calculate posterior mixture weights
#       lgwt1 = log(p1) .- logbeta.(1.0, η) .+ logbeta.(a1, b1)
#       lgwt2 = log(1.0 - p1) .- logbeta.(γ, δ) .+ logbeta.(a2, b2)
#       lsum = [ logsumexp( [lgwt1[i], lgwt2[i]] ) for i in 1:n ]
#
#       sum( lsum )
# end
