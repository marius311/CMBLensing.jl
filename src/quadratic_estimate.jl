export quadratic_estimate

"""

    quadratic_estimate(ds::DataSet, which; wiener_filtered=true)

Compute quadratic estimate for ϕ given data. 
    
* `d` - data
* `Cf, Cf̃` - unlensed and lensed _beamed_ theory covariances
* `Cn` - noise covariance (beam _not_ deconvolved)
* `Cϕ` - (optional) lensing potential theory covariance. if provided, the result
         is Wiener filtered, otherwise the unbiased estimate is retured. 
         
Returns a tuple of `(ϕqe, Nϕ)` where `ϕqe` is the quadratic estimate and `Nϕ` is
the N0 noise bias.
"""
function quadratic_estimate(ds::DataSet{F}, which; wiener_filtered=true) where {F}
    @assert (which in [:TT, :EE, :EB]) "which='$which' not implemented"
    if F<:FlatS02
        ds = subblock(ds, (which==:TT) ? :T : :P)
    end
    @unpack d, Cf, Cf̃, Cn̂, Cϕ, B̂ = ds
    quadratic_estimate_func = @match which begin
        :TT => quadratic_estimate_TT
        :EE => quadratic_estimate_EE
        :EB => quadratic_estimate_EB
        _   => error("`which` argument to `quadratic_estimate` should be one of (:TT, :EE, :EB)")
    end
    quadratic_estimate_func(d, B̂^2*Cf, B̂^2*Cf̃, Cn̂, Cϕ, wiener_filtered)
end
quadratic_estimate(ds::DataSet{<:Field{<:Any,S0}}; kwargs...) = quadratic_estimate(ds, :TT; kwargs...)



function quadratic_estimate_TT(d::FlatS0, Cf, Cf̃, Cn, Cϕ, wiener_filtered)

    L⃗,L² = get_L⃗_L²(d)
    
    # unnormalized estimate
    ϕqe_unnormalized = -sum(L⃗[i] * Fourier(Map((Cf̃+Cn)\d) * Map(L⃗[i]*(Cf*((Cf̃+Cn)\d)))) for i=1:2)
    
    # normalization
    I(i,j) = (  Map(Cf^2 * ((Cf̃+Cn) \ (L⃗[i] * L⃗[j]))) * Map(nan2zero.(inv(Cf̃+Cn).f))
              + Map(Cf   * ((Cf̃+Cn) \  L⃗[i]))         * Map(Cf * ((Cf̃+Cn) \ L⃗[j])))
    AL = Nϕ = 2π * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(I(i,j)) for i=1:2, j=1:2)))
    
    
    ϕqe_normalized = AL * ϕqe_unnormalized
    
    ϕqe = (wiener_filtered ? (Cϕ * inv(Cϕ + Nϕ)) : 1) * ϕqe_normalized
    
    @ntpack ϕqe Nϕ

end

# EB estimator, currently assuming B=0 (todo: remove this assumption)
function quadratic_estimate_EB(d::FlatS2, Cf, Cf̃, Cn, Cϕ, wiener_filtered)

    L⃗,L² = get_L⃗_L²(d.E)
    ϵ(x...) = levicivita([x...])
    CE,CB   = Cf[:E], Cf[:B]
    CẼ,CB̃   = Cf̃[:E], Cf̃[:B]
    CEn,CBn = Cn[:E], Cn[:B]

    # unnormalized estimate
    E(i,j,k) = Map(L² \ L⃗[i] * L⃗[j] * L⃗[k] * (CE * ((CẼ+CEn) \ d.E)))
    B(i,j)   = Map(L² \ L⃗[i] * L⃗[j]              *(((CB̃+CBn) \ d.B)))
    ϕqe_unnormalized = 2 * sum(L⃗[i] * Fourier(sum(ϵ(k,m,3) * E(i,j,k) * B(j,m) for j=1:2,k=1:2,m=1:2)) for i=1:2)
    
    # normalization
    E2(i,j,q,k,n,p) = Map(CE^2 * ((CẼ+CEn) \ (L²^2 \ L⃗[i] * L⃗[j] * L⃗[q] * L⃗[k] * L⃗[n] * L⃗[p])))
    B2(q,m,n,s)     = Map(       ((CB̃+CBn) \ (L²^2 \ L⃗[q] * L⃗[m] * L⃗[n] * L⃗[s])))
    I(i,j)          = sum(ϵ(k,m,3) * ϵ(p,s,3) * E2(i,j,q,k,n,p) * B2(q,m,n,s) for k=1:2,m=1:2,n=1:2,p=1:2,q=1:2,s=1:2)
    AL = Nϕ = π/2 * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(I(i,j)) for i=1:2,j=1:2)))
    
    ϕqe_normalized = AL * ϕqe_unnormalized
    
    ϕqe = (wiener_filtered ? (Cϕ * inv(Cϕ + Nϕ)) : 1) * ϕqe_normalized
    
    @ntpack ϕqe Nϕ
    
end


function quadratic_estimate_EE(d::FlatS2, Cf, Cf̃, Cn, Cϕ, wiener_filtered)

    L⃗,L² = get_L⃗_L²(d.E)
    ϵ(x...) = levicivita([x...])
    CE,CB   = Cf[:E], Cf[:B]
    CẼ,CB̃   = Cf̃[:E], Cf̃[:B]
    CEn,CBn = Cn[:E], Cn[:B]

    # unnormalized estimate
    E1(i,j,k) = 2*(Map(L² \ L⃗[i] * L⃗[j] * L⃗[k] * (CE * ((CẼ+CEn) \ d[:E]))) * Map(L² \ L⃗[j] * L⃗[k] *(((CẼ+CEn) \ d[:E]))))
    E2(i)     =  -(Map(     L⃗[i]               * (CE * ((CẼ+CEn) \ d[:E]))) * Map(                  (((CẼ+CEn) \ d[:E]))))
    ϕqe_unnormalized = -2sum(L⃗[i] * Fourier( sum(E1(i,j,k) for j=1:2,k=1:2) + E2(i) ) for i=1:2)
    
    # normalization
    E2(i,j,q,k,n,p) = Map(CE^2 * ((CẼ+CEn) \ (L²^2 \ L⃗[i] * L⃗[j] * L⃗[q] * L⃗[k] * L⃗[n] * L⃗[p])))
    B2(q,m,n,s)     = Map(       ((CB̃+CBn) \ (L²^2 \ L⃗[q] * L⃗[m] * L⃗[n] * L⃗[s])))
    I(i,j)          = sum(ϵ(k,m,3) * ϵ(p,s,3) * E2(i,j,q,k,n,p) * B2(q,m,n,s) for k=1:2,m=1:2,n=1:2,p=1:2,q=1:2,s=1:2)
    AL = Nϕ = π/√2 * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(I(i,j)) for i=1:2,j=1:2)))
    
    ϕqe_normalized = AL * ϕqe_unnormalized
    
    ϕqe = (wiener_filtered ? (Cϕ * inv(Cϕ + Nϕ)) : 1) * ϕqe_normalized
    
    @ntpack ϕqe Nϕ

end


function get_L⃗_L²(f)
    # todo: turn L⃗ into a full-fledged operator like ∇
    L⃗  = (@SVector [(∇[1] .+ 0Ð(f)), (∇[2] .+ 0Ð(f))])
    L² = FullDiagOp(L⃗'L⃗)
    L⃗,L²
end
