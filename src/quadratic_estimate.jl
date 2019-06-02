export quadratic_estimate

"""

    quadratic_estimate(ds::DataSet, which; wiener_filtered=true)
    quadratic_estimate((ds1::DataSet,ds2::DataSet), which; wiener_filtered=true)

Compute quadratic estimate of ϕ given data.

The `ds` or `(ds1,ds2)` tuple contain the DataSet object(s) which houses the
data and covariances used in the estimate. Note that only the Fourier-diagonal
approximations for the beam and noise, `ds.B̂` and `ds.Cn̂`, are accounted for,
and the mask `M` is completely ignored. To account for these fully, you should
compute their impact using Monte Carlo. 

If a tuple is passed in, the result will come from correlating the data from
`ds1` with that from `ds2`, which can be useful for debugging / isolating
various noise terms. 

Returns a NamedTuple `(ϕqe, Nϕ)` where `ϕqe` is the (possibly Wiener filtered,
depending on `wiener_filtered` option) quadratic estimate and `Nϕ` is the
analytic N0 noise bias.
"""
function quadratic_estimate((ds1,ds2)::NTuple{2,DataSet{F}}, which; wiener_filtered=true) where {F}
    @assert (which in [:TT, :EE, :EB]) "which='$which' not implemented"
    if F<:FlatS02
        ds1 = subblock(ds1, (which==:TT) ? :T : :P)
        ds2 = subblock(ds2, (which==:TT) ? :T : :P)
    end
    @unpack Cf, Cf̃, Cn̂, Cϕ, B̂ = ds1
    @assert (ds2.Cf==Cf && ds2.Cf̃==Cf̃ && ds2.Cn̂==Cn̂ && ds2.Cϕ==Cϕ && ds2.B̂==B̂) "operators in `ds1` and `ds2` should be the same"
    quadratic_estimate_func = @match which begin
        :TT => quadratic_estimate_TT
        :EE => quadratic_estimate_EE
        :EB => quadratic_estimate_EB
        _   => error("`which` argument to `quadratic_estimate` should be one of (:TT, :EE, :EB)")
    end
    quadratic_estimate_func((ds1.d, ds2.d), Cf, Cf̃, Cn̂, Cϕ, wiener_filtered)
end
quadratic_estimate(ds::DataSet, which; kwargs...) = quadratic_estimate((ds,ds), which; kwargs...)
quadratic_estimate(ds::DataSet{<:Field{<:Any,S0}}; kwargs...) = quadratic_estimate(ds, :TT; kwargs...)



quadratic_estimate_TT(d::Field, args...) = quadratic_estimate_TT((d,d), args...)
quadratic_estimate_EB(d::Field, args...) = quadratic_estimate_EB((d,d), args...)
quadratic_estimate_EE(d::Field, args...) = quadratic_estimate_EE((d,d), args...)


function quadratic_estimate_TT((d1,d2)::NTuple{2,FlatS0}, Cf, Cf̃, Cn, Cϕ, wiener_filtered)

    L⃗,L² = get_L⃗_L²(d1)
    
    # unnormalized estimate
    ϕqe_unnormalized = -sum(L⃗[i] * Fourier(Map((Cf̃+Cn)\d1) * Map(L⃗[i]*(Cf*((Cf̃+Cn)\d2)))) for i=1:2)
    
    # normalization
    I(i,j) = (  Map(Cf^2 * ((Cf̃+Cn) \ (L⃗[i] * L⃗[j]))) * Map(nan2zero.(inv(Cf̃+Cn).f))
              + Map(Cf   * ((Cf̃+Cn) \  L⃗[i]))         * Map(Cf * ((Cf̃+Cn) \ L⃗[j])))
    AL = Nϕ = 2π * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(I(i,j)) for i=1:2, j=1:2)))
    
    
    ϕqe_normalized = AL * ϕqe_unnormalized
    
    ϕqe = (wiener_filtered ? (Cϕ * inv(Cϕ + Nϕ)) : 1) * ϕqe_normalized
    
    @ntpack ϕqe Nϕ

end

# EB estimator, currently assuming B=0 (todo: remove this assumption)
function quadratic_estimate_EB((d1,d2)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, wiener_filtered)

    L⃗,L² = get_L⃗_L²(d1.E)
    ϵ(x...) = levicivita([x...])
    CE,CB   = Cf[:E], Cf[:B]
    CẼ,CB̃   = Cf̃[:E], Cf̃[:B]
    CEn,CBn = Cn[:E], Cn[:B]

    # unnormalized estimate
    @symmetric_memoized E(i,j,k) = Map(L² \ L⃗[i] * L⃗[j] * L⃗[k] * (CE * ((CẼ+CEn) \ d1.E)))
    @symmetric_memoized B(i,j)   = Map(L² \ L⃗[i] * L⃗[j]              *(((CB̃+CBn) \ d2.B)))
    ϕqe_unnormalized = 2 * sum(L⃗[i] * Fourier(sum(ϵ(k,m,3) * E(i,j,k) * B(j,m) for j=1:2,k=1:2,m=1:2)) for i=1:2)
    
    # normalization
    @symmetric_memoized AE(i,j,q,k,n,p) = Map(CE^2 * ((CẼ+CEn) \ (L²^2 \ L⃗[i] * L⃗[j] * L⃗[q] * L⃗[k] * L⃗[n] * L⃗[p])))
    @symmetric_memoized AB(q,m,n,s)     = Map(       ((CB̃+CBn) \ (L²^2 \ L⃗[q] * L⃗[m] * L⃗[n] * L⃗[s])))
    @symmetric_memoized I(i,j)          = 4 * sum(ϵ(k,m,3) * ϵ(p,s,3) * AE(i,j,q,k,n,p) * AB(q,m,n,s) for k=1:2,m=1:2,n=1:2,p=1:2,q=1:2,s=1:2)
    AL = Nϕ = 2π * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(I(i,j)) for i=1:2,j=1:2)))
    
    ϕqe_normalized = AL * ϕqe_unnormalized
    
    ϕqe = (wiener_filtered ? (Cϕ * inv(Cϕ + Nϕ)) : 1) * ϕqe_normalized
    
    @ntpack ϕqe Nϕ
    
end



"""


"""
function get_qe_quantities(f)
    L⃗  = (@SVector [(∇[1] .+ 0Ð(f)), (∇[2] .+ 0Ð(f))])
    L² = FullDiagOp(L⃗'L⃗)
    ϵ(x...) = levicivita([x...])
    @symmetric_memoized L⃗factors(inds...) = broadcast(*, getindex.(L⃗, inds)...)
    term(C,inds...) = Map(nan2zero.(C) .* L⃗factors(inds...))
    @ntpack L⃗ L² ϵ term
end
inds(n) = collect(product(repeated(1:2,n)...))[:]


function quadratic_estimate_EE2((d1,d2)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, wiener_filtered)
    
    @unpack L⃗,ϵ,term = get_qe_quantities(d1.E)
    CE,CẼ,CEn = Cf[:E], Cf̃[:E], Cn[:E]

    # unnormalized estimate
    I(i) = sum(
          term((CE * (CẼ+CEn) \ d1.E), i, j, k) * term((     ((CẼ+CEn) \ d2.E)),    k)
        - term((     (CẼ+CEn) \ d1.E), i,    k) * term((CE * ((CẼ+CEn) \ d2.E)), j, k)/2
        for (j,k) in inds(2)
    )
    ϕqe_unnormalized = -2 * sum(L⃗[i] * Fourier(I(i)) for i=1:2)

    # normalization
    A1(i,j) = sum( ϵ(m,p,3) * ϵ(n,q,3) * (
         term((@. CE^2 / (CẼ+CEn)), i, j, k, l, m, n) * term((@. 1    / (CẼ+CEn)),       k, l, p, q)
       - term((@. CE   / (CẼ+CEn)), i,    k, l, m, n) * term((@. CE   / (CẼ+CEn)),    j, k, l, p, q))
        for (k,l,m,n,o,p) in inds(6)
    )
    A2(i,j) = (
         term((@. CE^2 / (CẼ+CEn)), i, j, ) * term((@. 1    / (CẼ+CEn)),     )
       - term((@. CE   / (CẼ+CEn)), i     ) * term((@. CE   / (CẼ+CEn)),    j)
    )
    AL = Nϕ = 2π * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(A1(i,j) + A2(i,j)) for i=1:2,j=1:2)))

    ϕqe = (wiener_filtered ? (Cϕ*inv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    @ntpack ϕqe Nϕ

end


function quadratic_estimate_EB2((d1,d2)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, wiener_filtered; zeroB=false)
    
    @unpack L⃗,ϵ,term = get_qe_quantities(d1.E)
    CE,CB   = Cf[:E], Cf[:B]
    CẼ,CB̃   = Cf̃[:E], Cf̃[:B]
    CEn,CBn = Cn[:E], Cn[:B]

    # unnormalized estimate
    I(i) = sum(
                       term((CE * ((CẼ+CEn) \ d1.E)), i, j, k) * term(      ((CB̃+CBn) \ d2.B),     k)
        - (zeroB ? 0 : term((      (CẼ+CEn) \ d1.E),     j, k) * term((CB * ((CB̃+CBn) \ d2.E)), i, k))
        for (j,k) in inds(2)
    )
    ϕqe_unnormalized = -2 * sum(L⃗[i] * Fourier(I(i)) for i=1:2)

    # normalization
    @symmetric_memoized A(i,j) = sum( ϵ(m,p,3) * ϵ(n,q,3) * (
                        term((@. CE^2 / (CẼ+CEn)), i, j, k, l, m, n) * term((@. 1    / (CB̃+CBn)),       k, l, p, q)
        + zeroB ? 0 : -2term((@. CE   / (CẼ+CEn)), i,    k, l, m, n) * term((@. CB   / (CB̃+CBn)),    j, k, l, p, q)
        + zeroB ? 0 :   term((@. 1    / (CẼ+CEn)),       k, l, m, n) * term((@. CB^2 / (CB̃+CBn)), i, j, k, l, p, q))
        for (k,l,m,n,o,p) in inds(6)
    )
    AL = Nϕ = 2π * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(A(i,j)) for i=1:2,j=1:2)))

    ϕqe = (wiener_filtered ? (Cϕ*inv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    @ntpack ϕqe Nϕ

end


function quadratic_estimate_TT2((d1,d2)::NTuple{2,FlatS0}, Cf, Cf̃, Cn, Cϕ, wiener_filtered)

    @unpack L⃗,term = get_qe_quantities(d1)
    
    # unnormalized estimate
    ϕqe_unnormalized = -sum(L⃗[i] * Fourier(term(((Cf̃+Cn)\d1)) * term((Cf*((Cf̃+Cn)\d2)), i)) for i=1:2)
    
    # normalization
    A(i,j) = (
        term((@. Cf^2 / (Cf̃+Cn)), i, j) * term((@. 1  / (Cf̃+Cn))   )
      + term((@. Cf   / (Cf̃+Cn)), i   ) * term((@. Cf / (Cf̃+Cn)), j)
    )
    AL = Nϕ = 2π * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(A(i,j)) for (i,j) in inds(2))))
    
    ϕqe = (wiener_filtered ? (Cϕ*inv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    @ntpack ϕqe Nϕ

end
