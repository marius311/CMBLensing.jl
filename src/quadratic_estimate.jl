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

An optional keyword argument `Nϕ` can be passed in in case the noise was already
computed, in which case it won't be recomputed during the calculation.

Returns a NamedTuple `(ϕqe, Nϕ)` where `ϕqe` is the (possibly Wiener filtered,
depending on `wiener_filtered` option) quadratic estimate and `Nϕ` is the
analytic N0 noise bias.
"""
function quadratic_estimate((ds1,ds2)::Tuple{DataSet{F1},DataSet{F2}}, which; wiener_filtered=true, Nϕ=nothing) where {F1,F2}
    @assert (which in [:TT, :EE, :EB]) "which='$which' not implemented"
    @assert (ds1.Cf===ds2.Cf && ds1.Cf̃===ds2.Cf̃ && ds1.Cn̂===ds2.Cn̂ && ds1.Cϕ===ds2.Cϕ && ds1.B̂===ds2.B̂) "operators in `ds1` and `ds2` should be the same"
    @assert spin(F1)==spin(F2)
    if F1<:FlatS02
        ds1 = subblock(ds1, (which==:TT) ? :T : :P)
        ds2 = subblock(ds2, (which==:TT) ? :T : :P)
    end
    @unpack Cf, Cf̃, Cn̂, Cϕ, B̂ = ds1
    quadratic_estimate_func = @match which begin
        :TT => quadratic_estimate_TT
        :EE => quadratic_estimate_EE
        :EB => quadratic_estimate_EB
        _   => error("`which` argument to `quadratic_estimate` should be one of (:TT, :EE, :EB)")
    end
    quadratic_estimate_func((ds1.d, ds2.d), Cf, Cf̃, Cn̂, Cϕ, wiener_filtered, Nϕ)
end
quadratic_estimate(ds::DataSet, which; kwargs...) = quadratic_estimate((ds,ds), which; kwargs...)
quadratic_estimate(ds::DataSet{<:Field{<:Any,S0}}; kwargs...) = quadratic_estimate(ds, :TT; kwargs...)
quadratic_estimate(ds::DataSet{<:Field{<:Any,S2}}; kwargs...) = quadratic_estimate(ds, :EB; kwargs...) # somewhat arbitraritly make default P estimate be EB



quadratic_estimate_TT(d::Field, args...) = quadratic_estimate_TT((d,d), args...)
quadratic_estimate_EB(d::Field, args...) = quadratic_estimate_EB((d,d), args...)
quadratic_estimate_EE(d::Field, args...) = quadratic_estimate_EE((d,d), args...)


@doc doc"""

All of the terms in the quadratic estimate and normalization expressions look like

    C * l[i] * l̂[j] * l̂[k] * ... 

where C is some field or diagonal covariance. For example, there's a term in the EB
estimator that looks like:

    (CE * (CẼ+Cn) \ d.E)) * l[i] * l̂[j] * l̂[k]
    
(where note that `l̂[j]` and `l̂[k]` are unit vectors, but `l[i]` is not).  The
function `get_term_memoizer` returns function `term` which could be called in
the following way to compute this term:

    term((CE * (CẼ+Cn) \ d.E)), [i], j, k)
    
(note that the fact that `l[i]` is not a unit vector is specified by putting the
`[i]` index in brackets). 

Additionally, all of these terms are symmetric in their indices, i.e. in
`(i,j,k)` in this case. The `term` function is smart about this, and is memoized
so that each unique set of indices is only computed once. This leads to a pretty
drastic speedup for terms with many indices like those that arize in the EE and
EB normalizations, and lets us write code which is both clear and fast without
having to think too hard about these symmetries.

"""
function get_term_memoizer(f)
    Ðf = Ð(f)
    Lfactors() = 1
    Lfactors(inds...) = broadcast!(*, similar(Ðf), getindex.(Ref(∇),inds)...)
    term(C::FullDiagOp, inds...) = term(C.f, inds...)
    term(C::Field, inds...) = term(count((x->x isa Int),inds)/2, C, first.(inds)...)
    @sym_memo term(n, C::Field, @sym(inds...)) = Map(nan2zero.(C .* Lfactors(inds...) ./ ∇².^n))
    term
end
ϵ(x...) = levicivita([x...])
inds(n) = collect(product(repeated(1:2,n)...))[:]



function quadratic_estimate_TT((d1,d2)::NTuple{2,FlatS0}, Cf, Cf̃, Cn, Cϕ, wiener_filtered, Nϕ=nothing)

    term = get_term_memoizer(d1)
    
    # unnormalized estimate
    ϕqe_unnormalized = @subst -sum(∇[i] * Fourier(term($((Cf̃+Cn)\d1)) * term($(Cf*((Cf̃+Cn)\d2)), [i])) for i=1:2)
    
    # normalization
    if Nϕ == nothing
        Nϕ = @subst begin
            A(i,j) = (
                term($(@. Cf^2 / (Cf̃+Cn)), [i], [j]) * term($(@. 1  / (Cf̃+Cn))     )
              + term($(@. Cf   / (Cf̃+Cn)), [i]     ) * term($(@. Cf / (Cf̃+Cn)), [j])
            )
            2π * inv(FullDiagOp(sum(∇[i] .* ∇[j] .* Fourier(A(i,j)) for (i,j) in inds(2))))
        end
    end
    AL = Nϕ
    
    ϕqe = (wiener_filtered ? (Cϕ*inv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    @ntpack ϕqe Nϕ

end


function quadratic_estimate_EE((d1,d2)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, wiener_filtered, Nϕ=nothing)
    
    term = get_term_memoizer(d1.E)
    (CE, CẼ, CEn) = (Cf[:E], Cf̃[:E], Cn[:E])

    # unnormalized estimate
    ϕqe_unnormalized = @subst begin
        I(i) = -(
            2sum(term($(CE * ((CẼ+CEn) \ d1.E)), [i], j, k) * term($(((CẼ+CEn) \ d2.E)), j, k) for (j,k) in inds(2))
               - term($(CE * ((CẼ+CEn) \ d1.E)), [i]      ) * term($(((CẼ+CEn) \ d2.E))      )
        )
        sum(∇[i] * Fourier(I(i)) for i=1:2)
    end

    # normalization
    if Nϕ == nothing
        Nϕ = @subst begin
            A1(i,j) = -4 * sum( ϵ(m,p,3) * ϵ(n,q,3) * (
                  term($(@. CE^2 / (CẼ+CEn)), [i], [j], k, l, m, n) * term($(@. 1    / (CẼ+CEn)),           k, l, p, q)
                + term($(@. CE   / (CẼ+CEn)), [i],      k, l, m, n) * term($(@. CE   / (CẼ+CEn)),      [j], k, l, p, q))
                for (k,l,m,n,p,q) in inds(6)
            )
            A2(i,j) = (
                  term($(@. CE^2 / (CẼ+CEn)), [i], [j]) * term($(@. 1    / (CẼ+CEn))     )
                + term($(@. CE   / (CẼ+CEn)), [i]     ) * term($(@. CE   / (CẼ+CEn)), [j])
            )
            2π * inv(FullDiagOp(sum(∇[i] .* ∇[j] .* Fourier(A1(i,j) + A2(i,j)) for i=1:2,j=1:2)))
        end
    end
    AL = Nϕ
    
    ϕqe = (wiener_filtered ? (Cϕ*inv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    @ntpack ϕqe Nϕ

end


function quadratic_estimate_EB((d1,d2)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, wiener_filtered, Nϕ=nothing; zeroB=false)
    
    term = get_term_memoizer(d1.E)
    (CE, CB)   = (Cf[:E], Cf[:B])
    (CẼ, CB̃)   = (Cf̃[:E], Cf̃[:B])
    (CEn, CBn) = (Cn[:E], Cn[:B])

    # unnormalized estimate
    ϕqe_unnormalized = @subst begin
        I(i) = 2 * sum(  ϵ(k,l,3) * (
                           term($(CE * ((CẼ+CEn) \ d1.E)), [i], j, k) * term($(     ((CB̃+CBn) \ d2.B)),      j, l)
            - (zeroB ? 0 : term($(      (CẼ+CEn) \ d1.E),       j, k) * term($(CB * ((CB̃+CBn) \ d2.B)), [i], j, l)))
            for (j,k,l) in inds(3)
        )
        sum(∇[i] * Fourier(I(i)) for i=1:2)
    end

    # normalization
    if Nϕ == nothing
        Nϕ = @subst begin
            @sym_memo A(@sym(i,j)) = 4 * sum( ϵ(m,p,3) * ϵ(n,q,3) * (
                                 term($(@. CE^2 / (CẼ+CEn)), [i], [j], k, l, m, n) * term($(@. 1    / (CB̃+CBn)),           k, l, p, q)
                + (zeroB ? 0 : -2term($(@. CE   / (CẼ+CEn)), [i],      k, l, m, n) * term($(@. CB   / (CB̃+CBn)),      [j], k, l, p, q))
                + (zeroB ? 0 :   term($(@. 1    / (CẼ+CEn)),           k, l, m, n) * term($(@. CB^2 / (CB̃+CBn)), [i], [j], k, l, p, q)))
                for (k,l,m,n,p,q) in inds(6)
            )
            2π * inv(FullDiagOp(sum(∇[i] .* ∇[j] .* Fourier(A(i,j)) for i=1:2,j=1:2)))
        end
    end
    AL = Nϕ
    
    ϕqe = (wiener_filtered ? (Cϕ*inv(Cϕ+Nϕ)) : 1) * (AL * ϕqe_unnormalized)
    @ntpack ϕqe Nϕ

end
