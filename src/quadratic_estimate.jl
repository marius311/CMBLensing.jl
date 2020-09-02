export quadratic_estimate

"""

    quadratic_estimate(ds::DataSet, which; wiener_filtered=true)
    quadratic_estimate((ds1::DataSet,ds2::DataSet), which; wiener_filtered=true)

Compute quadratic estimate of ϕ given data.

The `ds` or `(ds1,ds2)` tuple contain the DataSet object(s) which houses the
data and covariances used in the estimate. Note that only the Fourier-diagonal
approximations for the beam, mask, and noise,, i.e. `ds.B̂`, `ds.M̂`, and
`ds.Cn̂`, are accounted for. To account full operators (if they are not actually
Fourier-diagonal), you should compute the impact using Monte Carlo.

If a tuple is passed in, the result will come from correlating the data from
`ds1` with that from `ds2`, which can be useful for debugging / isolating
various noise terms. 

An optional keyword argument `AL` can be passed in in case the QE normalization
was already computed, in which case it won't be recomputed during the
calculation.

Returns a NamedTuple `(ϕqe, AL, Nϕ)` where `ϕqe` is the (possibly Wiener
filtered, depending on `wiener_filtered` option) quadratic estimate, `AL` is the
normalization (which is already applied to ϕqe, it does not need to be applied
again), and `Nϕ` is the analytic N0 noise bias (Nϕ==AL if using unlensed
weights, currently only Nϕ==AL is always returned, no matter the weights)
"""
function quadratic_estimate((ds1,ds2)::NTuple{2,DataSet}, which; wiener_filtered=true, AL=nothing, weights=:unlensed) where {F1,F2}
    @assert weights in (:lensed, :unlensed) "weights should be :lensed or :unlensed"
    @assert (which in [:TT, :EE, :EB]) "which='$which' not implemented"
    @assert (ds1.Cf===ds2.Cf && ds1.Cf̃===ds2.Cf̃ && ds1.Cn̂===ds2.Cn̂ && ds1.Cϕ===ds2.Cϕ && ds1.B̂===ds2.B̂) "operators in `ds1` and `ds2` should be the same"
    @assert spin(ds1.d)==spin(ds2.d)
    check_hat_operators(ds1)
    @unpack Cf, Cf̃, Cn̂, Cϕ, B̂, M̂ = ds1()
    (quadratic_estimate_func, pol) = @match which begin
        :TT => (quadratic_estimate_TT, :I)
        :EE => (quadratic_estimate_EE, :P)
        :EB => (quadratic_estimate_EB, :P)
        _   => error("`which` argument to `quadratic_estimate` should be one of (:TT, :EE, :EB)")
    end
    quadratic_estimate_func((ds1.d[pol], ds2.d[pol]), Cf[pol], Cf̃[pol], Cn̂[pol], Cϕ, (M̂*B̂)[pol], wiener_filtered, weights, AL)
end
quadratic_estimate(ds::DataSet, which; kwargs...) = quadratic_estimate((ds,ds), which; kwargs...)
quadratic_estimate(ds::DataSet; kwargs...) = quadratic_estimate(ds, ds.d isa FlatS0 ? :TT : :EB; kwargs...) # somewhat arbitraritly make default P estimate be EB



quadratic_estimate_TT(d::Field, args...) = quadratic_estimate_TT((d,d), args...)
quadratic_estimate_EB(d::Field, args...) = quadratic_estimate_EB((d,d), args...)
quadratic_estimate_EE(d::Field, args...) = quadratic_estimate_EE((d,d), args...)


@doc doc"""

All of the terms in the quadratic estimate and normalization expressions look like

    C * l[i] * l̂[j] * l̂[k] * ... 

where C is some field or diagonal covariance. For example, there's a term in the EB
estimator that looks like:

    (CE * (CẼ+Cn) \ d[:E])) * l[i] * l̂[j] * l̂[k]
    
(where note that `l̂[j]` and `l̂[k]` are unit vectors, but `l[i]` is not).  The
function `get_term_memoizer` returns a function `term` which could be called in
the following way to compute this term:

    term((CE * (CẼ+Cn) \ d[:E])), [i], j, k)
    
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
    Lfactors(inds...) = broadcast!(*, similar(Ðf), getproperty.(getindex.(Ref(∇),inds),:diag)...)
    term(C::Diagonal, inds...) = term(C.diag, inds...)
    term(C::Field, inds...) = term(count((x->x isa Int),inds)/2f0, C, first.(inds)...)
    @sym_memo term(n, C::Field, @sym(inds...)) = Map(nan2zero.(C .* Lfactors(inds...) ./ real.(∇².diag).^n))
    term
end
ϵ(x...) = levicivita([x...])
inds(n) = collect(product(repeated(1:2,n)...))[:]


function quadratic_estimate_TT((d1,d2)::NTuple{2,FlatS0}, Cf, Cf̃, Cn, Cϕ, TF, wiener_filtered, weights, AL=nothing)

    term = get_term_memoizer(d1)
    ΣTtot = TF^2 * Cf̃ + Cn
    CT = (weights==:unlensed) ? Cf : Cf̃
    
    # unnormalized estimate
    ϕqe_unnormalized = @subst -sum(∇[i] * Fourier(term($(ΣTtot\(TF*d1))) * term($(CT*(ΣTtot\(TF*d2))), [i])) for i=1:2)
    
    # normalization
    if AL == nothing
        AL = @subst begin
            A(i,j) = (
                term($(@. TF^2 * CT^2 / ΣTtot), [i], [j]) * term($(@. TF^2      / ΣTtot)     )
              + term($(@. TF^2 * CT   / ΣTtot), [i]     ) * term($(@. TF^2 * CT / ΣTtot), [j])
            )
            pinv(Diagonal(sum(∇[i].diag .* ∇[j].diag .* Fourier(A(i,j)) for (i,j) in inds(2))))
        end
    end
    Nϕ = AL # true only for unlensed weights
    
    Memoization.empty_cache!(term)

    ϕqe = (wiener_filtered ? (Cϕ*pinv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    @namedtuple ϕqe AL Nϕ

end


function quadratic_estimate_EE((d1,d2)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, TF, wiener_filtered, weights, AL=nothing)
    
    term = get_term_memoizer(d1[:E])
    TF² = TF[:E]^2
    ΣEtot = TF² * Cf̃[:E] + Cn[:E]
    CE = ((weights==:unlensed) ? Cf : Cf̃)[:E]

    # unnormalized estimate
    ϕqe_unnormalized = @subst begin
        I(i) = -(
            2sum(term($(CE * (ΣEtot \ (TF*d1)[:E])), [i], j, k) * term($((ΣEtot \ (TF*d2)[:E])), j, k) for (j,k) in inds(2))
               - term($(CE * (ΣEtot \ (TF*d1)[:E])), [i]      ) * term($((ΣEtot \ (TF*d2)[:E]))      )
        )
        sum(∇[i] * Fourier(I(i)) for i=1:2)
    end

    # normalization
    if AL == nothing
        AL = @subst begin
            A1(i,j) = -4 * sum( ϵ(m,p,3) * ϵ(n,q,3) * (
                  term($(@. TF² * CE^2 / ΣEtot), [i], [j], k, l, m, n) * term($(@. TF²      / ΣEtot),      k, l, p, q)
                + term($(@. TF² * CE   / ΣEtot), [i],      k, l, m, n) * term($(@. TF² * CE / ΣEtot), [j], k, l, p, q))
                for (k,l,m,n,p,q) in inds(6)
            )
            A2(i,j) = (
                  term($(@. TF² * CE^2 / ΣEtot), [i], [j]) * term($(@. TF²      / ΣEtot)     )
                + term($(@. TF² * CE   / ΣEtot), [i]     ) * term($(@. TF² * CE / ΣEtot), [j])
            )
            pinv(Diagonal(sum(∇[i].diag .* ∇[j].diag .* Fourier(A1(i,j) + A2(i,j)) for i=1:2,j=1:2)))
        end
    end
    Nϕ = AL # true only for unlensed weights
    
    Memoization.empty_cache!(term)

    ϕqe = (wiener_filtered ? (Cϕ*pinv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    @namedtuple ϕqe AL Nϕ

end


function quadratic_estimate_EB((d1,d2)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, TF, wiener_filtered, weights, AL=nothing; zeroB=false)
    
    term = get_term_memoizer(d1[:E])
    CE, CB = getindex.(Ref((weights==:unlensed) ? Cf : Cf̃),(:E,:B))
    TF²E, TF²B = TF[:E]^2, TF[:B]^2
    ΣEtot = TF²E * Cf̃[:E] + Cn[:E]
    ΣBtot = TF²B * Cf̃[:B] + Cn[:B]
    

    # unnormalized estimate
    ϕqe_unnormalized = @subst begin
        I(i) = 2 * sum(  ϵ(k,l,3) * (
                           term($(CE * (ΣEtot \ (TF*d1)[:E])), [i], j, k) * term($(     (ΣBtot \ (TF*d2)[:B])),      j, l)
            - (zeroB ? 0 : term($(      ΣEtot \ (TF*d1)[:E]),       j, k) * term($(CB * (ΣBtot \ (TF*d2)[:B])), [i], j, l)))
            for (j,k,l) in inds(3)
        )
        sum(∇[i] * Fourier(I(i)) for i=1:2)
    end

    # normalization
    if AL == nothing
        AL = @subst begin
            @sym_memo A(@sym(i,j)) = 4 * sum( ϵ(m,p,3) * ϵ(n,q,3) * (
                                 term($(@. TF²E * CE^2 / ΣEtot), [i], [j], k, l, m, n) * term($(@. TF²B        / ΣBtot),           k, l, p, q)
                + (zeroB ? 0 : -2term($(@. TF²E * CE   / ΣEtot), [i],      k, l, m, n) * term($(@. TF²B * CB   / ΣBtot),      [j], k, l, p, q))
                + (zeroB ? 0 :   term($(@. TF²E        / ΣEtot),           k, l, m, n) * term($(@. TF²B * CB^2 / ΣBtot), [i], [j], k, l, p, q)))
                for (k,l,m,n,p,q) in inds(6)
            )
            AL = pinv(Diagonal(sum(∇[i].diag .* ∇[j].diag .* Fourier(A(i,j)) for i=1:2,j=1:2)))
            Memoization.empty_cache!(A)
            AL
        end
    end
    Nϕ = AL # true only for unlensed weights
    
    Memoization.empty_cache!(term)

    ϕqe = (wiener_filtered ? (Cϕ*pinv(Cϕ+Nϕ)) : 1) * (AL * ϕqe_unnormalized)
    @namedtuple ϕqe AL Nϕ

end