export quadratic_estimate

"""

    quadratic_estimate(ds::DataSet, which; wiener_filtered=true)
    quadratic_estimate((ds₁::DataSet, ds₂::DataSet), which; wiener_filtered=true)

Compute the quadratic estimate of `ϕ` given data.

The `ds` or `(ds₁,ds₂)` tuple contain the DataSet object(s) which house the
data and covariances used in the estimate. Note that only the Fourier-diagonal
approximations for the beam, mask, and noise, i.e. `B̂`, `M̂`, and
`Cn̂`, are accounted for. To account full operators (if they are not actually
Fourier-diagonal), you should compute the impact using Monte Carlo.

If a tuple is passed in, the result will come from correlating the data from
`ds₁` with that from `ds₂`.

An optional keyword argument `AL` can be passed in case the QE normalization
was already computed, in which case it won't be recomputed during the
calculation.

Returns a named tuple of `(;ϕqe, AL, Nϕ)` where `ϕqe` is the (possibly Wiener
filtered, depending on `wiener_filtered` option) quadratic estimate, `AL` is the
normalization (which is already applied to ϕqe, it does not need to be applied
again), and `Nϕ` is the analytic N⁰ noise bias (Nϕ==AL if using unlensed
weights, currently only Nϕ==AL is always returned, no matter the weights)
"""
function quadratic_estimate(
    (ds₁,ds₂) :: NTuple{2,DataSet}, 
    which = nothing; 
    wiener_filtered = true, 
    AL = nothing, 
    weights = :unlensed
)
    @assert weights in (:lensed, :unlensed) "weights should be :lensed or :unlensed"
    if isnothing(which)
        which = ds₁.d isa FlatS0 ? :TT : :EB
    end
    @assert (which in [:TT, :EE, :EB]) "which='$which' not implemented"
    @assert (ds₁.Cf===ds₂.Cf && ds₁.Cf̃===ds₂.Cf̃ && ds₁.Cn̂===ds₂.Cn̂ && ds₁.Cϕ===ds₂.Cϕ && ds₁.B̂===ds₂.B̂) "operators in `ds₁` and `ds₂` should be the same"
    @unpack Cf, Cf̃, Cn̂, Cϕ, B̂, M̂ = ds₁()
    pol = which == :TT ? :I : :P
    quadratic_estimate(Val(which), (ds₁.d[pol], ds₂.d[pol]), Cf[pol], Cf̃[pol], Cn̂[pol], Cϕ, (M̂*B̂)[pol], wiener_filtered, weights, AL)
end

quadratic_estimate(ds::DataSet, args...; kwargs...) = quadratic_estimate((ds,ds), args...; kwargs...)


@doc doc"""

    QE_leg(C::Diagonal, inds...)

The quadratic estimate and normalization expressions all consist of
terms involving products of two "legs", each leg which look like:

    C * l[i] * l̂[j] * l̂[k] * ... 

where C is some field or diagonal covariance, l[i] is the Fourier
wave-vector in direction i (for i=1:2), and l̂[i] = l[i]/‖l‖. For
example, there's a leg in the EB estimator that looks like: 

    (CE * (CẼ+Cn) \ d[:E])) * l[i] * l̂[j] * l̂[k]

The function `QE_leg` computes quatities like these, e.g. the above
would be given by:

    QE_leg((CE * (CẼ+Cn) \ d[:E])), [i], j, k)

(where note that specifying whether its the Fourier wave-vector l
instead of the unit-vector l̂ is done by putting that index in
brackets).

Additionally, all of these terms are symmetric in their indices, i.e.
in `(i,j,k)` in this case. The `QE_leg` function is smart about this,
and is memoized so that each unique set of indices is only computed
once. This leads to a pretty drastic speedup for terms with many
indices like those that arize in the EE and EB normalizations, and
lets us write code which is both clear and fast without having to
think too hard about these symmetries.

"""
QE_leg(C::Diagonal, inds...) = QE_leg(C.diag, inds...)
function QE_leg(C::Field, inds...)
    n = count((x->x isa Int),inds)
    p₁,p₂ = (count(==(i), first.(inds)) for i=1:2)
    QE_leg(C, (n, p₁, p₂))
end
@memoize function QE_leg(C::Field, (n, p₁, p₂)::Tuple)
    Map(@. nan2zero($Ð(C) * ∇[1].diag^p₁ * ∇[2].diag^p₂ / sqrt(∇².diag)^n))
end
ϵ(x...) = levicivita([x...])
inds(D) = collect(product(repeated(1:2,D)...))[:]

function quadratic_estimate(::Val{:TT}, (d₁,d₂)::NTuple{2,FlatS0}, Cf, Cf̃, Cn, Cϕ, TF, wiener_filtered, weights, AL=nothing)

    ΣTtot = TF^2 * Cf̃ + Cn
    CT = (weights==:unlensed) ? Cf : Cf̃
    
    # unnormalized estimate
    ϕqe_unnormalized = @subst -sum(∇[i] * Fourier(QE_leg($(ΣTtot\(TF*d₁))) * QE_leg($(CT*(ΣTtot\(TF*d₂))), [i])) for i=1:2)
    
    # normalization
    if AL == nothing
        AL = @subst begin
            A(i,j) = (
                QE_leg($(TF^2 * CT^2 / ΣTtot), [i], [j]) * QE_leg($(TF^2      / ΣTtot)     )
              + QE_leg($(TF^2 * CT   / ΣTtot), [i]     ) * QE_leg($(TF^2 * CT / ΣTtot), [j])
            )
            pinv(Diagonal(sum(real.(∇[i].diag .* ∇[j].diag .* Fourier(A(i,j))) for (i,j) in inds(2))))
        end
    end
    Nϕ = AL # true only for unlensed weights
    
    Memoization.empty_cache!(QE_leg)

    ϕqe = (wiener_filtered ? (Cϕ*pinv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    (;ϕqe, AL, Nϕ)

end


function quadratic_estimate(::Val{:EE}, (d₁,d₂)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, TF, wiener_filtered, weights, AL=nothing)
    
    TF² = TF[:E]^2
    ΣEtot = TF² * Cf̃[:E] + Cn[:E]
    CE = ((weights==:unlensed) ? Cf : Cf̃)[:E]

    # unnormalized estimate
    ϕqe_unnormalized = @subst begin
        I(i) = -(
            2sum(QE_leg($(CE * (ΣEtot \ (TF*d₁)[:E])), [i], j, k) * QE_leg($((ΣEtot \ (TF*d₂)[:E])), j, k) for (j,k) in inds(2))
               - QE_leg($(CE * (ΣEtot \ (TF*d₁)[:E])), [i]      ) * QE_leg($((ΣEtot \ (TF*d₂)[:E]))      )
        )
        sum(∇[i] * Fourier(I(i)) for i=1:2)
    end

    # normalization
    if AL == nothing
        AL = @subst begin
            A1(i,j) = -4 * sum( ϵ(m,p,3) * ϵ(n,q,3) * (
                  QE_leg($(TF² * CE^2 / ΣEtot), [i], [j], k, l, m, n) .* QE_leg($(TF²      / ΣEtot),      k, l, p, q)
                + QE_leg($(TF² * CE   / ΣEtot), [i],      k, l, m, n) .* QE_leg($(TF² * CE / ΣEtot), [j], k, l, p, q))
                for (k,l,m,n,p,q) in inds(6)
            )
            A2(i,j) = (
                  QE_leg($(TF² * CE^2 / ΣEtot), [i], [j]) .* QE_leg($(TF²      / ΣEtot)     )
                + QE_leg($(TF² * CE   / ΣEtot), [i]     ) .* QE_leg($(TF² * CE / ΣEtot), [j])
            )
            pinv(Diagonal(sum(real.(∇[i].diag .* ∇[j].diag .* Fourier(A1(i,j) + A2(i,j))) for (i,j) in inds(2))))
        end
    end
    Nϕ = AL # true only for unlensed weights
    
    Memoization.empty_cache!(QE_leg)

    ϕqe = (wiener_filtered ? (Cϕ*pinv(Cϕ+Nϕ)) : 1) * (AL*ϕqe_unnormalized)
    (;ϕqe, AL, Nϕ)

end


function quadratic_estimate(::Val{:EB}, (d₁,d₂)::NTuple{2,FlatS2}, Cf, Cf̃, Cn, Cϕ, TF, wiener_filtered, weights, AL=nothing; zeroB=false)
    
    CE, CB = getindex.(Ref((weights==:unlensed) ? Cf : Cf̃),(:E,:B))
    TF²E, TF²B = TF[:E]^2, TF[:B]^2
    ΣEtot = TF²E * Cf̃[:E] + Cn[:E]
    ΣBtot = TF²B * Cf̃[:B] + Cn[:B]
    

    # unnormalized estimate
    ϕqe_unnormalized = @subst begin
        I(i) = 2 * sum(  ϵ(k,l,3) * (
                           QE_leg($(CE * (ΣEtot \ (TF*d₁)[:E])), [i], j, k) * QE_leg($(     (ΣBtot \ (TF*d₂)[:B])),      j, l)
            - (zeroB ? 0 : QE_leg($(      ΣEtot \ (TF*d₁)[:E]),       j, k) * QE_leg($(CB * (ΣBtot \ (TF*d₂)[:B])), [i], j, l)))
            for (j,k,l) in inds(3)
        )
        sum(∇[i] * Fourier(I(i)) for i=1:2)
    end

    # normalization
    if AL == nothing
        AL = @subst begin
            A(i,j) = 4 * sum( ϵ(m,p,3) * ϵ(n,q,3) * (
                                 QE_leg($(TF²E * CE^2 / ΣEtot), [i], [j], k, l, m, n) * QE_leg($(TF²B        / ΣBtot),           k, l, p, q)
                + (zeroB ? 0 : -2QE_leg($(TF²E * CE   / ΣEtot), [i],      k, l, m, n) * QE_leg($(TF²B * CB   / ΣBtot),      [j], k, l, p, q))
                + (zeroB ? 0 :   QE_leg($(TF²E        / ΣEtot),           k, l, m, n) * QE_leg($(TF²B * CB^2 / ΣBtot), [i], [j], k, l, p, q)))
                for (k,l,m,n,p,q) in inds(6)
            )
            pinv(Diagonal(sum(real.(∇[i].diag .* ∇[j].diag .* Fourier(A(i,j))) for (i,j) in inds(2))))
        end
    end
    Nϕ = AL # true only for unlensed weights
    
    Memoization.empty_cache!(QE_leg)

    ϕqe = (wiener_filtered ? (Cϕ*pinv(Cϕ+Nϕ)) : 1) * (AL * ϕqe_unnormalized)
    (;ϕqe, AL, Nϕ)

end