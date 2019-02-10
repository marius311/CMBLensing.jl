export ϕqe

macro sym(ex)
    # todo: write this such that it converts a function f(i,j,k,..) which is
    # assumed to be symmetric in its arguments into a memoized function such
    # that each unique set of parameters is only calculated once
    :($(esc(ex)))
end


"""

    ϕqe(d, Cf, Cf̃, Cn, Cϕ=nothing)

Compute quadratic estimate for ϕ given data, d, and signal, lensed signal,
and noise covariances Cf, Cf̃, and Cn, respectively. If signal covariance Cϕ is
given, the result is Wiener filtered.
"""
function ϕqe(d::FlatS0, Cf, Cf̃, Cn, Cϕ=nothing)

    L⃗,L² = get_L⃗_L²(d)
    
    # quadratic estimate
    ϕqe_unnormalized = -L⃗' * Fourier(Map((Cf̃+Cn)\d) * Map(L⃗*(Cf*((Cf̃+Cn)\d))))
    
    # normalization
    I(i,j) = (  Map(Cf^2*inv(Cf̃+Cn)*L⃗[i]*L⃗[j]) * Map(   inv(Cf̃+Cn).f)
              + Map(Cf  *inv(Cf̃+Cn)*L⃗[i])      * Map(Cf*inv(Cf̃+Cn)*L⃗[j]))
    AL = Nϕ = 2π * inv(FullDiagOp(sum(L⃗[i]*L⃗[j]*Fourier(I(i,j)) for i=1:2, j=1:2)))
    
    
    ϕqe_normalized = AL * ϕqe_unnormalized
    
    if Cϕ != nothing
        (Cϕ * inv(Cϕ + Nϕ) * ϕqe_normalized), Nϕ
    else
        ϕqe_normalized, Nϕ
    end

end

# EB estimator, currently assuming B=0 (todo: remove this assumption)
function ϕqe(d::FlatS2{T,P}, Cf, Cf̃, Cn, Cϕ=nothing) where {T,P}

    L⃗,L² = get_L⃗_L²(d[:E])

    ϵ(x...) = levicivita([x...])
    
    CE,CB   = Cf[:E], Cf[:B]
    CẼ,CB̃   = Cf̃[:E], Cf̃[:B]
    CEn,CBn = Cn[:E], Cn[:B]

    # quadratic estimate
    E(i,j,k) = Map(L² \ L⃗[i] * L⃗[j] * L⃗[k] * (CE * inv(CẼ+CEn) * d[:E]))
    B(i,j)   = Map(L² \ L⃗[i] * L⃗[j]              *(inv(CB̃+CBn) * d[:B]))
    ϕqe_unnormalized = sum(L⃗[i] * Fourier(sum(ϵ(k,m,3) * E(i,j,k) * B(j,m) for j=1:2,k=1:2,m=1:2)) for i=1:2)
    
    # normalization
    E2(i,j,q,k,n,p) = Map(CE^2 * inv(CẼ+CEn) * (L²^2 \ L⃗[i] * L⃗[j] * L⃗[q] * L⃗[k] * L⃗[n] * L⃗[p]))
    B2(q,m,n,s)     = Map(       inv(CB̃+CBn) * (L²^2 \ L⃗[q] * L⃗[m] * L⃗[n] * L⃗[s]))
    I(i,j)          = sum(ϵ(k,m,3) * ϵ(p,s,3) * E2(i,j,q,k,n,p) * B2(q,m,n,s) for k=1:2,m=1:2,n=1:2,p=1:2,q=1:2,s=1:2)
    AL = Nϕ = π * inv(FullDiagOp(sum(L⃗[i] * L⃗[j] * Fourier(I(i,j)) for i=1:2,j=1:2)))
    
    ϕqe_normalized = AL * ϕqe_unnormalized
    
    if Cϕ != nothing
        (Cϕ * inv(Cϕ + Nϕ) * ϕqe_normalized), Nϕ
    else
        ϕqe_normalized, Nϕ
    end
    
end

function get_L⃗_L²(f)
    # todo: turn L⃗ into a full-fledged operator like ∇
    L⃗  = (@SVector [(∇[1] .+ 0Ð(f)), (∇[2] .+ 0Ð(f))])
    L² = FullDiagOp(L⃗'L⃗)
    L⃗,L²
end
