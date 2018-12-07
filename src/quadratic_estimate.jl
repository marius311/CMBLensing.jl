export ϕqe

sinv(x) = nan2zero.(1 ./ x)

macro sym(ex)
    # todo: write this such that it converts a function f(i,j,k,..) which is
    # assumed to be symmetric in its arguments into a memoized function such
    # that each unique set of parameters is only calculated once
    :($(esc(ex)))
end



"""

    ϕqe(d::FlatS0, Cf, Cf̃, Cn, Cϕ=nothing)

Compute quadratic estimate for ϕ given data, d, and signal, lensed signal,
and noise covariances Cf, Cf̃, and Cn, respectively. If signal covariance Cϕ is
given, the result is Wiener filtered.
"""
function ϕqe(d::FlatS0, Cf, Cf̃, Cn, Cϕ=nothing)

    L̅,L² = get_L̅_L²(d)

    # quadratic estimate
    ∫d²l = L̅' * Fourier(Map(inv(Cf̃+Cn)*d) * Map(L̅*(Cf*inv(Cf̃+Cn)*d)))
    
    # normalization
    term1 = Map((Cf.^2)*inv(Cf̃+Cn).*(L̅*L̅')) .* Map(  sinv(Cf̃+Cn).f )
    term2 = Map( Cf    *inv(Cf̃+Cn).* L̅    ) .* Map(Cf*inv(Cf̃+Cn).*L̅)'
    AL = 2π * L² * sinv(L̅' * Fourier(term1 + term2) * L̅)
    
    ϕL = -AL*(L²\∫d²l)
    Nϕ = FullDiagOp(abs.(inv(L²)*AL))
    
    if Cϕ != nothing
        Cϕ * inv(Cϕ + Nϕ) * ϕL
    else
        ϕL, Nϕ
    end

end

# EB estimator, currently assuming B=0 (todo: remove this assumption)
function ϕqe(d::FlatS2{T,P}, Cf, Cf̃, Cn, Cϕ=nothing) where {T,P}

    L̅,L² = get_L̅_L²(d[:E])

    ϵ(x...) = levicivita([x...])
    
    CE,CB   = Cf[:E], Cf[:B]
    CẼ,CB̃   = Cf̃[:E], Cf̃[:B]
    CEn,CBn = Cn[:E], Cn[:B]

    # quadratic estimate
    @sym E(i,j,k) = Map(L²\L̅[i]*L̅[j]*L̅[k]*(CE*inv(CẼ+CEn)*d[:E]))
    @sym B(i,j)   = Map(L²\L̅[i]*L̅[j]        *(inv(CB̃+CBn)*d[:B]))
    ∫d²l = sum(L̅[i] * Fourier(sum(ϵ(k,m,3) * E(i,j,k) * B(j,m) for j=1:2,k=1:2,m=1:2)) for i=1:2)
    
    # normalization
    @sym E2(i,j,k,m) = Map(L²\L̅[i]*L̅[j]*L̅[k]*L̅[m]*(CE.^2*inv(CẼ+CEn)))
    @sym B2(i,j)     = Map(L²\L̅[i]*L̅[j]                *(inv(CB̃+CBn)))
    AL = π * L² * sinv(sum(L̅[i]*L̅[j] * Fourier(sum(ϵ(k,n,3) * ϵ(m,p,3) * E2(i,j,k,m) * B2(n,p) for k=1:2,m=1:2,n=1:2,p=1:2)) for i=1:2,j=1:2))
    
    ϕL = AL*(L²\∫d²l)
    Nϕ = FullDiagOp(abs.(AL*inv(L²)))
    
    if Cϕ != nothing
        (Cϕ * inv(Cϕ + Nϕ) * ϕL), ((1+Nϕ\Cϕ).^2)\(Nϕ\Cϕ)
    else
        ϕL, Nϕ
    end
    
end

function get_L̅_L²(f)
    # todo: make L automatically broadcast generically like ∇ does
    L̅  = (@SVector [∇[1] .+ 0Ð(f), ∇[2] .+ 0Ð(f)])
    L² = FullDiagOp(L̅'L̅)
    L̅,L²
end
