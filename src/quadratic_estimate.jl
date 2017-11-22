export ϕqeTT

sinv(x) = nan2zero.(1./x)

"""Compute quadratic estimate for ϕ given TT data, d, and signal, lensed signal, and noise covariances Cf, Cf̃, and Cn"""
function ϕqeTT(d::FlatS0, Cf, Cf̃, Cn, Cϕ=nothing; WF=false)

    L̅  = (@SVector [∇[1] .+ 0Ð(d), ∇[2] .+ 0Ð(d)]);
    L² = FullDiagOp(L̅'L̅)

    # quadratic estimate
    dL = L̅' * Fourier(Map(inv(Cf̃+Cn)*d) * Map(L̅*(Cf*inv(Cf̃+Cn)*d)))
    
    # normalization
    term1 = Map((Cf^2)*inv(Cf̃+Cn).*(L̅*L̅')) .* Map(  sinv(Cf̃+Cn).f )
    term2 = Map( Cf   *inv(Cf̃+Cn).* L̅    ) .* Map(Cf*inv(Cf̃+Cn).*L̅)'
    AL = 2π * L² * sinv(L̅' * Fourier(term1 + term2) * L̅)
    
    ϕest = -AL*(L²\dL)
    
    if WF
        @assert Cϕ != nothing "Need to provide Cϕ if WF=true"
        Nϕ = FullDiagOp(abs.(AL*inv(L²)))
        Cϕ * inv(Cϕ + Nϕ) * ϕest
    else
        ϕest
    end

end
