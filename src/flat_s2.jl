
# this file defines a flat-sky pixelized spin-2 map (like  a polarization Q&U map)
# and operators on this map


# I'm not totally happy with having it this way, because it necessitates the
# long list of Map and Fourier definitions below. Probably needs more thoughts... 
abstract QUMap <: Basis
abstract EBMap <: Basis
abstract QUFourier <: Basis
abstract EBFourier <: Basis


immutable FlatS2EBMap{T<:Real,P<:Flat} <: Field{P,S2,EBMap}
    Ex::Matrix{T}
    Bx::Matrix{T}
end

immutable FlatS2EBFourier{T<:Real,P<:Flat} <: Field{P,S2,EBFourier}
    El::Matrix{Complex{T}}
    Bl::Matrix{Complex{T}}
end

immutable FlatS2QUMap{T<:Real,P<:Flat} <: Field{P,S2,QUMap}
    Qx::Matrix{T}
    Ux::Matrix{T}
end

immutable FlatS2QUFourier{T<:Real,P<:Flat} <: Field{P,S2,QUFourier}
    Ql::Matrix{Complex{T}}
    Ul::Matrix{Complex{T}}
end


QUFourier{T,P}(f::FlatS2QUMap{T,P}) = FlatS2QUFourier{T,P}(ℱ{P}*f.Qx, ℱ{P}*f.Ux)
QUFourier{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier
function QUFourier{T,P}(f::FlatS2EBFourier{T,P})
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    Ql = - f.El .*cos2ϕ + f.Bl .* sin2ϕ
    Ul = - f.El .*sin2ϕ - f.Bl .* cos2ϕ
    (Ql,Ul)
end

QUMap{T,P}(f::FlatS2QUFourier{T,P}) = FlatS2QUMap{T,P}(ℱ{P}\f.Ql, ℱ{P}\f.Ul)
QUMap{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier |> QUMap
QUMap{T,P}(f::FlatS2EBFourier{T,P}) = f |> QUFourier |> QUMap

EBFourier{T,P}(f::FlatS2EBMap{T,P}) = FlatS2EBFourier{T,P}(ℱ{P}*f.Ex, ℱ{P}*f.Bx)
EBFourier{T,P}(f::FlatS2QUMap{T,P}) = f |> QUFourier |> EBFourier
function EBFourier{T,P}(f::FlatS2QUFourier{T,P}) 
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    El = - Ql.*cos2ϕ - Ul.*sin2ϕ
    Bl =   Ql.*sin2ϕ - Ul.*cos2ϕ
    (El,Bl)
end

EBMap{T,P}(f::FlatS2EBFourier{T,P}) = FlatS2EBMap{T,P}(ℱ{P}\f.El, ℱ{P}\f.Bl)
EBMap{T,P}(f::FlatS2QUMap{T,P}) = f |> QUFourier |> EBFourier |> EBMap
EBMap{T,P}(f::FlatS2QUFourier{T,P}) = f |> EBFourier |> EBMap

# don't like these:
Map(f::FlatS2QUMap) = f
Map(f::FlatS2QUFourier) = QUMap(f)
Map(f::FlatS2EBMap) = f
Map(f::FlatS2EBFourier) = EBMap(f)
Fourier(f::FlatS2QUMap) = QUFourier(f)
Fourier(f::FlatS2QUFourier) = f
Fourier(f::FlatS2EBMap) = EBFourier(f)
Fourier(f::FlatS2EBFourier) = f


# basically we always err on the side of keeping things in the EB and Fourier
# (TODO: probably QU and Fourier or even QU and Map might make more sense...)
rules = Dict(
    (FlatS2QUMap,     FlatS2QUFourier)  => FlatS2QUFourier,
    (FlatS2EBMap,     FlatS2EBFourier)  => FlatS2EBFourier,
    (FlatS2QUMap,     FlatS2EBMap)      => FlatS2EBMap,
    (FlatS2QUFourier, FlatS2EBFourier)  => FlatS2EBFourier,
    (FlatS2QUMap,     FlatS2EBFourier)  => FlatS2EBFourier,
    (FlatS2QUFourier, FlatS2EBMap)      => FlatS2QUFourier
)

for ((F1,F2),Tout) in rules
    @eval @swappable promote_type{T,P}(::Type{$F1{T,P}},::Type{$F2{T,P}})=$Tout{T,P}
end

""" A covariance of a spin-2 flat sky map which is diagonal in pixel space"""
immutable FlatS2EBFourierDiagCov{T<:Real,P<:Flat} <: LinearFieldDiagOp{P,S2,EBFourier}
    CEEl::Matrix{Complex{T}}
    CBBl::Matrix{Complex{T}}
end
*{T,P}(Σ::FlatS2EBFourierDiagCov{T,P}, f::FlatS2EBFourier{T,P}) = FlatS2EBFourier{T,P}(Σ.CEEl .* f.El, Σ.CBBl .* f.Bl)
simulate{T,P}(Σ::FlatS2EBFourierDiagCov{T,P}) = FlatS2EBFourier{T,P}(ℱ{P} * randn(Nside(P),Nside(P)) .* √Σ.CEEl / FFTgrid(T,P).Δx, ℱ{P} * randn(Nside(P),Nside(P)) .* √Σ.CBBl / FFTgrid(T,P).Δx)


function Cℓ_to_cov{T,P}(::Type{FlatS2EBFourierDiagCov{T,P}}, ℓ, CℓEE, CℓBB)
    g = FFTgrid(T,P)
    FlatS2EBFourierDiagCov{T,P}(
        cls_to_cXXk(ℓ, CℓEE, g.r)[1:round(Int,g.nside/2)+1,:],
        cls_to_cXXk(ℓ, CℓBB, g.r)[1:round(Int,g.nside/2)+1,:]
    )
end
