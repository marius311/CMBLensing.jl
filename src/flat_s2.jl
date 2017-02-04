
# this file defines a flat-sky pixelized spin-2 map (like  a polarization Q&U map)
# and operators on this map

abstract QUMap <: Basis
abstract EBMap <: Basis
abstract QUFourier <: Basis
abstract EBFourier <: Basis


immutable FlatS2EBMap{T<:Real,P<:Flat} <: Field{P,S2,EBMap}
    Ex::Matrix{T}
    Bx::Matrix{T}
end

immutable FlatS2EBFourier{T<:Real,P<:Flat} <: Field{P,S2,EBFourier}
    El::Matrix{T}
    Bl::Matrix{T}
end

immutable FlatS2QUMap{T<:Real,P<:Flat} <: Field{P,S2,QUMap}
    Qx::Matrix{T}
    Ux::Matrix{T}
end

immutable FlatS2QUFourier{T<:Real,P<:Flat} <: Field{P,S2,QUFourier}
    Ql::Matrix{T}
    Ul::Matrix{T}
end

# typealias FlatS2Map{T,P} Union{FlatS2QUMap{T,P}, FlatS2EBMap{T,P}}
# typealias FlatS2Fourier{T,P} Union{FlatS2QUFourier{T,P}, FlatS2EBFourier{T,P}}


QUFourier{T,P}(f::FlatS2QUMap{T,P}) = FlatS2QUFourier{T,P}(ℱ{P}*f.Qx, ℱ{P}*f.Ux)
QUFourier{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier
function QUFourier{T,P}(f::FlatS2EBFourier{T,P})
    sinϕ, cosϕ = FFTgrid(T,P).sincos2ϕ
    Ql = - El.*cosϕ + Bl.* sinϕ
    Ul = - El.*sinϕ - Bl.* cosϕ
    (Ql,Ul)
end

QUMap{T,P}(f::FlatS2QUFourier{T,P}) = FlatS2QUMap{T,P}(ℱ{P}\f.Ql, ℱ{P}\f.Ul)
QUMap{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier |> QUMap
QUMap{T,P}(f::FlatS2EBFourier{T,P}) = f |> QUFourier |> QUMap

EBFourier{T,P}(f::FlatS2EBMap{T,P}) = FlatS2EBFourier{T,P}(ℱ{P}*f.Ex, ℱ{P}*f.Bx)
EBFourier{T,P}(f::FlatS2QUMap{T,P}) = f |> QUFourier |> EBFourier
function EBFourier{T,P}(f::FlatS2QUFourier{T,P}) 
    sinϕ, cosϕ = FFTgrid(T,P).sincos2ϕ
    El = - Ql.*cosϕ - Ul.*sinϕ
    Bl =   Ql.*sinϕ - Ul.*cosϕ
    (El,Bl)
end

EBMap{T,P}(f::FlatS2EBFourier{T,P}) = FlatS2EBMap{T,P}(ℱ{P}\f.El, ℱ{P}\f.Bl)
EBMap{T,P}(f::FlatS2QUMap{T,P}) = f |> QUFourier |> EBFourier |> EBMap
EBMap{T,P}(f::FlatS2QUFourier{T,P}) = f |> EBFourier |> EBMap


# basically we always err on the side of keeping things in the EB and Fourier
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



# # define derivative operators
# ∂ops{P<:Flat,S<:S2}(::Type{P},::Type{S}) = ∂Op{:x,P,S,QUFourier}(), ∂Op{:y,P,S,QUFourier}()
# *{T,P,S}(∂x::∂Op{:x,P,S}, f::FlatS2QUFourier{T,P}) = FlatS0Fourier{T,P}(im .* FFTgrid(T,P).k' .* f.Tl)
# *{T,P,S}(∂y::∂Op{:y,P,S}, f::FlatS2QUFourier{T,P}) = FlatS0Fourier{T,P}(im .* FFTgrid(T,P).k[1:round(Int,Nside(P)/2+1)] .* f.Tl)
