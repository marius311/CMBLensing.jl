
# this file defines a flat-sky pixelized spin-0 map (like T or ϕ)
# and operators on this map

using BayesLensSPTpol: cls_to_cXXk

abstract Map <: Basis
abstract Fourier <: Basis


immutable FlatS0Map{T<:Real,P<:Flat} <: Field{P,S0,Map}
    Tx::Matrix{T}
end

immutable FlatS0Fourier{T<:Real,P<:Flat} <: Field{P,S0,Fourier}
    Tl::Matrix{Complex{T}}
end

typealias FlatS0{T,P} Union{FlatS0Map{T,P},FlatS0Fourier{T,P}}

Fourier{T,P}(f::FlatS0Map{T,P}) = FlatS0Fourier{T,P}(ℱ{P}*f.Tx)
Map{T,P}(f::FlatS0Fourier{T,P}) = FlatS0Map{T,P}(ℱ{P}\f.Tl)

@swappable promote_type{T,P}(::Type{FlatS0Map{T,P}}, ::Type{FlatS0Fourier{T,P}}) = FlatS0Map{T,P}


""" A covariance of a spin-0 flat sky map which is diagonal in pixel space"""
immutable FlatS0MapDiagCov{T<:Real,P<:Flat} <: LinearFieldDiagOp{P,S0,Map}
    Cx::Matrix{T}
end
*{T,P}(Σ::FlatS0MapDiagCov{T,P}, f::FlatS0Map{T,P}) = FlatS0Map{T,P}(Σ.Cx .* f.Tx)
simulate{T,P}(Σ::FlatS0MapDiagCov{T,P}) = FlatS0Map{T,P}(randn(Nside(P),Nside(P)) .* √Σ.Cx)


""" A covariance of a spin-0 flat sky map which is diagonal in pixel space"""
immutable FlatS0FourierDiagCov{T<:Real,P<:Flat} <: LinearFieldDiagOp{P,S0,Fourier}
    Cl::Matrix{Complex{T}}
end
*{T,P}(Σ::FlatS0FourierDiagCov{T,P}, f::FlatS0Fourier{T,P}) = FlatS0Fourier{T,P}(Σ.Cl .* f.Tl)
simulate{T,P}(Σ::FlatS0FourierDiagCov{T,P}) = FlatS0Fourier{T,P}(ℱ{P} * randn(Nside(P),Nside(P)) .* √Σ.Cl / FFTgrid(T,P).Δx)

# define derivates
*{T,P}(::∂Op{:x}, f::FlatS0{T,P}) = FlatS0Fourier{T,P}(im * FFTgrid(T,P).k' .* Fourier(f).Tl)
*{T,P}(::∂Op{:y}, f::FlatS0{T,P}) = FlatS0Fourier{T,P}(im * FFTgrid(T,P).k[1:round(Int,Nside(P)/2+1)] .* Fourier(f).Tl)

""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov{T,P}(::Type{FlatS0FourierDiagCov{T,P}}, ℓ, CℓTT)
    g = FFTgrid(T,P)
    FlatS0FourierDiagCov{T,P}(complex(cls_to_cXXk(ℓ, CℓTT, g.r))[1:round(Int,g.nside/2)+1,:])
end


# how to convert to and from vectors (when needing to feed into other algorithms)
tovec(f::FlatS0Map) = f.Tx[:]
tovec(f::FlatS0Fourier) = f.Tl[:]
fromvec{T,P}(::Type{FlatS0Map{T,P}}, vec::AbstractVector) = FlatS0Map{T,P}(reshape(vec,(Nside(P),Nside(P))))
fromvec{T,P}(::Type{FlatS0Fourier{T,P}}, vec::AbstractVector) = FlatS0Fourier{T,P}(reshape(vec,(round(Int,Nside(P)/2+1),Nside(P))))
