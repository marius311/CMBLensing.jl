
# this file defines a flat-sky pixelized spin-0 map (like T or ϕ)
# and operators on this map

using BayesLensSPTpol: cls_to_cXXk

export FlatS0Fourier, FlatS0FourierDiagCov, FlatS0Map, FlatS0MapDiagCov

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

function white_noise{F<:FlatS0}(::Type{F})
    T,P = F.parameters #will be less hacky in 0.6
    FlatS0Map{F.parameters...}(randn(Nside(P),Nside(P)) / FFTgrid(T,P).Δx)
end

# define derivatives
*{T,P}(::∂Op{:x}, f::FlatS0{T,P}) = FlatS0Fourier{T,P}(im * FFTgrid(T,P).k' .* Fourier(f).Tl)
*{T,P}(::∂Op{:y}, f::FlatS0{T,P}) = FlatS0Fourier{T,P}(im * FFTgrid(T,P).k[1:Nside(P)÷2+1] .* Fourier(f).Tl)

""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov{T,P}(::Type{P}, ::Type{S0}, ℓ::Vector{T}, CℓTT::Vector{T})
    g = FFTgrid(T,P)
    LinearDiagOp(FlatS0Fourier{T,P}(cls_to_cXXk(ℓ, CℓTT, g.r)[1:g.nside÷2+1,:]))
end

# how to convert to and from vectors (when needing to feed into other algorithms)
tovec(f::FlatS0Map) = f.Tx[:]
tovec(f::FlatS0Fourier) = f.Tl[:]
fromvec{T,P}(::Type{FlatS0Map{T,P}}, vec::AbstractVector) = FlatS0Map{T,P}(reshape(vec,(Nside(P),Nside(P))))
fromvec{T,P}(::Type{FlatS0Fourier{T,P}}, vec::AbstractVector) = FlatS0Fourier{T,P}(reshape(vec,(Nside(P)÷2+1,Nside(P))))
