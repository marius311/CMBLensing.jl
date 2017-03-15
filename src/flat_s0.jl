
# this file defines a flat-sky pixelized spin-0 map (like T or ϕ)
# and operators on this map

using BayesLensSPTpol: cls_to_cXXk

export FlatS0Fourier, FlatS0Map

abstract type Map <: Basis end
abstract type Fourier <: Basis end

struct FlatS0Map{T<:Real,P<:Flat} <: Field{P,S0,Map}
    Tx::Matrix{T}
    FlatS0Map{T,P}(Tx) where {T,P} = new(checkmap(P,Tx))
end

struct FlatS0Fourier{T<:Real,P<:Flat} <: Field{P,S0,Fourier}
    Tl::Matrix{Complex{T}}
    FlatS0Fourier{T,P}(Tl) where {T,P} = new(checkfourier(P,Tl))
end

const FlatS0{T,P}=Union{FlatS0Map{T,P},FlatS0Fourier{T,P}}

# convenience constructors
FlatS0Map{T}(Tx::Matrix{T},Θpix=Θpix₀) = FlatS0Map{T,Flat{Θpix,size(Tx,2)}}(Tx)
FlatS0Fourier{T}(Tl::Matrix{Complex{T}},Θpix=Θpix₀) = FlatS0Fourier{T,Flat{Θpix,size(Tl,2)}}(Tl)

# basis conversion
import Base: promote_rule
promote_rule{T,P}(::Type{FlatS0Map{T,P}}, ::Type{FlatS0Fourier{T,P}}) = FlatS0Map{T,P}


@swappable promote_type{T,P}(::Type{FlatS0Map{T,P}}, ::Type{FlatS0Fourier{T,P}}) = FlatS0Map{T,P}
Fourier{T,P}(f::FlatS0Map{T,P}) = FlatS0Fourier{T,P}(ℱ{P}*f.Tx)
Map{T,P}(f::FlatS0Fourier{T,P}) = FlatS0Map{T,P}(ℱ{P}\f.Tl)


LenseBasis{F<:FlatS0}(::Type{F}) = Map

function white_noise(::Type{F}) where {Θ,Nside,T,P<:Flat{Θ,Nside},F<:FlatS0{T,P}}
    FlatS0Map{T,P}(randn(Nside,Nside) / FFTgrid(T,P).Δx)
end

""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov{T,P}(::Type{P}, ::Type{S0}, ℓ::Vector{T}, CℓTT::Vector{T})
    g = FFTgrid(T,P)
    FullDiagOp(FlatS0Fourier{T,P}(cls_to_cXXk(ℓ, CℓTT, g.r)[1:g.nside÷2+1,:]))
end

zero{F<:FlatS0}(::Type{F}) = ((T,P)=F.parameters; FlatS0Map{T,P}(zeros(Nside(P),Nside(P))))

# dot products
dot{T,P}(a::FlatS0Map{T,P}, b::FlatS0Map{T,P}) = (a.Tx ⋅ b.Tx) * FFTgrid(T,P).Δx^2
dot{T,P}(a::FlatS0Fourier{T,P}, b::FlatS0Fourier{T,P}) = real((a.Tl[:] ⋅ b.Tl[:]) + (a.Tl[2:Nside(P)÷2,:][:] ⋅ b.Tl[2:Nside(P)÷2,:][:])) * FFTgrid(T,P).Δℓ^2


# vector conversion
tovec(f::FlatS0Map) = f.Tx[:]
tovec(f::FlatS0Fourier) = f.Tl[:]
fromvec{T,P}(::Type{FlatS0Map{T,P}}, vec::AbstractVector) = FlatS0Map{T,P}(reshape(vec,(Nside(P),Nside(P))))
fromvec{T,P}(::Type{FlatS0Fourier{T,P}}, vec::AbstractVector) = FlatS0Fourier{T,P}(reshape(vec,(Nside(P)÷2+1,Nside(P))))
eltype{T,P}(::Type{FlatS0Map{T,P}}) = T
eltype{T,P}(::Type{FlatS0Fourier{T,P}}) = Complex{T}
length{T,P}(::Type{FlatS0Map{T,P}}) = Nside(P)^2
length{T,P}(::Type{FlatS0Fourier{T,P}}) = Nside(P)*(Nside(P)÷2+1)

# plotting
function plot{T,P}(f::FlatS0{T,P}; ax=nothing, kwargs...)
    ax == nothing ? ax = figure()[:add_subplot](111) : ax
    m = ax[:matshow](f[:Tx]; kwargs...)
    Θpix,nside = P.parameters
    ax[:set_title]("$(nside)x$(nside) flat $T map at $(Θpix)' resolution")
    colorbar(m,ax=ax)
end

function plot{F<:FlatS0}(fs::AbstractVecOrMat{F}; kwargs...)
    figure()
    for i=eachindex(fs)
        plot(fs[i]; ax=subplot(size(fs,1),size(fs,2),i), kwargs...)
    end
end
