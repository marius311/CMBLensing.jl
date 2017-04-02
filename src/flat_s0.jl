
# this file defines a flat-sky pixelized spin-0 map (like T or ϕ)
# and operators on this map

export FlatS0Fourier, FlatS0Map

abstract type Map <: Basis end
abstract type Fourier <: Basis end

struct FlatS0Map{T<:Real,P<:Flat} <: Field{P,S0,Map}
    Tx::Matrix{T}
    FlatS0Map{T,P}(Tx::AbstractMatrix) where {T,P} = new{T,P}(checkmap(P,Tx))
end

struct FlatS0Fourier{T<:Real,P<:Flat} <: Field{P,S0,Fourier}
    Tl::Matrix{Complex{T}}
    FlatS0Fourier{T,P}(Tl::AbstractMatrix) where {T,P} = new{T,P}(checkfourier(P,Tl))
end

const FlatS0{T,P}=Union{FlatS0Map{T,P},FlatS0Fourier{T,P}}

# convenience constructors
FlatS0Map{T}(Tx::Matrix{T},Θpix=Θpix₀) = FlatS0Map{T,Flat{Θpix,size(Tx,2)}}(Tx)
FlatS0Fourier{T}(Tl::Matrix{Complex{T}},Θpix=Θpix₀) = FlatS0Fourier{T,Flat{Θpix,size(Tl,2)}}(Tl)

# basis conversion
promote_rule{T,P}(::Type{FlatS0Map{T,P}}, ::Type{FlatS0Fourier{T,P}}) = FlatS0Map{T,P}


Fourier{T,P}(f::FlatS0Map{T,P}) = FlatS0Fourier{T,P}(ℱ{P}*f.Tx)
Map{T,P}(f::FlatS0Fourier{T,P}) = FlatS0Map{T,P}(ℱ{P}\f.Tl)


LenseBasis{F<:FlatS0}(::Type{F}) = Map

function white_noise(::Type{F}) where {Θ,Nside,T,P<:Flat{Θ,Nside},F<:FlatS0{T,P}}
    FlatS0Map{T,P}(randn(Nside,Nside) / FFTgrid(T,P).Δx)
end

""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov{T,P}(::Type{T}, ::Type{P}, ::Type{S0}, ℓ, CℓTT)
    g = FFTgrid(T,P)
    FullDiagOp(FlatS0Fourier{T,P}(Cℓ_2D(ℓ, CℓTT, g.r)[1:g.nside÷2+1,:]))
end

function get_Cℓ(f::FlatS0{T,P}; ledges=0:50:6000) where {T,P}
    g = FFTgrid(T,P)
    α = g.Δx^2/(4π^2)*g.nside^2
    power = fit(Histogram,g.r[:],WeightVec((@. g.r^2 * abs2($unfold(f[:Tl])))[:]),ledges)
    counts = fit(Histogram,g.r[:],ledges)
    h = Histogram(ledges, @. power.weights / counts.weights / (2π) / α)
    ((h.edges[1][1:end-1]+h.edges[1][2:end])/2, h.weights)
end


zero(::Union{Type{FlatS0Map{T,P}},Type{FlatS0Fourier{T,P}}}) where {T,P} = FlatS0Map{T,P}(zeros(Nside(P),Nside(P)))

# dot products
dot{T,P}(a::FlatS0Map{T,P}, b::FlatS0Map{T,P}) = (a.Tx ⋅ b.Tx) * FFTgrid(T,P).Δx^2
dot{T,P}(a::FlatS0Fourier{T,P}, b::FlatS0Fourier{T,P}) = real((a.Tl[:] ⋅ b.Tl[:]) + (a.Tl[2:Nside(P)÷2,:][:] ⋅ b.Tl[2:Nside(P)÷2,:][:])) * FFTgrid(T,P).Δℓ^2 #todo: could compute this with less ops


# vector conversion
fromvec{T,P}(::Type{FlatS0Map{T,P}}, vec::AbstractVector) = FlatS0Map{T,P}(reshape(vec,(Nside(P),Nside(P))))
fromvec{T,P}(::Type{FlatS0Fourier{T,P}}, vec::AbstractVector) = FlatS0Fourier{T,P}(reshape(vec,(Nside(P)÷2+1,Nside(P))))
length{T,P}(::Type{FlatS0Map{T,P}}) = Nside(P)^2
length{T,P}(::Type{FlatS0Fourier{T,P}}) = Nside(P)*(Nside(P)÷2+1)


# norms (for e.g. ODE integration error tolerance)
pixstd(f::FlatS0Map) = sqrt(var(f.Tx))
pixstd{T,Θ,N}(f::FlatS0Fourier{T,Flat{Θ,N}}) = sqrt(sum(2abs2(f.Tl[2:N÷2,:]))+sum(abs2(f.Tl[1,:]))) / N^2 / deg2rad(Θ/60)^2 * 2π
