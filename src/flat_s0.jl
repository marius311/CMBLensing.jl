
# this file defines a flat-sky pixelized spin-0 map (like T or ϕ)
# and operators on this map

export FlatS0Fourier, FlatS0Map

abstract type Map <: Basis end
abstract type Fourier <: Basis end

struct FlatS0Map{T<:Real,P<:Flat} <: Field{P,S0,Map}
    Tx::Matrix{T}
    FlatS0Map{T,P}(Tx::Matrix) where {T,P} = new{T,P}(checkmap(P,Tx))
end

struct FlatS0Fourier{T<:Real,P<:Flat} <: Field{P,S0,Fourier}
    Tl::Matrix{Complex{T}}
    FlatS0Fourier{T,P}(Tl::Matrix) where {T,P} = new{T,P}(checkfourier(P,Tl))
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
function Cℓ_to_cov{T,P}(::Type{P}, ::Type{S0}, ℓ::Vector{T}, CℓTT::Vector{T})
    g = FFTgrid(T,P)
    FullDiagOp(FlatS0Fourier{T,P}(cls_to_cXXk(ℓ, CℓTT, g.r)[1:g.nside÷2+1,:]))
end

zero(::Union{Type{FlatS0Map{T,P}},Type{FlatS0Fourier{T,P}}}) where {T,P} = FlatS0Map{T,P}(zeros(Nside(P),Nside(P)))

# dot products
dot{T,P}(a::FlatS0Map{T,P}, b::FlatS0Map{T,P}) = (a.Tx ⋅ b.Tx) * FFTgrid(T,P).Δx^2
dot{T,P}(a::FlatS0Fourier{T,P}, b::FlatS0Fourier{T,P}) = real((a.Tl[:] ⋅ b.Tl[:]) + (a.Tl[2:Nside(P)÷2,:][:] ⋅ b.Tl[2:Nside(P)÷2,:][:])) * FFTgrid(T,P).Δℓ^2


# vector conversion
fromvec{T,P}(::Type{FlatS0Map{T,P}}, vec::AbstractVector) = FlatS0Map{T,P}(reshape(vec,(Nside(P),Nside(P))))
fromvec{T,P}(::Type{FlatS0Fourier{T,P}}, vec::AbstractVector) = FlatS0Fourier{T,P}(reshape(vec,(Nside(P)÷2+1,Nside(P))))
length{T,P}(::Type{FlatS0Map{T,P}}) = Nside(P)^2
length{T,P}(::Type{FlatS0Fourier{T,P}}) = Nside(P)*(Nside(P)÷2+1)

# plotting
function plot(f::FlatS0{T,P}; ax=nothing, kwargs...) where {T,P}
    ax == nothing ? ax = figure()[:add_subplot](111) : ax
    ax = pyimport(:seaborn)[:heatmap](f[:Tx]; xticklabels=false, yticklabels=false, square=true, kwargs...)
    Θ,N = P.parameters # until https://github.com/JuliaLang/julia/issues/21147 is fixed...
    ax[:set_title]("T map ($(N)x$(N) @ $(Θ)')")
end 

function plot(fs::AbstractVecOrMat{F}; plotsize=4, kwargs...) where {F<:FlatS0}
    figure(figsize=plotsize.*(size(fs,1),size(fs,2)))
    for i=eachindex(fs)
        plot(fs[i]; ax=subplot(size(fs,1),size(fs,2),i), kwargs...)
    end
end
