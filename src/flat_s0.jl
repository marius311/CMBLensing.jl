
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
immutable FlatS0MapDiagCov{T<:Real,P<:Flat} <: FieldCov{P,S0,Map}
    Cx::Matrix{T}
end
*{T,P}(Σ::FlatS0MapDiagCov{T,P}, f::FlatS0Map{T,P}) = FlatS0Map{T,P}(f.Tx .* Σ.Cx)
simulate{T,P}(Σ::FlatS0MapDiagCov{T,P}) = FlatS0Map{T,P}(randn(Nside(P),Nside(P)) .* √Σ.Cx)


""" A covariance of a spin-0 flat sky map which is diagonal in pixel space"""
immutable FlatS0FourierDiagCov{T<:Real,P<:Flat} <: FieldCov{P,S0,Fourier}
    Cl::Matrix{Complex{T}}
end
*{T,P}(Σ::FlatS0FourierDiagCov{T,P}, f::FlatS0Fourier{T,P}) = FlatS0Fourier{T,P}(f.Tl .* Σ.Cl)
simulate{T,P}(Σ::FlatS0FourierDiagCov{T,P}) = FlatS0Fourier{T,P}(ℱ{P} * randn(Nside(P),Nside(P)) .* √Σ.Cl / FFTgrid(T,P).Δx)


""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov{T,P}(::Type{FlatS0FourierDiagCov{T,P}}, ℓ, CℓTT)
    g = FFTgrid(T,P)
    FlatS0FourierDiagCov{T,P}(complex(cls_to_cXXk(ℓ, CℓTT, g.r))[1:round(Int,g.nside/2)+1,:])
end

# Can raise these guys to powers explicitly since they're diagonal
^{T<:Union{FlatS0MapDiagCov,FlatS0FourierDiagCov}}(f::T, n::Number) = T(map(.^,data(f),repeated(n))..., meta(f)...)

# how to convert to and from vectors (when needing to feed into other algorithms)
tovec(f::FlatS0Map) = f.Tx[:]
tovec(f::FlatS0Fourier) = f.Tl[:]
fromvec{T,P}(::Type{FlatS0Map{T,P}}, vec::AbstractVector) = FlatS0Map{T,P}(reshape(vec,(Nside(P),Nside(P))))
fromvec{T,P}(::Type{FlatS0Fourier{T,P}}, vec::AbstractVector) = FlatS0Fourier{T,P}(reshape(vec,(round(Int,Nside(P)/2+1),Nside(P))))

# size{T,P}(f::FlatS0MapDiagCov{T,P}) = (Nside(P)^2,Nside(P)^2)
# size{T,P}(f::FlatS0FourierDiagCov{T,P}) = fill(,2)


immutable FlatS0LensingOp{T<:Real,P<:Flat} <: LinearFieldOp{P,S0,Map}
    # pixel remapping
    i::Matrix{Int}
    j::Matrix{Int}

    # residual displacement
    rx::Matrix{T}
    ry::Matrix{T}

    # precomputed quantities
    kα::Dict{Any,Matrix{Complex{T}}}
    xα::Dict{Any,Matrix{T}}

    order::Int
    taylens::Bool
end

function FlatS0LensingOp{T,P}(ϕ::FlatS0{T,P}; order=4, taylens=true)

    g = FFTgrid(T,P)
    Nside = g.nside
    
    # total displacement
    dx, dy = LenseDecomp_helper1(ϕ[:Tl], zeros(Complex{T},(Nside,Nside)), g);

    # nearest pixel displacement
    if taylens
        di, dj = (round(Int,d/g.deltx) for d=(dx,dy)) # end
        i = indexwrap.(di .+ (1:Nside)', Nside)
        j = indexwrap.(dj .+ (1:Nside) , Nside)
    else
        di = dj = i = j = zeros(Int,Nside,Nside)
    end

    # residual displacement
    rx, ry = ((d - i.*g.deltx) for (d,i)=[(dx,di),(dy,dj)]) # end

    # precomputation
    kα = Dict{Any,Matrix{Complex{T}}}()
    xα = Dict{Any,Matrix{T}}()
    for n in 1:order, α₁ in 0:n
        kα[n,α₁] = im ^ n .* g.k[1] .^ α₁ .* g.k[2] .^ (n - α₁)
        xα[n,α₁] = rx .^ α₁ .* ry .^ (n - α₁) ./ factorial(α₁) ./ factorial(n - α₁)
    end

    FlatS0LensingOp{T,P}(i,j,rx,ry,kα,xα,order,taylens)
end

# our implementation of Taylens
function *{T,P}(lens::FlatS0LensingOp, f::FlatS0Map{T,P})

    intlense(fx) = lens.taylens ? broadcast_getindex(fx, lens.j, lens.i) : fx
    fl = f[:Tl]

    # lens to the nearest whole pixel
    Lfx = intlense(f.Tx)

    # add in Taylor series correction
    for n in 1:lens.order, α₁ in 0:n
        Lfx .+= lens.xα[n,α₁] .* intlense(ℱ{P} \ (lens.kα[n,α₁] .* fl))
    end

    FlatS0Map{T,P}(Lfx,meta(f)...)
end
