module CMBFields

using BayesLensSPTpol
using BayesLensSPTpol: LenseDecomp_helper1, indexwrap
using DataArrays: @swappable
import Base: +, -, *, \, /, ^, ~, .*, ./, getindex, size, eltype


export Cℓ_to_cov, Field, FlatS0Fourier, FlatS0FourierDiagCov, FlatS0Map, FlatS0MapDiagCov, Map, simulate

""" 
A field with a particular pixelization scheme, spin, and basis.
"""
abstract Field{Pix, Spin, Basis}


""" 
A linear operator acting on a field with particular pixelization scheme and spin, and 
which can only be applied in a given basis (i.e. the one in which its diagonal)
"""
abstract LinearFieldOp{Pix, Spin, Basis} 

"""
Covariances are linear operators on the fields, and some algebra (like adding or subtracting
two of them in the same basis) can be done expclitly. 
"""
abstract FieldCov{Pix, Spin, Basis} <: LinearFieldOp{Pix, Spin, Basis}


abstract Pix 
abstract Flat <: Pix 
abstract Healpix <: Pix

abstract Spin
abstract S0 <: Spin
abstract S1 <: Spin
abstract S2 <: Spin
abstract S02 <: Spin

abstract Basis
abstract Map <: Basis
abstract Fourier <: Basis
# abstract Both <: Basis


"""
ℱ * f and ℱ \ f converts fields between map and fourier space
(mostly this is done automatically so you don't ever use this explicitly)
"""
abstract ℱ

""" Gradient of a S0 field gives you an S1 field """
abstract ∇


# """
# Stores a field where we have computed it both in map and fourier space
# """
# immutable FieldBoth{P,S} <: Field{P,S,Both}
#     map::Field{P,S,Map}
#     fourier::Field{P,S,Fourier}
# end
# meta(f::FieldBoth) = tuple()
# data(f::FieldBoth) = (f.map, f.fourier)



# ---------------
# Flat sky spin-0 
# ---------------

""" A flat sky spin-0 map (like T or ϕ) """
immutable FlatS0Map{T<:Real} <: Field{Flat,S0,Map}
    Tx::Matrix{T}
    g::FFTgrid
end

""" The fourier transform of a flat sky spin-0 map """
immutable FlatS0Fourier{T<:Real} <: Field{Flat,S0,Fourier}
    Tl::Matrix{Complex{T}}
    g::FFTgrid
end

meta(f::Union{FlatS0Map,FlatS0Fourier}) = (f.g,)
data(f::FlatS0Map) = (f.Tx,)
data(f::FlatS0Fourier) = (f.Tl,)
ℱ(f::FlatS0Map) = FlatS0Fourier(f.g.FFT * f.Tx, meta(f)...)
iℱ(f::FlatS0Fourier) = FlatS0Map(real(f.g.FFT \ f.Tl), meta(f)...)


""" A covariance which is diagonal in pixel space of a spin-0 flat sky map """
immutable FlatS0MapDiagCov{T<:Real} <: FieldCov{Flat,S0,Map}
    Cx::Matrix{T}
    g::FFTgrid
end
meta(Σ::FlatS0MapDiagCov) = (Σ.g,)
data(Σ::FlatS0MapDiagCov) = (Σ.Cx,)
*{T}(Σ::FlatS0MapDiagCov{T}, f::FlatS0Map{T}) = FlatS0Map{T}(f.Tx .* Σ.Cx, meta(f)...)
simulate(Σ::FlatS0MapDiagCov) = FlatS0Map(randn(Σ.g.nside,Σ.g.nside) .* √Σ.Cx, meta(Σ)...)


""" A covariance which is diagonal in pixel space of a spin-0 flat sky map """
immutable FlatS0FourierDiagCov{T<:Real} <: FieldCov{Flat,S0,Fourier}
    Cl::Matrix{Complex{T}}
    g::FFTgrid
end
meta(Σ::FlatS0FourierDiagCov) = (Σ.g,)
data(Σ::FlatS0FourierDiagCov) = (Σ.Cl,)
*{T}(Σ::FlatS0FourierDiagCov{T}, f::FlatS0Fourier{T}) = FlatS0Fourier(f.Tl .* Σ.Cl, meta(f)...)
simulate(Σ::FlatS0FourierDiagCov) = FlatS0Fourier(Σ.g.FFT * randn(Σ.g.nside,Σ.g.nside) .* √Σ.Cl / Σ.g.deltx, meta(Σ)...)


""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov(::Type{FlatS0FourierDiagCov}, ℓ, CℓTT, g::FFTgrid)
    FlatS0FourierDiagCov(complex(BayesLensSPTpol.cls_to_cXXk(ℓ, CℓTT, g.r)), g)
end

# Can raise these guys to powers explicitly since they're diagonal
^{T<:Union{FlatS0MapDiagCov,FlatS0FourierDiagCov}}(f::T, n::Number) = T(map(.^,data(f),repeated(n))..., meta(f)...)

# how to convert to and from vectors (when needing to feed into other algorithms)
tovec(f::FlatS0Map) = f.Tx[:] 
tovec(f::FlatS0Fourier) = f.Tl[:] 
fromvec{T<:Union{FlatS0Map,FlatS0Fourier}}(::Type{T}, vec::AbstractVector, g) = T(reshape(vec,(g.nside,g.nside)),g)
eltype{T}(::Type{FlatS0Map{T}}) = T
eltype{T}(::Type{FlatS0Fourier{T}}) = Complex{T}
size(f::Union{FlatS0MapDiagCov,FlatS0FourierDiagCov}) = (f.g.nside^2, f.g.nside^2)


immutable FlatS0LensingOp{T<:Real} <: LinearFieldOp{Flat,S0,Map}
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

FlatS0LensingOp{B}(ϕ::Field{Flat,S0,B}; kwargs...) = FlatS0LensingOp(∇(ϕ); kwargs...)

function FlatS0LensingOp{B}(d::FlatS1Map; order=4, taylens=true)
    g = ϕ.g
    nside = g.nside
    
    # total displacement
    dx, dy = d.Tx1, d.Tx2
    
    # nearest pixel displacement
    if taylens
        di, dj = (round(Int,d/g.deltx) for d=(dx,dy))
        i = indexwrap.(di .+ (1:nside)', nside)
        j = indexwrap.(dj .+ (1:nside) , nside)
    else
        di = dj = i = j = zeros(Int,nside,nside)
    end
    
    # residual displacement
    rx, ry = ((d - i.*g.deltx) for (d,i)=[(dx,di),(dy,dj)])
            
    # precomputation
    T = eltype(ϕ[:Tx])
    kα = Dict{Any,Matrix{Complex{T}}}()
    xα = Dict{Any,Matrix{T}}()
    for n in 1:order, α₁ in 0:n
        kα[n,α₁] = im ^ n .* g.k[1] .^ α₁ .* g.k[2] .^ (n - α₁)
        xα[n,α₁] = rx .^ α₁ .* ry .^ (n - α₁) ./ factorial(α₁) ./ factorial(n - α₁)
    end
            
    FlatS0LensingOp(i,j,rx,ry,kα,xα,order,taylens)
end

# our implementation of Taylens
function *{T<:Field{Flat,S0,Map}}(lens::FlatS0LensingOp, f::T)
    
    intlense(fx) = lens.taylens ? broadcast_getindex(fx, lens.j, lens.i) : fx
    fl = f[:Tl]
    g = f.g
    
    # lens to the nearest whole pixel
    fx˜ = intlense(f[:Tx])

    # add in Taylor series correction
    for n in 1:lens.order, α₁ in 0:n
        fx˜ .+= lens.xα[n,α₁] .* intlense(real(g.FFT \ (lens.kα[n,α₁] .* fl)))
    end

    T(fx˜,meta(f)...)
end



# ---------------
# Flat sky spin-2 
# ---------------

""" A flat sky spin-2 map (like E/B) """
immutable FlatS2Map{T<:Real} <: Field{Flat,S2,Map}
    Tx::Matrix{T}
    Ex::Matrix{T}
    Bx::Matrix{T}
    g::FFTgrid
end

# etc...


# ------------------------------
# Flat sky spin-1 (vector) field 
# ------------------------------

immutable FlatS1Map{T<:Real} <: Field{Flat,S1,Map}
    Tx1::Matrix{T}
    Tx2::Matrix{T}
    g::FFTgrid
end
data(f::FlatS1Map) = (f.Tx1,f.Tx2)
meta(f::FlatS1Map) = (f.g,)
(::Type{∇})(f::FlatS0Fourier) = FlatS1Map((real(f.g.FFT \ (im * f.g.k[i] .* f.Tl)) for i=1:2)..., f.g)
(::Type{∇})(f::FlatS0Map) = ∇(Fourier(f))
*(op::FlatS0LensingOp, f::FlatS1Map) = FlatS1Map((op*FlatS0Map(f.Tx1,g))[:Tx], (op*FlatS0Map(f.Tx2,g))[:Tx], f.g)







# ======================================
# algebra with Fields and LinearFieldOps
# ======================================


# addition and subtraction of two Fields or FieldCovs
for op in (:+, :-) 
    @eval @swappable ($op){P,S}(a::Field{P,S,Map}, b::Field{P,S,Fourier}) = ($op)(a, Map(b))
    # @eval @swappable ($op){P,S}(a::Field{P,S,Map}, b::Field{P,S,Both}) = ($op)(a, b.map)
    # @eval @swappable ($op){P,S}(a::Field{P,S,Fourier}, b::Field{P,S,Both}) = ($op)(a, b.fourier)
    for F in (Field,FieldCov)
        @eval ($op){T<:($F)}(a::T, b::T) = T(map($op,map(data,(a,b))...)..., meta(a)...) 
    end
end

# element-wise multiplication or division of two Fields
for op in (:.*, :./) 
    @eval ($op){T<:Field}(a::T, b::T) = T(map($op,map(data,(a,b))...)..., meta(a)...) 
end

# ops with a Field or FieldCov and a scalar
for op in (:+, :-, :*, :/), F in (Field,FieldCov)
    @eval ($op){T<:($F)}(f::T, n::Number) = T(map($op,data(f),repeated(n))..., meta(f)...)
    @eval ($op){T<:($F)}(n::Number, f::T) = T(map($op,repeated(n),data(f))..., meta(f)...)
end

# ℱ \ f and ℱ * f compute (inv)transform of a Field (and store along with original)
# *{P,S}(op::Type{ℱ}, f::Field{P,S,Map}) = FieldBoth(f,ℱ(f))
# \{P,S}(op::Type{ℱ}, f::Field{P,S,Fourier}) = FieldBoth(iℱ(f),f)

(::Type{Map}){P,S}(f::Field{P,S,Map}) = f
(::Type{Map}){P,S}(f::Field{P,S,Fourier}) = iℱ(f)
(::Type{Fourier}){P,S}(f::Field{P,S,Map}) = ℱ(f)
(::Type{Fourier}){P,S}(f::Field{P,S,Fourier}) = f


# convert Fields to right basis before feeding into a LinearFieldOp
*{P,S}(op::LinearFieldOp{P,S,Map}, f::Field{P,S,Fourier}) = op * Map(f)
# *{P,S}(op::LinearFieldOp{P,S,Map}, f::Field{P,S,Both}) = op * f.map
*{P,S}(op::LinearFieldOp{P,S,Fourier}, f::Field{P,S,Map}) = op * Fourier(f)
# *{P,S}(op::LinearFieldOp{P,S,Fourier}, f::Field{P,S,Both}) = op * f.fourier




# allow composition of LinearFieldOps
immutable LazyBinaryOp{Op} <: LinearFieldOp
    a::Union{LinearFieldOp,Number}
    b::Union{LinearFieldOp,Number}
    function LazyBinaryOp(a::LinearFieldOp,b::LinearFieldOp)
        # @assert meta(a)==meta(b) "Can't '$Op' two operators with different metadata"
        new(a,b)
    end
    @swappable LazyBinaryOp(op::LinearFieldOp, n::Number) = new(op,n)
end

## construct them with operators
for op in (:+, :-, :*) 
    @eval ($op)(a::LinearFieldOp, b::LinearFieldOp) = LazyBinaryOp{$op}(a,b) 
    @eval @swappable ($op)(a::LinearFieldOp, b::Number) = LazyBinaryOp{$op}(a,b) 
end
/(op::LinearFieldOp, n::Number) = LazyBinaryOp{/}(op,n)

## evaluation rules when applying them
for op in (:+, :-) 
    @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
end
*(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
*(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)

## getting metadata
linop(lz::LazyBinaryOp) = isa(lz.a, LinearFieldOp) ? lz.a : lz.b
meta(lz::LazyBinaryOp) = meta(linop(lz))
size(lz::LazyBinaryOp) = size(linop(lz))



# allow converting Fields and LinearFieldOp to/from vectors
getindex(f::Field, i::Colon) = tovec(f)
getindex{P,S,B}(op::LinearFieldOp, s::Tuple{Field{P,S,B}}) = LazyVecApply{typeof(s[1]),B}(op)
immutable LazyVecApply{T,B} 
    op::LinearFieldOp 
end
*{T,B}(lazy::LazyVecApply{T,B}, vec::AbstractVector) = tovec(B(lazy.op*fromvec(T,vec,meta(lazy.op)...)))
~(f::Field) = (f,)

eltype{T}(lz::LazyVecApply{T}) = eltype(T)
size(lz::LazyVecApply) = size(lz.op)
size(f::Union{LinearFieldOp,LazyVecApply}, d) = (d<=2 ? size(f)[d] : 1)
getindex{P,S,B}(vec::AbstractVector, s::Tuple{Field{P,S,B}}) = fromvec(typeof(s[1]),vec,meta(s[1])...)


# automatically convert between map and fourier when getting fields
for (B1,B2) in [(Map,Fourier),(Fourier,Map)]
    @eval getindex{P,S}(f::Field{P,S,($B1)}, s::Symbol) = s in fieldnames(f) ? getfield(f, s) : getfield($B2(f),s)
end



end
