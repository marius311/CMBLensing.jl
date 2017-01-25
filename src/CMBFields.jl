module CMBFields

using Util
using BayesLensSPTpol
using BayesLensSPTpol: LenseDecomp_helper1, indexwrap
import BayesLensSPTpol: FFTgrid
using DataArrays: @swappable
import Base: +, -, *, \, /, ^, ~, .*, ./, getindex, size, eltype


export Cℓ_to_cov, Field, FlatS0Fourier, FlatS0FourierDiagCov, FlatS0Map, FlatS0MapDiagCov, Map, simulate

abstract Pix

# a flat sky pixelization with `Nside` pixels per side and pixels of width `Θpix` arcmins 
abstract Flat{Θpix,Nside} <: Pix
Nside{T<:Flat}(::Type{T}) = T.parameters[2] #convenience method, will look less hacky in 0.6

# Healpix pixelization with particular `Nside` value
abstract Healpix{Nside<:Int} <: Pix

abstract Spin
abstract S0 <: Spin
abstract S2 <: Spin
abstract S02 <: Spin

abstract Basis
abstract Map <: Basis
abstract Fourier <: Basis


"""
A field with a particular pixelization scheme, spin, and basis.
"""
abstract Field{P<:Pix, S<:Spin, B<:Basis}


"""
A linear operator acting on a field with particular pixelization scheme and spin, and
which can only be applied in a given basis (i.e. the one in which its diagonal)
"""
abstract LinearFieldOp{P<:Pix, S<:Spin, B<:Basis}

"""
Covariances are linear operators on the fields, and some algebra (like adding or subtracting
two of them in the same basis) can be done expclitly.
"""
abstract FieldCov{P<:Pix, S<:Spin, B<:Basis} <: LinearFieldOp{P,S,B}



"""
ℱ * f and ℱ \ f converts fields between map and fourier space
(mostly this is done automatically so you don't ever use this explicitly)
"""
abstract ℱ


# by default, Field objects have no metadata and all of their fields are "data"
# which is operated on by various operators, +,-,*,...  
# this can, of course, be overriden for any particular field
meta(::Field) = tuple()
data{T<:Field}(f::T) = fieldvalues(f)


# Use generated functions to get planned FFT's only once for any given (Ωpix,
# Nside) combination
@generated function FFTgrid{T<:Flat}(::Type{T})
    Ωpix, Nside = T.parameters
    FFTgrid(2, Ωpix*Nside*pi/(180*60), Nside)
end
@generated ℱ{T<:Flat}(::Type{T}) = FFTgrid(T).FFT


# ---------------
# Flat sky spin-0
# ---------------

""" A flat sky spin-0 map (like T or ϕ) """
immutable FlatS0Map{T<:Real,P<:Flat} <: Field{P,S0,Map}
    Tx::Matrix{T}
end

""" The fourier transform of a flat sky spin-0 map """
immutable FlatS0Fourier{T<:Real,P<:Flat} <: Field{P,S0,Fourier}
    Tl::Matrix{Complex{T}}
end
*{T,P}(::Type{ℱ}, f::FlatS0Map{T,P}) = FlatS0Fourier{T,P}(ℱ(P) * f.Tx)
\{T,P}(::Type{ℱ}, f::FlatS0Fourier{T,P}) = FlatS0Map{T,P}(real(ℱ(P) \ f.Tl))


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
simulate{T,P}(Σ::FlatS0FourierDiagCov{T,P}) = FlatS0Fourier{T,P}(ℱ(P) * randn(Nside(P),Nside(P)) .* √Σ.Cl / FFTgrid(P).deltx)


""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov{P<:Flat}(::Type{FlatS0FourierDiagCov}, ℓ, CℓTT, ::Type{P})
    FlatS0FourierDiagCov{Float64,P}(complex(BayesLensSPTpol.cls_to_cXXk(ℓ, CℓTT, FFTgrid(P).r)))
end

# Can raise these guys to powers explicitly since they're diagonal
^{T<:Union{FlatS0MapDiagCov,FlatS0FourierDiagCov}}(f::T, n::Number) = T(map(.^,data(f),repeated(n))..., meta(f)...)

# how to convert to and from vectors (when needing to feed into other algorithms)
tovec(f::FlatS0Map) = f.Tx[:]
tovec(f::FlatS0Fourier) = f.Tl[:]
fromvec{T<:Union{FlatS0Map,FlatS0Fourier}}(::Type{T}, vec::AbstractVector, g) = T(reshape(vec,(g.Nside,g.Nside)),g)
eltype{T}(::Type{FlatS0Map{T}}) = T
eltype{T}(::Type{FlatS0Fourier{T}}) = Complex{T}
size(f::Union{FlatS0MapDiagCov,FlatS0FourierDiagCov}) = (f.g.Nside^2, f.g.Nside^2)


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

function FlatS0LensingOp{B}(ϕ::Field{Flat,S0,B}; order=4, taylens=true)
    g = ϕ.g
    Nside = g.Nside

    # total displacement
    dx, dy = LenseDecomp_helper1(ϕ[:Tl], zeros(ϕ[:Tl]), g);

    # nearest pixel displacement
    if taylens
        di, dj = (round(Int,d/g.deltx) for d=(dx,dy))
        i = indexwrap.(di .+ (1:Nside)', Nside)
        j = indexwrap.(dj .+ (1:Nside) , Nside)
    else
        di = dj = i = j = zeros(Int,Nside,Nside)
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



# ======================================
# algebra with Fields and LinearFieldOps
# ======================================


# addition and subtraction of two Fields or FieldCovs
for op in (:+, :-)
    @eval @swappable ($op){P,S}(a::Field{P,S,Map}, b::Field{P,S,Fourier}) = ($op)(a, Map(b))
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

Map{P,S}(f::Field{P,S,Map}) = f
Map{P,S}(f::Field{P,S,Fourier}) = ℱ\f
Fourier{P,S}(f::Field{P,S,Map}) = ℱ*f
Fourier{P,S}(f::Field{P,S,Fourier}) = f


# convert Fields to right basis before feeding into a LinearFieldOp
*{P,S}(op::LinearFieldOp{P,S,Map}, f::Field{P,S,Fourier}) = op * Map(f)
*{P,S}(op::LinearFieldOp{P,S,Fourier}, f::Field{P,S,Map}) = op * Fourier(f)



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
