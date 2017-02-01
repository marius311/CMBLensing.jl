module CMBFields

using Util
using DataArrays: @swappable
import Base: +, -, *, \, /, ^, ~, .*, ./, getindex, size, eltype
import Base: promote_type, convert


export 
    Field, simulate, Cℓ_to_cov,
    Flat, Map, Fourier, FlatS0Fourier, FlatS0FourierDiagCov, FlatS0Map, FlatS0MapDiagCov    


abstract Pix

# Healpix pixelization with particular `Nside` value
abstract Healpix{Nside<:Int} <: Pix

abstract Spin
abstract S0 <: Spin
abstract S2 <: Spin
abstract S02 <: Spin


abstract Basis


"""
A field with a particular pixelization scheme, spin, and basis.
"""
abstract Field{P<:Pix, S<:Spin, B<:Basis}

"""
A linear operator acting on a field with particular pixelization scheme and spin, and
which is by default applied to a field in a given basis (i.e. the one in which its diagonal)
"""
abstract LinearFieldOp{P<:Pix, S<:Spin, B<:Basis}

"""
Covariances are linear operators on the fields, and are special because some
algebra (like adding or subtracting two of them in the same basis) can be done
explicitly.
"""
abstract FieldCov{P<:Pix, S<:Spin, B<:Basis} <: LinearFieldOp{P,S,B}


# by default, Field objects have no metadata and all of their fields are "data"
# which is operated on by various operators, +,-,*,...  
# this can, of course, be overriden _for any particular Field
meta(::Union{Field,LinearFieldOp}) = tuple()
data{T<:Union{Field,LinearFieldOp}}(f::T) = fieldvalues(f)


include("flat.jl")
include("flat_s0.jl")
include("flat_s2.jl")
include("algebra.jl")


# allow converting Fields and LinearFieldOp to/from vectors
getindex(f::Field, i::Colon) = tovec(f)
function getindex{P,S,B}(op::LinearFieldOp, f::Tuple{Field{P,S,B}})
    v = f[1][:]
    LazyVecApply{typeof(f[1]),B}(op,eltype(v),tuple(fill(length(v),2)...))
end
immutable LazyVecApply{F,B}
    op::LinearFieldOp
    eltype::Type
    size::Tuple
end
*{F,B}(lazy::LazyVecApply{F,B}, vec::AbstractVector) = tovec(B(lazy.op*fromvec(F,vec,meta(lazy.op)...)))
~(f::Field) = (f,)

eltype(lz::LazyVecApply) = lz.eltype
size(lz::LazyVecApply) = lz.size
size(lz::LazyVecApply, d) = d<=2 ? lz.size[d] : 1
getindex{P,S,B}(vec::AbstractVector, s::Tuple{Field{P,S,B}}) = fromvec(typeof(s[1]),vec,meta(s[1])...)


function getindex(f::Field,x::Symbol)
    T = supertype(typeof(f))
    parameters = T.parameters
    T.parameters = Core.svec(T.parameters[1:2]..., Field.parameters[3])
    l = filter(S->x in fieldnames(S), subtypes(T))
    T.parameters = parameters #todo: get rid this hack of saving T.parameters
    if (length(l)==1)
        getfield(supertype(l[1]).parameters[3](f),x)
    elseif (length(l)==0)
        throw("No subtype of $T has a field $x")
    else
        throw("Amiguous field. Multiple subtypes of $T have a field $x: $l")
    end
end



end
