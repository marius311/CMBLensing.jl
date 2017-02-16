module CMBFields

using Util
using DataArrays: @swappable
import Base: +, -, .+, .-, *, \, /, ^, ~, .*, ./, getindex, size, eltype
import Base: promote_type, convert


export 
    Field, LinearFieldOp, simulate, Cℓ_to_cov,
    Map, Fourier,
    ∂x, ∂y, ∇


# a _type of (Pix,Spin,Basis) defines the generic behavior of our fields
abstract Pix
abstract Spin
abstract Basis


# spin types (pix/spin types are defined in corresponding files included below)
abstract S0 <: Spin
abstract S2 <: Spin
abstract S02 <: Spin


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
Diagonal operators are special because some math can be done on them expclitly. 
"""
abstract LinearFieldDiagOp{P,S,B} <: LinearFieldOp{P,S,B}


# by default, Field objects have no metadata and all of their fields are "data"
# which is operated on by various operators, +,-,*,...  
# this can, of course, be overriden _for any particular Field
meta(::Union{Field,LinearFieldOp}) = tuple()
data{T<:Union{Field,LinearFieldOp}}(f::T) = fieldvalues(f)


"""
Operator used to take derivatives
"""
immutable ∂Op{s} <: LinearFieldOp end
∂x, ∂y = ∂Op{:x}(),∂Op{:y}()
∇ = [∂x, ∂y]


include("flat.jl")
include("flat_s0.jl")
include("flat_s2.jl")
include("algebra.jl")
include("vec_conv.jl")
include("healpix.jl")
include("lenseflow.jl")
include("taylens.jl")


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
