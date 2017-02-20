module CMBFields

using Util
using DataArrays: @swappable
import Base: +, -, .+, .-, *, \, /, ^, ~, .*, ./, sqrt, getindex, size, eltype
import Base: promote_type, convert


export 
    Field, LinearOp, LinearDiagOp, simulate, Cℓ_to_cov,
    S0, S2, S02, Map, Fourier,
    ∂x, ∂y, ∇


# a type of (Pix,Spin,Basis) defines the generic behavior of our fields
abstract Pix
abstract Spin
abstract Basis


# spin types (pix/spin types are defined in corresponding files included below)
abstract S0 <: Spin
abstract S2 <: Spin
abstract S02 <: Spin


"""
A field with a particular pixelization scheme, spin, and described in a particular basis.
"""
abstract Field{P<:Pix, S<:Spin, B<:Basis}

"""
A linear operator acting on a field with a particular pixelization scheme and
spin. The meaning of the basis (B) is not necessarliy the basis the operator is
stored in, rather it specifies that fields should be converted to basis B before
being acted on by the operator. 
"""
abstract LinearOp{P<:Pix, S<:Spin, B<:Basis}

"""
Operators which are stored explicitly as their non-zero coefficients in the basis
in which they are diagonal. 
"""
immutable LinearDiagOp{P<:Pix, S<:Spin, B<:Basis} <: LinearOp{P,S,B}
    f::Field{P,S,B} #todo: this can be made type stable in 0.6
end
*{P,S,B}(op::LinearDiagOp{P,S,B}, f::Field{P,S,B}) = op.f * f
simulate(Σ::LinearDiagOp) = √Σ * white_noise(typeof(Σ.f))



# by default, Field objects have no metadata and all of their fields are "data"
# which is operated on by various operators, +,-,*,...  
# this can, of course, be overriden for any particular Field
meta(::Union{Field,LinearOp}) = tuple()
data{T<:Union{Field,LinearOp}}(f::T) = fieldvalues(f)


"""
Operator used to take derivatives
"""
immutable ∂Op{s} <: LinearOp end
∂x, ∂y = ∂Op{:x}(), ∂Op{:y}()
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
