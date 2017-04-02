module CMBFields

using IterativeSolvers
using Interpolations
using MacroTools
using NamedTuples
using ODE
using PyCall
using PyPlot
using StaticArrays
using StatsBase


using Base.Iterators: repeated

import PyPlot: plot
import Base: +, -, *, \, /, ^, ~, .*, ./, .^, sqrt, getindex, size, eltype, zero, length
import Base: convert, promote_rule
import Base.LinAlg: dot, norm, isnan
import Base: start, done, next



export 
    Field, LinOp, LinDiagOp, FullDiagOp, Ð, Ł, simulate, Cℓ_to_cov,
    S0, S2, S02, Map, Fourier,
    ∂x, ∂y, ∇,
    Cℓ_2D, class, ⨳, @⨳, shortname, Squash


# a type of (Pix,Spin,Basis) defines the generic behavior of our fields
abstract type Pix end
abstract type Spin end
abstract type Basis end


# spin types (pix/spin types are defined in corresponding files included below)
abstract type S0 <: Spin end
abstract type S2 <: Spin end
abstract type S02 <: Spin end

"""
A field with a particular pixelization scheme, spin, and described in a particular basis.
"""
abstract type Field{P<:Pix, S<:Spin, B<:Basis} end


"""
A linear operator acting on a field with a particular pixelization scheme and
spin. The meaning of the basis (B) is not necessarliy the basis the operator is
stored in, rather it specifies that fields should be converted to basis B before
being acted on by the operator. 

All LinOps receive the following functionality:

* Automatic conversion: A*f where A is a LinOp and f is a Field first converts f
  to A's basis, then calls the * function

* Lazy evaluation: C = A + B returns a LazyBinaryOp object which when
  applied to a field, C*f, computes A*f + B*f. 
  
* Vector conversion: Af = A[~f] returns an object which when acting on an
  AbstractVector, Af * v, converts v to a Field, then applies A. 

"""
abstract type LinOp{P<:Pix, S<:Spin, B<:Basis} end


"""
Operators which are diagonal in basis B. This is imporant because it means we
can do explicit broadcasting between these operators and other fields which are
also in basis B.

Each LinDiagOp needs to implement broadcast_data(::Type{F}, op::LinDiagOp) which
should return a tuple of data which can be broadcasted together with the data of a
field of type F.
"""
abstract type LinDiagOp{P,S,B} <: LinOp{P,S,B} end


"""
An LinDiagOp which is stored explicitly as all of its diagonal coefficients in
the basis in which it's diagonal. 
"""
struct FullDiagOp{F<:Field,P,S,B} <: LinDiagOp{P,S,B}
    f::F
    FullDiagOp(f::F) where {P,S,B,F<:Field{P,S,B}} = new{F,P,S,B}(f)
end
for op=(:*,:\)
    @eval ($op)(O::FullDiagOp{F}, f::F) where {F} = $(Symbol(:.,op))(O.f,f)
end
simulate(Σ::FullDiagOp{F}) where {F} = sqrt.(Σ) .* F(white_noise(F))
broadcast_data(::Type{F}, op::FullDiagOp{F}) where {F} = broadcast_data(F,op.f)
containertype(op::FullDiagOp) = containertype(op.f)
shortname(::Type{<:FullDiagOp{F}}) where {F} = "FullDiagOp{$(shortname(F))}"


""" 
A "basis-like" object, e.g. the lensing basis Ł or derivative basis Ð.
For any particular types of fields, these might be different actual bases, e.g.
the lensing basis is Map for S0 but QUMap for S2.
"""
abstract type Basislike <: Basis end
(::Type{B})(f::F) where {F<:Field,B<:Basislike} = B(F)(f)
(::Type{B})(a::AbstractArray{<:Field}) where {B<:Basislike} = B.(a)

# Operator used to take derivatives.
abstract type DerivBasis <: Basislike end
const Ð = DerivBasis
struct ∂{s} <: LinDiagOp{Pix,Spin,DerivBasis} end
const ∂x,∂y= ∂{:x}(),∂{:y}()
const ∇ = @SVector [∂x,∂y]
const ∇ᵀ = RowVector(∇)
*(∂::∂, f::Field) = ∂ .* Ð(f)
shortname(::Type{∂{s}}) where {s} = "∂$s"

""" An Op which applies some arbitrary function to its argument """
struct FuncOp{F<:Function} <: LinOp{Pix,Spin,Basis} 
    func::F
end
*(op::FuncOp, f::Field) = op.func.(f)



shortname(::Type{F}) where {F<:Field} = F.name.name




include("util.jl")
include("algebra.jl")
include("lensing.jl")
include("field_tuples.jl")
include("flat.jl")
include("vec_conv.jl")
include("healpix.jl")
include("plotting.jl")
include("cls.jl")
include("likelihood.jl")
include("wiener.jl")


""" An Op which turns all NaN's to zero """
const Squash = FuncOp(nan2zero)


getbasis(::Type{F}) where {P,S,B,F<:Field{P,S,B}} = B
function getindex(f::F,x::Symbol) where {P,S,B,F<:Field{P,S,B}}
    l = filter(S->x in fieldnames(S), subtypes(Field{P,S}))
    if (length(l)==1)
        getfield(getbasis(l[1])(f),x)
    elseif (length(l)==0)
        error("No subtype of $F has a field $x")
    else
        error("Ambiguous field. Multiple subtypes of $F have a field $x: $l")
    end
end



end
