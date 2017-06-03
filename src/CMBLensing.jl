module CMBLensing

using Base.Iterators: repeated
using Base.Threads
using Interpolations
using IterativeSolvers
using MacroTools
using NamedTuples
using ODE
using Parameters
using PyCall
using StaticArrays
using StatsBase

include("RFFTVectors.jl"); using .RFFTVectors


import Base: +, -, *, \, /, ^, ~, .*, ./, .^,
    Ac_mul_B, Ac_ldiv_B, broadcast, convert, done, eltype, getindex,
    inv, length, literal_pow, next, promote_rule, size, sqrt, start, transpose, ctranspose, one, zero, sqrtm
import Base.LinAlg: dot, norm, isnan

include("util.jl")

function __init__()
    global classy = pyimport("classy")
    # global hp = pyimport("healpy")
end


export
    Field, LinOp, LinDiagOp, FullDiagOp, Ð, Ł, simulate, Cℓ_to_cov,
    S0, S2, S02, Map, Fourier,
    ∂x, ∂y, ∇,
    Cℓ_2D, class, ⨳, @⨳, shortname, Squash, pixstd

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
sqrtm(f::FullDiagOp) = sqrt.(f)
simulate(Σ::FullDiagOp{F}) where {F} = sqrtm(Σ) .* F(white_noise(F))
broadcast_data(::Type{F}, op::FullDiagOp{F}) where {F} = broadcast_data(F,op.f)
containertype(op::FullDiagOp) = containertype(op.f)
literal_pow(^,op::FullDiagOp,::Type{Val{-1}}) = inv(op)
inv(op::FullDiagOp) = FullDiagOp(1./op.f)


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
function gradhess(f)
    (∂xf,∂yf)=∇*Ð(f)
    ∂xyf = ∂x*∂yf
    @SVector([∂xf,∂yf]), @SMatrix([∂x*∂xf ∂xyf; ∂xyf ∂y*∂yf])
end
shortname(::Type{∂{s}}) where {s} = "∂$s"

"""
An Op which applies some arbitrary function to its argument.
Transpose and/or inverse operations which are not specified will return an error.
"""
@with_kw struct FuncOp <: LinOp{Pix,Spin,Basis}
    op   = nothing
    opᴴ  = nothing
    op⁻¹ = nothing
    op⁻ᴴ = nothing
end
SymmetricFuncOp(;op=nothing, op⁻¹=nothing) = FuncOp(op,op,op⁻¹,op⁻¹)
@∷ *(op::FuncOp, f::Field) = op.op   != nothing ? op.op(f)   : error("op*f not implemented")
@∷ *(f::Field, op::FuncOp) = op.opᴴ  != nothing ? op.opᴴ(f)  : error("f*op not implemented")
@∷ \(op::FuncOp, f::Field) = op.op⁻¹ != nothing ? op.op⁻¹(f) : error("op\\f not implemented")
ctranspose(op::FuncOp) = FuncOp(op.opᴴ,op.op,op.op⁻ᴴ,op.op⁻¹)
const IdentityOp = FuncOp(repeated(identity,4)...)
literal_pow(^,op::FuncOp,::Type{Val{-1}}) = FuncOp(op.op⁻¹,op.op⁻ᴴ,op.op,op.opᴴ)
shortname(::Type{F}) where {F<:Field} = replace(string(F),"CMBLensing.","")




include("algebra.jl")
include("field_tuples.jl")
include("lensing.jl")
include("flat.jl")
include("taylens.jl")
include("vec_conv.jl")
include("healpix.jl")
displayable(MIME("image/png")) && include("plotting.jl")
include("cls.jl")
include("likelihood.jl")
include("wiener.jl")


""" An Op which turns all NaN's to zero """
const Squash = FuncOp(op=x->broadcast(nan2zero,x))


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
getindex(f::Field2Tuple{<:Field{<:Any,<:S0},<:Field{<:Any,<:S2}},s::Symbol) = startswith(string(s),"T") ? f.f1[s] : f.f2[s]

# submodules
include("minimize.jl")
include("masking.jl")

end
