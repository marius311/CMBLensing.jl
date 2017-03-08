module CMBFields

using PyCall
using PyPlot
using DataArrays: @swappable
using IterativeSolvers
import PyPlot: plot
import Base: +, -, .+, .-, *, \, /, ^, ~, .*, ./, .^, sqrt, getindex, size, eltype, zero, length
import Base: promote_type, convert
import Base.LinAlg: dot


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


# Operator used to take derivatives.
# Fields should implement *(op::∂Op, f::Field) to take derivatives, and
# ∂Basis to specify a basis into which to automatically convert fields before
# they are fed into this "*" method.
# Note: defining ∂Op in this way allows it be a bonafide LinearOp which can be
# both lazily evaluated and applied to all field types. 
immutable ∂Op{s,n} <: LinearOp end
^{s,n}(::∂Op{s,n}, m::Integer) = ∂Op{s,n*m}()
∂x, ∂y = ∂Op{:x,1}(), ∂Op{:y,1}()
∇ = [∂x, ∂y]; ∇ᵀ = [∂x ∂y]
*(op::∂Op,f::Field) = op * ∂Basis(typeof(f))(f)
∂Basis{F<:Field}(::Type{F}) = error("""To take a derivative a field of type $F, ∂Basis(f::$F) needs to be implemented.""")


# For each Field type, lensing algorithms needs to know the basis in which lensing is a
# remapping. E.g. for FlatS0 and FlatS2 this is Map and QUMap, respectively.
# Fields should implement their own LenseBasis(::Type{F}) to specify. 
LenseBasis{F<:Field}(f::F) = LenseBasis(F)(f)
LenseBasis{F<:Field}(::Type{F}) = error("""To lense a field of type $(typeof(f)), LenseBasis(f::$(typeof(f))) needs to be implemented.""")
LenseBasis{F<:Field}(x::AbstractArray{F}) = map(LenseBasis,x)
Ł = LenseBasis


# Generic Wiener filter
immutable WienerFilter{tol,TS<:LinearOp,TN<:LinearOp} <: LinearOp
    S::TS
    N::TN
end
typealias 𝕎 WienerFilter
𝕎{TS,TN}(S::TS,N::TN,tol=1e-3) = 𝕎{tol,TS,TN}(S,N)
function *{tol}(w::𝕎{tol}, d::Field)
    A = w.S^-1+w.N^-1
    if isa(A,LinearDiagOp)  
        # if S & N are diagonal in the same basis they can be added/inverted directly
        A^-1 * w.N^-1 * d
    else
        # otherwise solve using conjugate gradient
        swf, hist = cg(A[~d], (w.N^-1*d)[:], tol=tol, log=true)
        hist.isconverged ? swf[~d] : error("Conjugate gradient solution of Wiener filter did not converge.")
    end
end

include("util.jl")
include("flat.jl")
include("flat_s0.jl")
include("flat_s2.jl")
include("algebra.jl")
include("vec_conv.jl")
include("healpix.jl")
include("lenseflow.jl")
include("taylens.jl")
include("powerlens.jl")


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
