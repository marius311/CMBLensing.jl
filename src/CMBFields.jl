module CMBFields

using Base.Iterators: repeated
using PyCall
using PyPlot
using DataArrays: @swappable
using IterativeSolvers
using StaticArrays
using BayesLensSPTpol: cls_to_cXXk, class
import PyPlot: plot
import Base: +, -, *, \, /, ^, ~, .*, ./, .^, sqrt, getindex, size, eltype, zero, length
import Base: convert, promote_rule
import Base.LinAlg: dot


export 
    Field, LinOp, LinDiagOp, FullDiagOp, Ð, Ł, simulate, Cℓ_to_cov,
    S0, S2, S02, Map, Fourier,
    ∂x, ∂y, ∇,
    cls_to_cXXk, class, ⨳


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
Diagonal operators are operators over which we can do explicit broadcasting.
Each LinDiagOp needs to implement broadcast_data(::Type{F}, op::LinDiagOp) which
should return a tuple of data which can be broadcasted togther with the data of a
field of type F.
"""
abstract type LinDiagOp{P,S,B} <: LinOp{P,S,B} end


"""
An LinDiagOp which is stored explicitly as all of its diagonal coefficients in
the basis in which it's diagonal. 
"""
struct FullDiagOp{P,S,B,F<:Field{P,S,B}} <: LinDiagOp{P,S,B}
    f::F
    FullDiagOp{P,S,B,F}(f::F) where {P,S,B,F<:Field{P,S,B}} = new{P,S,B,F}(f)
    FullDiagOp{P,S,B,F}(args...) where {P,S,B,F<:Field{P,S,B}} = new{P,S,B,F}(F(args...))
    FullDiagOp(f::F) where {P,S,B,F<:Field{P,S,B}} = new{P,S,B,F}(f)
end
for op=(:*,:\)
    @eval ($op){P,S,B,F}(O::FullDiagOp{P,S,B,F}, f::Field{P,S,B}) = $(Symbol(:.,op))(O.f,f)
end
simulate(Σ::FullDiagOp{P,S,B,F}) where {P,S,B,F} = sqrt.(Σ) .* B(white_noise(F))
broadcast_promote_type{D<:FullDiagOp}(::Type{D},::Type{D}) = D
broadcast_data(::Type{F}, op::FullDiagOp{P,S,B,F}) where {P,S,B,F<:Field} = broadcast_data(F,op.f)
broadcast_data(::Type{D}, op::D) where {P,S,B,F,D<:FullDiagOp{P,S,B,F}} = broadcast_data(F,op.f)


# Operator used to take derivatives.
struct ∂{s} <: LinDiagOp{Pix,Spin,Basis} end
const ∂x = ∂{:x}()
const ∂y = ∂{:y}()
const ∇ = @SVector [∂x,∂y]
const ∇ᵀ = RowVector(∇)
∂Basis{F<:Field}(f::F) = ∂Basis(F)(f)
∂Basis(a::AbstractArray{<:Field}) = ∂Basis.(a)
∂Basis{F<:Field}(::Type{F}) = error("""To take a derivative a field of type $F, ∂Basis(f::$F) needs to be implemented.""")
const Ð = ∂Basis
*(∂::∂, f::Field) = ∂ .* Ð(f)
broadcast_promote_type(::Type{<:∂},::Type{<:∂}) = LinDiagOp{Pix,Spin,Basis}
# todo: maybe define broadcast_data(::Type{FullDiagOp}) to allow things like B =
# ∂x * A (with A,B as operators ?)





# Generic Wiener filter
struct WienerFilter{tol,TS<:LinOp,TN<:LinOp} <: LinOp{Pix,Spin,Basis}
    S::TS
    N::TN
end
const 𝕎 = WienerFilter
𝕎{TS,TN}(S::TS,N::TN,tol=1e-3) = 𝕎{tol,TS,TN}(S,N)
function *{tol}(w::𝕎{tol}, d::Field)
    A = w.S^-1+w.N^-1
    if isa(A,LinDiagOp)  
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
include("algebra.jl")
include("vec_conv.jl")
include("healpix.jl")
include("lensing.jl")
include("plotting.jl")

getbasis(::Type{F}) where {P,S,B,F<:Field{P,S,B}} = B
function getindex(f::F,x::Symbol) where {P,S,B,F<:Field{P,S,B}}
    l = filter(S->x in fieldnames(S), subtypes(Field{P,S}))
    if (length(l)==1)
        getfield(getbasis(l[1])(f),x)
    elseif (length(l)==0)
        throw("No subtype of $F has a field $x")
    else
        throw("Ambiguous field. Multiple subtypes of $F have a field $x: $l")
    end
end



end
