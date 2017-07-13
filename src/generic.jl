# a type of (Pix,Spin,Basis) defines the generic behavior of our fields
abstract type Pix end
abstract type Spin end
abstract type Basis end

"""
A field with a particular pixelization scheme, spin, and described in a particular basis.
"""
abstract type Field{P<:Pix, S<:Spin, B<:Basis} end


"""
A "basis-like" object, e.g. the lensing basis Ł or derivative basis Ð.
For any particular types of fields, these might be different actual bases, e.g.
the lensing basis is Map for S0 but QUMap for S2.
"""
abstract type Basislike <: Basis end
(::Type{B})(f::F) where {F<:Field,B<:Basislike} = B(F)(f)
(::Type{B})(a::AbstractArray{<:Field}) where {B<:Basislike} = B.(a)


# spin types
abstract type S0 <: Spin end
abstract type S2 <: Spin end
abstract type S02 <: Spin end


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



shortname(::Type{T}) where {T<:Union{Field,LinOp,Basis}} = replace(string(T),"CMBLensing.","")

zero(::F) where {F<:Field} = zero(F)
similar(f::F) where {F<:Field} = F(map(similar,broadcast_data(F,f))...)
copy(f::Field) = deepcopy(f)

getbasis(::Type{F}) where {P,S,B,F<:Field{P,S,B}} = B
getindex(f::Field,x::Symbol) = getindex(f,Val{x})
function getindex(f::F,::Type{Val{x}}) where {x,P,S,B,F<:Field{P,S,B}}
    l = filter(S->x in fieldnames(S), subtypes(Field{P,S}))
    if (length(l)==1)
        getfield(getbasis(l[1])(f),x)
    elseif (length(l)==0)
        error("No subtype of $F has a field $x")
    else
        error("Ambiguous field. Multiple subtypes of $F have a field $x: $l")
    end
end 


function get_Cℓ(args...; kwargs...) end
get_αℓⁿCℓ(α=1,n=0,args...; kwargs...) = ((ℓ,Cℓ)=get_Cℓ(args...; kwargs...); (ℓ, @. (α*ℓ^n*Cℓ)))
get_Dℓ(args...; kwargs...)            = ((ℓ,Cℓ)=get_Cℓ(args...; kwargs...); (ℓ, @. ℓ*(ℓ+1)*Cℓ/(2π)))
get_ℓ⁴Cℓ(args...; kwargs...)          = get_αℓⁿCℓ(1,4,args...; kwargs...)
