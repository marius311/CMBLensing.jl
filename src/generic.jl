
# A tuple of (Pix,Spin,Basis) defines the generic behavior of our fields
abstract type Pix end
abstract type Spin end
abstract type Basis end

# All fields are a subtype of this. 
abstract type Field{B<:Basis, S<:Spin, P<:Pix} end

# Spin types, "S0" is spin-0, i.e. a scalar map. "S2" is spin-2 like QU, and S02
# is a tuple of S0 and S2 like TQU. 
abstract type S0 <: Spin end
abstract type S2 <: Spin end
abstract type S02 <: Spin end

# Basis types
abstract type Map <: Basis end
abstract type Fourier <: Basis end
abstract type QUMap <: Basis end
abstract type EBMap <: Basis end
abstract type QUFourier <: Basis end
abstract type EBFourier <: Basis end

# A "basis-like" object, e.g. the lensing basis Ł or derivative basis Ð. For any
# particular types of fields, these might be different actual bases, e.g. the
# lensing basis is Map for S0 but QUMap for S2.
abstract type Basislike <: Basis end
(::Type{B})(f::F) where {F<:Field,B<:Basislike} = B(F)(f)
(::Type{B})(a::AbstractArray{<:Field}) where {B<:Basis} = B.(a)



### LinOp

#
# A LinOp{B,S,P} represents a linear operator which acts on a field with a
# particular pixelization scheme P and spin S. The meaning of basis B is not
# that the operator is stored in this basis, but rather that fields should be
# converted to this basis before the operator is applied.
# 
# In the simplest case, LinOps should implement *, inv, and adjoint. 
# 
#     * *(::LinOp, ::Field) - apply the operator
#     * inv(::LinOp) - return the inverse operator (called by L^-1 and L\f)
#     * adjoint(::LinOp) - return the conjugate transpose operator (called by L'*f and L'\f)
# 
# These three functions are used by default in the following fallbacks. These
# fallbacks can be overriden with more efficient implementations of the
# following functions if desired.
#
#     * *(::Field, ::LinOp) - apply the transpose operator
#     * \(::LinOp, ::Field) - apply the inverse operator
#     * Ac_ldiv_B(::LinOp, ::Field) - apply the inverse transpose operator
#
# Note that *(::Field, ::LinOp) implies the transpose operation and it is this
# function rather than Ac_mul_B(::LinOp, ::Field) which we choose to have LinOps
# optionally override. 
# 
# Other functions which can be implemented:
#
#     * sqrtm(L::LinOp) - the sqrt of the operator s.t. sqrtm(L)*sqrtm(L) = L
# 
#
# By default, LinOps receive the following functionality:
# 
#     * Automatic basis conversion: L * f first converts f to L's basis, then
#     applies *, so that LinOps only need to implement *(::LinOp, ::F) where F
#     is a type already in the correct basis. 
# 
#     * Lazy evaluation: C = A + B returns a LazyBinaryOp object which when
#     applied to a field, C*f, computes A*f + B*f.
# 
#     * Vector conversion: Af = A[~f] returns an object which when acting on an
#     AbstractVector, Af * v, converts v to a Field, then applies A.
# 
abstract type LinOp{B<:Basis, S<:Spin, P<:Pix} end

# Assuming *, inv, and adjoint are implemented, the following fallbacks make
# everything work, or can be overriden by individual LinOps with more efficient
# versions.
literal_pow(::typeof(^), L::LinOp, ::Type{Val{-1}}) = inv(L)
Ac_mul_B(L::LinOp, f::Field) = f*L
*(f::Field, L::LinOp) = adjoint(L)*f
\(L::LinOp, f::Field) = inv(L)*f
Ac_ldiv_B(L::LinOp, f::Field) = f*inv(L)

# automatic basis conversion
for op=(:*,:\)
    @eval ($op)(L::LinOp{B}, f::Field) where {B} = $op(L,B(f))
end


### LinDiagOp

#
# LinDiagOp{B,S,P} is an operator which is diagonal in basis B. This is
# important because it means we can do fast broadcasting between these
# operators and other fields which are also in basis B.
# 
# Each LinDiagOp needs to implement broadcast_data(::Type{F}, L::LinDiagOp) which
# should return a tuple of data which can be broadcast together with the data of a
# field of type F.
#
abstract type LinDiagOp{B,S,P} <: LinOp{B,S,P} end
adjoint(L::LinDiagOp) = L #TODO: actually this isnt true unless is also hermitian

# automatic basis conversion & broadcasting
for op=(:*,:\)
    @eval ($op)(L::LinDiagOp{B}, f::Field) where {B} = broadcast($op,L,B(f))
end


### Scalars

# scalars which are allowed in our expressions must be real because we
# implicitly assume our maps are real, and addition/multiplication by a complex
# number, even of the fourier transform, would break this.
const Scalar = Real
const FieldOpScal = Union{Field,LinOp,Scalar}


### Other generic stuff


# convenience "getter" functions for the Basis/Spin/Pix
basis(::Type{<:Field{B,S,P}}) where {B,S,P} = B
basis(::F) where {F<:Field} = basis(F)
spin(::Type{<:Field{B,S,P}}) where {B,S,P} = S
spin(::F) where {F<:Field} = spin(F)
pix(::Type{<:Field{B,S,P}}) where {B,S,P} = P
pix(::F) where {F<:Field} = pix(F)


shortname(::Type{T}) where {T<:Union{Field,LinOp,Basis}} = replace(replace(string(T),"CMBLensing."=>""), "Main."=>"")

zero(::F) where {F<:Field} = zero(F)
similar(f::F) where {F<:Field} = F(map(similar,broadcast_data(F,f))...)
copy(f::Field) = deepcopy(f)


## properties

# this allows you write eg f.Tl even when f::FlatS0Map, and it automatically
# first converts f to FlatS0Fourier *then* takes the Tl field (this is
# type-stable and has no overhead otherwise)

# under the hood, the code guesses what the given field could be converted to by
# finding other Field types that differ only in basis, scans through the
# fieldnames of those types, then converts to the appropriate one


# gets other fields types which share the same spin and pixelzation, differing
# only in basis, meaning you should be able to convert F to these types
convertable_fields(::Type{F}) where {B,S,P,F<:Field{B,S,P}} = subtypes(Field{B′,S,P} where B′)

# use convertable_fields to get possible properties
@generated propertynames(::Type{F}) where {F<:Field} = tuple(chain(fieldnames.(convertable_fields(F))...)...)
propertynames(::F) where {F<:Field} = propertynames(F)

# implement getproperty using possible conversions
getproperty(f::Field, s::Symbol) = getproperty(f,Val(s))
@generated function getproperty(f::F,::Val{s}) where {F<:Field, s}
    l = filter(F′->(s in fieldnames(F′)), convertable_fields(F))
    if (length(l)==1)
        :(getfield($(l[1])(f),s))
    elseif (length(l)==0)
        error("type $F has no property $s")
    else
        error("Ambiguous property. Multiple types that $F could be converted to have a field $s: $l")
    end
end


function get_Cℓ(args...; kwargs...) end
get_αℓⁿCℓ(α=1,n=0,args...; kwargs...) = ((ℓ,Cℓ)=get_Cℓ(args...; kwargs...); (ℓ, @. (α*ℓ^n*Cℓ)))
get_Dℓ(args...; kwargs...)            = ((ℓ,Cℓ)=get_Cℓ(args...; kwargs...); (ℓ, @. ℓ*(ℓ+1)*Cℓ/(2π)))
get_ℓ⁴Cℓ(args...; kwargs...)          = get_αℓⁿCℓ(1,4,args...; kwargs...)
