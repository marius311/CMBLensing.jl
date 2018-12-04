
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

basis_promotion_rules = Dict(
    # S0
    (Map,       Fourier)    => Map,
    # S2
    (QUMap,     QUFourier)  => QUMap,
    (EBMap,     EBFourier)  => EBFourier,
    (QUMap,     EBMap)      => QUMap,
    (QUFourier, EBFourier)  => QUFourier,
    (QUMap,     EBFourier)  => QUMap,
    (QUFourier, EBMap)      => QUFourier
)
for ((B1,B2),B′) in basis_promotion_rules
    @eval promote_rule(::Type{$B1}, ::Type{$B2}) = $B′
end


# A "basis-like" object, e.g. the lensing basis Ł or derivative basis Ð. For any
# particular types of fields, these might be different actual bases, e.g. the
# lensing basis is Map for S0 but QUMap for S2.
abstract type Basislike <: Basis end
(::Type{B})(f::F) where {F<:Field,B<:Basislike} = B(F)(f)
(::Type{B})(f′::Field, f::F) where {F<:Field,B<:Basislike} = B(F)(f′,f)
(::Type{B})(a::AbstractArray{<:Field}...) where {B<:Basis} = B.(a...)



### Adjoint Fields

# Fields implicitly have pixel indices which run from 1:Npix and "spin" indices
# which run e.g. from 1:3 labelling (I,Q,U) in the case of an S02 Field.
# Additionally, we may have vectors of fields like ∇*f which have an additional
# index labeling the component. By choice, we chose to make the Julia `'` symbol
# only take the adjoint of the spin and vector components, but not of the pixel.
# This means that f' * f will return a spin-0 field. To fully transpose and
# contract all indices, one would use the `⋅` function, f⋅f, which yields a
# scalar, as expected.

# Much like the Julia Base library, doing an adjoint like `f'` just creates a
# wrapper around the field to keep track of things. Adjoint fields have very
# limited functionality implemented as compared to normal fields, so its best to
# work with normal fields and only take the adjoint at the end if you need it.
struct AdjField{B,S,P,F<:Field{B,S,P}} <: Field{B,S,P}
    f :: F
    # need this seeingly redundant constructor to avoid ambiguity with 
    # F(f) = convert(F,f) definition in algebra.jl:
    AdjField(f::F) where {B,S,P,F<:Field{B,S,P}} = new{B,S,P,F}(f) 
end
adjoint(f::Field) = AdjField(f)
adjoint(f::AdjField) = f.f
*(a::AdjField{<:Any,S0}, b::Field{<:Any,<:S0}) = a.f * b
*(a::Field{<:Any,S0}, b::AdjField{<:Any,<:S0}) = a * b.f



### LinOp

#
# A LinOp{B,S,P} represents a linear operator which acts on a field with a
# particular pixelization scheme P and spin S. The meaning of basis B is not
# that the operator is stored in this basis, but rather that fields should be
# converted to this basis before the operator is applied (this makes writing the
# implementing functions somewhat more convenient)
# 
# In the simplest case, LinOps should implement mul!, ldiv!, and adjoint. 
# 
#     * mul!(result, ::LinOp, ::Field) - apply the operator, storing the answer in `result`
#     * ldiv!(result, ::LinOp, ::Field) - apply the inverse operator, storing the answer in `result`
#     * adjoint(::LinOp) - return the adjoint operator
# 
# By default `*` and `\` use `mul!` and `ldiv!`, assuming the result will be
# `simlar` to the field being acted on. If the operator returns a different kind
# of field, this can be specified by overloading `allocate_result(::LinOp,
# ::Field)` (see below)
#
# Other functions which can be implemented:
#
#     * sqrt(L::LinOp) - the sqrt of the operator s.t. sqrt(L)*sqrt(L) = L
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

# allocate the result of applying a LinOp to a given field. 
# the default below assumes the result is the same type as the Field itself, but
# this can be specialized (e.g. ∇*f returns instead a vector of fields)
allocate_result(::LinOp, f::Field) = similar(f)

# `*` and `\` use `mul!` and `ldiv!` which we require each LinOp implement
# here we also do the automatic basis conversion to the LinOps specified basis
*(L::LinOp{B}, f::Field) where {B} = (f′=B(f);  mul!(allocate_result(L,f′),L,f′))
\(L::LinOp{B}, f::Field) where {B} = (f′=B(f); ldiv!(allocate_result(L,f′),L,f′))

# Left multiplication uses `adjoint` which we require each LinOp implement
*(f::AdjField, L::LinOp) = (L'*f')'


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
transpose(L::LinDiagOp) = L
Diagonal(L::LinDiagOp) = L
for op=(:*,:\)
    @eval ($op)(L::LinDiagOp{B}, f::Field) where {B} = broadcast($op,L,B(f))
end


### Scalars

# scalars which are allowed in our expressions must be real because we
# implicitly assume our maps are real, and addition/multiplication by a complex
# number, even of the fourier transform, would break this.
const Scalar = Real
const FieldOpScal = Union{Field,LinOp,Scalar}


### Vectors of fields

const FieldVector{F<:Union{Field,LinOp}} = StaticVector{2,F}
const FieldRowVector{F<:Union{Field,LinOp}} = Adjoint{F,<:FieldVector}
const FieldMatrix{F<:Union{Field,LinOp}} = StaticMatrix{2,2,F}
const FieldArray{F<:Union{Field,LinOp}} = Union{FieldVector{F}, FieldRowVector{F}, FieldMatrix{F}}


### Matrix conversion

# We can build explicit matrix representations of linear operators by applying
# them to a set of basis vectors which form a complete basis (in practice this
# is prohibitive except for fairly small maps, e.g. 16x16 pixels, but is quite
# useful)

"""
    full(::Type{Fin}, ::Type{Fout}, L::LinOp; progress=true)
    full(::Type{F}, L::LinOp; progress=true)
    
Construct an explicit matrix representation of the linear operator `L` by
applying it to a set of vectors which form a complete basis. The `Fin` and
`Fout` types should be fields which specify the input and output bases for the
representation (or just `F` if `L` is square and we want the same input/output
bases)

The name `full` is to be consistent with Julia's SparseArrays where `full`
builds a full matrix from a sparse one.
"""
full(::Type{Fin}, ::Type{Fout}, L::LinOp; progress=true) where {Fin<:Field, Fout<:Field} =
    hcat(@showprogress(progress ? 1 : Inf, [(Fout(L*(x=zeros(length(Fin)); x[i]=1; x)[Tuple{Fin}]))[:] for i=1:length(Fin)])...)
full(::Type{F}, L::LinOp; kwargs...) where {F<:Field} = full(F,F,L; kwargs...)


### Other generic stuff

@doc doc"""
    norm²(f::Field, L::LinOp)
    
Shorthand for `f⋅(L\f)`, i.e. the squared-norm of `f` w.r.t. the operator `L`.
"""
norm²(f::Field, L::LinOp) = f⋅(L\f)

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


# gets other concrete fields types which share the same spin and pixelzation,
# differing only in basis, meaning you should be able to convert F to these
# types
concrete_subtypes(::Type{F}) where {F} = isabstracttype(F) ? vcat(map(concrete_subtypes,subtypes(F))...) : F
convertable_fields(::Type{F}) where {B,S,P,F<:Field{B,S,P}} = filter(x->!(x<:AdjField),concrete_subtypes(Field{B′,S,P} where B′))

# use convertable_fields to get possible properties
@generated propertynames(::Type{F}) where {F<:Field} = tuplejoin(fieldnames.(convertable_fields(F))...)
propertynames(::F) where {F<:Field} = propertynames(F)

# implement getproperty using possible conversions
getproperty(f::Field, s::Symbol) = getproperty(f,Val(s))
@generated function getproperty(f::F,::Val{s}) where {F<:Field, s}
    l = filter(F′->(s in fieldnames(F′)), convertable_fields(F))
    if s in fieldnames(F)
        :(getfield(f,s))
    elseif (length(l)==1)
        :(getfield($(l[1])(f),s))
    elseif (length(l)==0)
        error("type $F has no property $s")
    else
        error("Ambiguous property. Multiple types that $F could be converted to have a field $s: $l")
    end
end
function getindex(f::Field, s::Symbol)
    Base.depwarn("Syntax: f[:$s] is deprecated. Use f.$s or getproperty(f,:$s) instead.", "getindex")
    getproperty(f,Val(s))
end


function get_Cℓ(args...; kwargs...) end
get_αℓⁿCℓ(α=1,n=0,args...; kwargs...) = ((ℓ,Cℓ)=get_Cℓ(args...; kwargs...); (ℓ, @. (α*ℓ^n*Cℓ)))
get_Dℓ(args...; kwargs...)            = ((ℓ,Cℓ)=get_Cℓ(args...; kwargs...); (ℓ, @. ℓ*(ℓ+1)*Cℓ/(2π)))
get_ℓ⁴Cℓ(args...; kwargs...)          = get_αℓⁿCℓ(1,4,args...; kwargs...)
function get_ρℓ(f1,f2; kwargs...)
    ℓ,Cℓ1 = get_Cℓ(f1; kwargs...)
    ℓ,Cℓ2 = get_Cℓ(f2; kwargs...)
    ℓ,Cℓx = get_Cℓ(f1,f2; kwargs...)
    ℓ, @. Cℓx/sqrt(Cℓ1*Cℓ2)
end


module PolType
    export _PolType
    @enum _PolType T QU TQU
end
using .PolType
