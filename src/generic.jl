
# A tuple of (Pix,Spin,Basis) defines the generic behavior of our fields
abstract type Pix end
abstract type Spin end
abstract type Basis end

# All fields are a subtype of this. 
abstract type Field{B<:Basis, S<:Spin, P<:Pix, T} <: AbstractVector{T} end

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

# Basis in which derivatives are sparse for a given field
abstract type DerivBasis <: Basislike end
const Ð = DerivBasis
Ð!(args...) = Ð(args...)

# Basis in which lensing is a pixel remapping for a given field
abstract type LenseBasis <: Basislike end
const Ł = LenseBasis
Ł!(args...) = Ł(args...)


# B(f) where B is a basis converts f to that basis. This is the fallback if the
# field is already in the right basis.
(::Type{B})(f::Field{B}) where {B} = f

# The abstract `Basis` type means "any basis", hence this conversion rule:
Basis(f::Field) = f

# B(f′, f) converts f to basis B and stores the result inplace in f′. If f is
# already in basis B, we just return f (but note, we never actually set f′ in
# this case, which is more efficient, but necessitates some care when using this
# construct)
(::Type{B})(f′::Field{B}, f::Field{B}) where {B} = f



### ImplicitOp
# ImplicitOps and ImplicitFields are operators and fields which may not be
# explicitly stored, but can be multiplied, etc... and generally implicitly take
# on whatever size they need. The specification of Float32 as the the Array type
# is only for inference, such that its widened to whatever its needed to be the
# end. 
abstract type ImplicitOp{B<:Basis, S<:Spin, P<:Pix} <: AbstractMatrix{Float32} end
abstract type ImplicitField{B<:Basis, S<:Spin, P<:Pix} <: Field{B,S,P,Float32} end

# no size and 0-length represents the fact that ImplicitOps aquire the size of
# the fields they're applied to. it also helps makes the generic printing
# methods for AbstractMatrix work well.
size(::Union{ImplicitOp,ImplicitField}) = ()
length(::Union{ImplicitOp,ImplicitField}) = 0

# printing
show(io::IO, ::MIME"text/plain", L::ImplicitOp) = show(io,L)
show(io::IO, ::MIME"text/plain", L::Adjoint{<:Any,<:ImplicitOp}) = show(io,L)
show(io::IO, L::Adjoint{<:Any,<:ImplicitOp}) = (print(io,"Adjoint{"); show(io,parent(L)); print(io,"}"))
# this is the main function ImplicitOps should specialize if this default behavior isn't enough:
show(io::IO, L::ImplicitOp) = showarg(io, L, true)

# all CMBLensing operators are then either Diagonals or ImplicitOps
const DiagOp{F<:Field, T} = Diagonal{T,F} 
const LinOp{B,S,P} = Union{ImplicitOp{B,S,P},DiagOp{<:Field{B,S,P}}}
const ImplicitOrAdjOp{B,S,P} = Union{ImplicitOp{B,S,P}, Adjoint{<:Any,<:ImplicitOp{B,S,P}}}
const LinOrAdjOp{B,S,P} = Union{ImplicitOrAdjOp{B,S,P},DiagOp{<:Field{B,S,P}}}

### Scalars
# scalars which are allowed in our expressions must be real because we
# implicitly assume our maps are real, and addition/multiplication by a complex
# number, even of the fourier transform, would break this.
const Scalar = Real
const FieldOrOp = Union{Field,LinOp}
const FieldOpScal = Union{Field,LinOp,Scalar}


# 
# ### Matrix conversion
# 
# # We can build explicit matrix representations of linear operators by applying
# # them to a set of basis vectors which form a complete basis (in practice this
# # is prohibitive except for fairly small maps, e.g. 16x16 pixels, but is quite
# # useful)
# 
# """
#     full(::Type{Fin}, ::Type{Fout}, L::LinOp; progress=true)
#     full(::Type{F}, L::LinOp; progress=true)
# 
# Construct an explicit matrix representation of the linear operator `L` by
# applying it to a set of vectors which form a complete basis. The `Fin` and
# `Fout` types should be fields which specify the input and output bases for the
# representation (or just `F` if `L` is square and we want the same input/output
# bases)
# 
# The name `full` is to be consistent with Julia's SparseArrays where `full`
# builds a full matrix from a sparse one.
# """
# full(::Type{Fin}, ::Type{Fout}, L::LinOp; progress=true) where {Fin<:Field, Fout<:Field} =
#     hcat(@showprogress(progress ? 1 : Inf, [(Fout(L*(x=zeros(length(Fin)); x[i]=1; x)[Tuple{Fin}]))[:] for i=1:length(Fin)])...)
# full(::Type{F}, L::LinOp; kwargs...) where {F<:Field} = full(F,F,L; kwargs...)
# 
# 


### Other generic stuff


# convenience "getter" functions for the Basis/Spin/Pix
basis(::Type{<:Field{B,S,P}}) where {B,S,P} = B
basis(::F) where {F<:Field} = basis(F)
spin(::Type{<:Field{B,S,P}}) where {B,S,P} = S
spin(::F) where {F<:Field} = spin(F)
pix(::Type{<:Field{B,S,P}}) where {B,S,P} = P
pix(::F) where {F<:Field} = pix(F)


# we use Field cat'ing mainly for plotting, e.g. plot([f f; f f]) plots a 2×2
# matrix of maps. the following definitions make it so that Fields aren't
# splatted into a giant matrix when doing [f f; f f] (which they would othewise
# be since they're Arrays)
hvcat(rows::Tuple{Vararg{Int}}, values::Field...) = hvcat(rows, ([x] for x in values)...)
hcat(values::Field...) = hcat(([x] for x in values)...)

### printing
print_array(io::IO, f::Field) = print_array(io, f[:])
show_vector(io::IO, f::Field) = show_vector(io, f[:])


# addition/subtraction works between any fields and scalars, promotion is done
# automatically if fields are in different bases
for op in (:+,:-), (T1,T2) in ((:Field,:Scalar),(:Scalar,:Field),(:Field,:Field))
    @eval ($op)(a::$T1, b::$T2) = broadcast($op,($T1==$T2 ? promote : tuple)(a,b)...)
end

# multiplication/division is not strictly defined for abstract vectors, but
# make it work anyway if the two fields are exactly the same type, in which case
# its clear we wanted broadcasted multiplication/division. 
for op in (:*, :/)
    @eval ($op)(A::F, B::F) where {F<:Field} = broadcast($op, A, B)
end
