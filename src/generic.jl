
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

# the basis type for a FieldTuple can be a BasisTuple which just holds the bases
# of the sub-fields
abstract type BasisTuple{T} <: Basis end
promote_type(::Type{BasisTuple{BT1}}, ::Type{BasisTuple{BT2}}) where {BT1,BT2} = BasisTuple{Tuple{map_tupleargs(promote_type,BT1,BT2)...}}

# Basis types
abstract type Map       <: Basis end
abstract type Fourier   <: Basis end
abstract type QUMap     <: Basis end
abstract type EBMap     <: Basis end
abstract type QUFourier <: Basis end
abstract type EBFourier <: Basis end
const IQUFourier = BasisTuple{Tuple{Fourier,QUFourier}}
const IEBFourier = BasisTuple{Tuple{Fourier,EBFourier}}
const IQUMap     = BasisTuple{Tuple{Map,QUMap}}
const IEBMap     = BasisTuple{Tuple{Map,EBMap}}



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
size(::Union{ImplicitOp,ImplicitField},i) = nothing
length(::Union{ImplicitOp,ImplicitField}) = 0
checksquare(::ImplicitOp) = nothing


adapt_structure(to, x::Union{ImplicitOp,ImplicitField}) = x
adapt_structure(to, arr::Array{<:Field}) = map(f -> adapt(to,f), arr)


# printing
show(io::IO, ::MIME"text/plain", L::ImplicitOp) = show(io,L)
show(io::IO, ::MIME"text/plain", L::Adjoint{<:Any,<:ImplicitOp}) = show(io,L)
show(io::IO, L::Adjoint{<:Any,<:ImplicitOp}) = (print(io,"Adjoint{"); show(io,parent(L)); print(io,"}"))
# this is the main function ImplicitOps should specialize if this default behavior isn't enough:
show(io::IO, L::ImplicitOp) = showarg(io, L, true)

# All CMBLensing operators are then either Diagonals or ImplicitOps.
# ImplicitOrAdjOp are things for which algebra is done lazily. This used to
# include Diagonal{<:ImplicitField}, but that was leading so some really
# annoying ambiguities, so its removed for now.
const DiagOp{F<:Field,T} = Diagonal{T,F}
const LinOp{B,S,P} = Union{ImplicitOp{B,S,P},DiagOp{<:Field{B,S,P}}}
const ImplicitOrAdjOp{B,S,P} = Union{ImplicitOp{B,S,P}, Adjoint{<:Any,<:ImplicitOp{B,S,P}}}
const LinOrAdjOp{B,S,P} = Union{ImplicitOrAdjOp{B,S,P},DiagOp{<:Field{B,S,P}}}

# assume no dependence on parameters θ unless otherwise specified
(L::LinOrAdjOp)(θ::NamedTuple) = L


### Scalars
# scalars which are allowed in our expressions must be real because we
# implicitly assume our maps are real, and addition/multiplication by a complex
# number, even of the fourier transform, would break this.
const Scalar = Real
const FieldOrOp = Union{Field,LinOp}
const FieldOpScal = Union{Field,LinOp,Scalar}


"""
    logdet(L::LinOp, θ)
    
If L depends on θ, evaluates `logdet(L(θ))` offset by its fiducial value at
`L()`. Otherwise, returns 0.
"""
logdet(L::LinOp, θ) = depends_on(L,θ) ? logdet(L()\L(θ)) : 0
logdet(L::Int, θ) = 0 # the default returns Float64 which unwantedly poisons the backprop to Float64
logdet(L, θ) = logdet(L)


### Simulation

@doc doc"""
    simulate(Σ; rng=global_rng_for(Σ), seed=nothing)
    
Draw a simulation from the covariance matrix `Σ`, i.e. draw a random vector
$\xi$ such that the covariance $\langle \xi \xi^\dagger \rangle = \Sigma$. 

The random number generator `rng` will be used and advanced in the proccess, and
is by default the appropriate one depending on if `Σ` is backed by `Array` or
`CuArray`.

The `seed` argument can also be used to seed the `rng`.
"""
function simulate(Σ; rng=global_rng_for(Σ), seed=nothing)
    (seed != nothing) && Random.seed!(rng, seed)
    simulate(rng, Σ)
end
function white_noise(Σ; rng=global_rng_for(Σ), seed=nothing)
    (seed != nothing) && Random.seed!(rng, seed)
    white_noise(rng, Σ)
end
global_rng_for(x::T) where {T<:AbstractArray} = global_rng_for(T)
global_rng_for(::Type{<:Array}) = Random.GLOBAL_RNG


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

# allows one to pass `1` to something which expect a LenseFlow-like operator
(s::Scalar)(::Field) = s
alloc_cache(x, ::Any) = x
cache(x, ::Any) = x
cache!(x, ::Any) = x

# caching for adjoints
cache(L::Adjoint, f) = cache(L',f)'
cache!(L::Adjoint, f) = cache!(L',f)'

# convenience "getter" functions for the Basis/Spin/Pix
# basis of UnionAlls like basis(Field) will return Basis (which means any Basis)
basis(::Type{<:Field}) = Basis
basis(::Type{<:Field{B,S,P}}) where {B<:Basis,S<:Spin,P<:Pix} = B
basis(::F) where {F<:Field} = basis(F)
spin(::Type{<:Field}) = Spin
spin(::Type{<:Field{B,S,P}}) where {B<:Basis,S<:Spin,P<:Pix} = S
spin(::F) where {F<:Field} = spin(F)
pix(::Type{<:Field}) = Pix
pix(::Type{<:Field{B,S,P}}) where {B<:Basis,S<:Spin,P<:Pix} = P
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
show_vector(io::IO, f::ImplicitField) = print(io, "[…]")


# addition/subtraction works between any fields and scalars, promotion is done
# automatically if fields are in different bases
for op in (:+,:-), (T1,T2,promote) in ((:Field,:Scalar,false),(:Scalar,:Field,false),(:Field,:Field,true))
    @eval ($op)(a::$T1, b::$T2) = broadcast($op, ($promote ? promote(a,b) : (a,b))...)
end

≈(a::Field, b::Field) = ≈(promote(a,b)...)

# multiplication/division is not strictly defined for abstract vectors, but
# make it work anyway if the two fields are exactly the same type, in which case
# its clear we wanted broadcasted multiplication/division. 
for op in (:*, :/)
    @eval ($op)(a::Field{B,S,P}, b::Field{B,S,P}) where {B,S,P} = broadcast($op, a, b)
end

# needed unless I convince them to undo the changes here:
# https://github.com/JuliaLang/julia/pull/35257#issuecomment-657901503
if VERSION>v"1.4"
    *(x::Adjoint{<:Number,<:Field}, y::Field) = dot(x.parent,y)
end

(::Type{T})(f::Field{<:Any,<:Any,<:Any,<:Real}) where {T<:Real} = T.(f)
(::Type{T})(f::Field{<:Any,<:Any,<:Any,<:Complex}) where {T<:Real} = Complex{T}.(f)

# misc
one(f::Field) = fill!(similar(f), one(eltype(f)))
norm(f::Field) = sqrt(dot(f,f))


function invalid_broadcast_error(B1,F1,B2,F2)
    if B1!=B2
        error("""Can't broadcast across fields in $B1 and $B2 bases.
        Try the same operation without broadcasting (which will do an automatic basis conversion).""")
    else
        error("""Broadcasting across fields with the following differing broadcast styles is not implemented:
        * $F1
        * $F2
        """)
    end
end


@init @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
    # never try to auto-convert Fields or LinOps to Python arrays
    PyCall.PyObject(x::Union{LinOp,Field}) = PyCall.pyjlwrap_new(x)
end
