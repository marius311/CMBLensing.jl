

### abstract type hierarchy

abstract type Basis end
abstract type Basislike <: Basis end

struct DerivBasis <: Basislike end # Basis in which derivatives are sparse for a given field
const Ð! = DerivBasis
const Ð  = DerivBasis
struct LenseBasis <: Basislike end # Basis in which lensing is a pixel remapping for a given field
const Ł! = LenseBasis
const Ł  = LenseBasis

## fields
abstract type Field{B<:Basis,T} <: AbstractVector{T} end
abstract type ImplicitField{B<:Basis,T} <: Field{B,T} end

## linear operators
abstract type ImplicitOp{T} <: AbstractMatrix{T} end
const DiagOp{F<:Field,T} = Diagonal{T,F}
# includes Diagonal and Adjoint wrappers, which can't be made <:ImplicitOp directly
const FieldOp{T} = Union{ImplicitOp{T},Adjoint{T,<:ImplicitOp{T}},Diagonal{T,<:Field{<:Any,T}}}

# scalars which are allowed in our expressions must be real because we
# implicitly assume our maps are real, and addition/multiplication by a complex
# number, even of the fourier transform, would break this.
const Scalar = Real

# other useful unions
const FieldOrOp   = Union{Field,FieldOp}
const FieldOpScal = Union{Field,FieldOp,Scalar}

## basis types
abstract type BasisProd <: Basis end
struct Basis2Prod{B₁,B₂}    <: BasisProd end
struct Basis3Prod{B₁,B₂,B₃} <: BasisProd end

struct Map     <: Basis end
struct Fourier <: Basis end
struct 𝐈       <: Basis end
struct 𝐐𝐔      <: Basis end
struct 𝐄𝐁      <: Basis end

const QUMap      = Basis2Prod{    𝐐𝐔, Map     }
const EBMap      = Basis2Prod{    𝐄𝐁, Map     }
const QUFourier  = Basis2Prod{    𝐐𝐔, Fourier }
const EBFourier  = Basis2Prod{    𝐄𝐁, Fourier }
const IQUMap     = Basis3Prod{ 𝐈, 𝐐𝐔, Map     }
const IEBMap     = Basis3Prod{ 𝐈, 𝐄𝐁, Map     }
const IQUFourier = Basis3Prod{ 𝐈, 𝐐𝐔, Fourier }
const IEBFourier = Basis3Prod{ 𝐈, 𝐄𝐁, Fourier }

# for printing
for F in ["QUMap", "EBMap", "QUFourier", "EBFourier", "IQUMap", "IEBMap", "IQUFourier", "IEBFourier"]
    @eval typealias(::Type{$(Symbol(F))}) = $F
end


### basis

basis_promotion_rules = Dict(
    # spin-0
    (Map,        Fourier)     => Map,
    # spin-2
    (QUMap,      QUFourier)   => QUMap,
    (EBMap,      EBFourier)   => EBFourier,
    (QUMap,      EBMap)       => QUMap,
    (QUFourier,  EBFourier)   => QUFourier,
    (QUMap,      EBFourier)   => QUMap,
    (QUFourier,  EBMap)       => QUFourier,
    # spin-(0,2)
    (IQUMap,     IQUFourier)  => IQUMap,
    (IEBMap,     IEBFourier)  => IEBFourier,
    (IQUMap,     IEBMap)      => IQUMap,
    (IQUFourier, IEBFourier)  => IQUFourier,
    (IQUMap,     IEBFourier)  => IQUMap,
    (IQUFourier, IEBMap)      => IQUFourier
)
for ((B₁,B₂),B) in basis_promotion_rules
    @eval promote_rule_generic(::$B₁, ::$B₂) = $B()
end
promote_rule_generic(b::B,::B) where {B<:Basis} = b
promote_rule_generic(::Any,::Any) = Unknown()
promote_generic(x,y) = _select_known_rule(x, y, promote_rule_generic(x,y), promote_rule_generic(y,x))


# a spin-0 can broadcast with a spin-2 or spin-(0,2) as long the Map/Fourier part is the same
promote_bcast_rule(b::B,   ::B ) where {B <:Basis} = b
promote_bcast_rule(b::B₂,  ::B₀) where {B₀<:Union{Map,Fourier}, B₂ <:Basis2Prod{  <:Union{𝐐𝐔,𝐄𝐁},B₀}} = b
promote_bcast_rule(b::B₀₂, ::B₀) where {B₀<:Union{Map,Fourier}, B₀₂<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁},B₀}} = b

# Map(B) or Fourier(B) for another basis, B
(::Type{B})(::Type{B′}) where {B<:Union{Map,Fourier}, B′<:Union{Map,Fourier}} = B
(::Type{B})(::Type{Basis2Prod{  Pol,B′}}) where {Pol, B<:Union{Map,Fourier}, B′<:Union{Map,Fourier}} = Basis2Prod{  Pol,B}
(::Type{B})(::Type{Basis3Prod{𝐈,Pol,B′}}) where {Pol, B<:Union{Map,Fourier}, B′<:Union{Map,Fourier}} = Basis3Prod{𝐈,Pol,B}

# A "basis-like" object, e.g. the lensing basis Ł or derivative basis Ð. For any
# particular types of fields, these might be different actual bases, e.g. the
# lensing basis is Map for S0 but QUMap for S2.
(::Type{B})(f::F) where {F<:Field,B<:Basislike} = B(F)(f)
(::Type{B})(f′::Field, f::F) where {F<:Field,B<:Basislike} = B(F)(f′,f)
(::Type{B})(a::AbstractArray{<:Field}...) where {B<:Basis} = B.(a...)

# B(f) where B is a basis converts f to that basis. This is the fallback if the
# field is already in the right basis.
(::Type{B})(f::Field{B}) where {B<:Basis} = f

# The abstract `Basis` type means "any basis", hence this conversion rule:
Basis(f::Field) = f

# B(f′, f) converts f to basis B and stores the result inplace in f′. If f is
# already in basis B, we just return f (but note, we never actually set f′ in
# this case, which is more efficient, but necessitates some care when using this
# construct)
(::Type{B})(f′::Field{B}, f::Field{B}) where {B<:Basis} = f

# convenience "getter" functions for the B basis type parameter
basis(f::F) where {F<:Field} = basis(F)
basis(::Type{<:Field{B}}) where {B<:Basis} = B
basis(::Type{<:Field}) = Basis


### printing
typealias(::Type{B}) where {B<:Basis} = string(B.name.name)
Base.show_datatype(io::IO, t::Type{<:Union{Field,FieldOp}}) = print(io, typealias(t))
Base.isempty(::ImplicitOp) = true
Base.isempty(::ImplicitField) = true
function Base.summary(io::IO, x::FieldOp)
    try
        print(io, join(size(x), "×"), " ")
    catch err
        @assert err isa MethodError
        print(io, "⍰×⍰ ")
    end
    Base.showarg(io, x, true)
end
function Base.summary(io::IO, x::ImplicitField)
    try
        print(io, size(x,1), "-element ")
    catch err
        @assert err isa MethodError
        print(io, "⍰-element ")
    end
    Base.showarg(io, x, true)
end


### logdet

"""
    logdet(L::FieldOp, θ)
    
If L depends on θ, evaluates `logdet(L(θ))` offset by its fiducial value at
`L()`. Otherwise, returns 0.
"""
logdet(L::FieldOp, θ) = depends_on(L,θ) ? logdet(L()\L(θ)) : 0
logdet(L::Int, θ) = 0 # the default returns Float64 which unwantedly poisons the backprop to Float64
logdet(L, θ) = logdet(L)



### Simulation

@doc doc"""
    simulate(Σ;     rng=global_rng_for(Σ), seed=nothing)
    
Draw a simulation from the covariance matrix `Σ`, i.e. draw a random vector
$\xi$ such that the covariance $\langle \xi \xi^\dagger \rangle = \Sigma$. 

The random number generator `rng` will be used and advanced in the proccess, and
is by default the appropriate one depending on if `Σ` is backed by `Array` or
`CuArray`.

The `seed` argument can also be used to seed the `rng`.
"""
simulate(         Σ; rng=global_rng_for(Σ), seed=nothing, kwargs...) = (seed!(rng, seed); simulate(rng, Σ; kwargs...))
white_noise(      ξ; rng=global_rng_for(ξ), seed=nothing, kwargs...) = (seed!(rng, seed); white_noise(ξ, rng; kwargs...))
fixed_white_noise(ξ; rng=global_rng_for(Σ), seed=nothing, kwargs...) = (seed!(rng, seed); fixed_white_noise(ξ, rng; kwargs...))


global_rng_for(x::T) where {T<:AbstractArray} = global_rng_for(T)
global_rng_for(::Type{<:Array}) = Random.GLOBAL_RNG



### spin adjoints
# represents a field which is adjoint over just the "spin" indices.
# multiplying such a field by a non-adjointed one should be the inner
# product over just the spin indices, hence return a spin-0 field,
# rather than a scalar. note: these are really only lightly used in
# one place in LenseFlow, so they have almost no real functionality
# for now.
struct SpinAdjoint{F<:Field}
    f :: F
end
spin_adjoint(f::Field) = SpinAdjoint(f)



### misc

# allows L(θ) or L(ϕ) to work and be a no-op for things which don't
# depend on parameters or a field
(L::Union{FieldOp,UniformScaling})(::Union{Field,NamedTuple}) = L

# allow using `I` as a lensing operator to represent no lensing
alloc_cache(x, ::Any) = x
cache(x, ::Any) = x
cache!(x, ::Any) = x

# caching for adjoints
cache(L::Adjoint, f) = cache(L',f)'
cache!(L::Adjoint, f) = cache!(L',f)'

# todo: fix this
*(::UniformScaling{Bool}, L::FieldOp) = L

# we use Field cat'ing mainly for plotting, e.g. plot([f f; f f]) plots a 2×2
# matrix of maps. the following definitions make it so that Fields aren't
# splatted into a giant matrix when doing [f f; f f] (which they would othewise
# be since they're Arrays)
hvcat(rows::Tuple{Vararg{Int}}, values::Field...) = hvcat(rows, ([x] for x in values)...)
hcat(values::Field...) = hcat(([x] for x in values)...)

### printing
print_array(io::IO, f::Field) = print_array(io, f[:])
show_vector(io::IO, f::Field) = show_vector(io, f[:])
Base.has_offset_axes(::Field) = false # needed for Diagonal(::Field) if the Field is implicitly-sized


# addition/subtraction works between any fields and scalars, promotion is done
# automatically if fields are in different bases
for op in (:+,:-), (T1,T2,promote) in ((:Field,:Scalar,false),(:Scalar,:Field,false),(:Field,:Field,true))
    @eval ($op)(a::$T1, b::$T2) = broadcast($op, ($promote ? promote(a,b) : (a,b))...)
end

# multiplication/division is not strictly defined for abstract vectors, but
# make it work anyway if the two fields are exactly the same type, in which case
# its clear we wanted broadcasted multiplication/division. 
for op in (:*, :/)
    @eval ($op)(a::Field{B}, b::Field{B}) where {B} = broadcast($op, a, b)
end

# needed unless I convince them to undo the changes here:
# https://github.com/JuliaLang/julia/pull/35257#issuecomment-657901503
if VERSION>v"1.4"
    *(x::Adjoint{<:Number,<:Field}, y::Field) = dot(x.parent,y)
end


# adapt_structure(to, x::Union{ImplicitOp,ImplicitField}) = x
# adapt_structure(to, arr::Array{<:Field}) = map(f -> adapt(to,f), arr)


one(f::Field) = fill!(similar(f), one(eltype(f)))
norm(f::Field) = sqrt(dot(f,f))
# sum_kbn(f::Field) = sum_kbn(f[:])

@init @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
    # never try to auto-convert Fields or FieldOps to Python arrays
    PyCall.PyObject(x::Union{FieldOp,Field}) = PyCall.pyjlwrap_new(x)
end