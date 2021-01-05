

### abstract type hierarchy

abstract type Basis end
abstract type Basislike <: Basis end
abstract type DerivBasis <: Basislike end # Basis in which derivatives are sparse for a given field
const Ð! = Ð = DerivBasis
abstract type LenseBasis <: Basislike end # Basis in which lensing is a pixel remapping for a given field
const Ł! = Ł = LenseBasis

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
const FieldOrOp = Union{Field,FieldOp}
const FieldOpScal = Union{Field,FieldOp,Scalar}

## Basis types
struct Map        <: Basis end
struct Fourier    <: Basis end
struct QUMap      <: Basis end
struct EBMap      <: Basis end
struct QUFourier  <: Basis end
struct EBFourier  <: Basis end
struct IQUMap     <: Basis end
struct IEBMap     <: Basis end
struct IQUFourier <: Basis end
struct IEBFourier <: Basis end


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
    @eval promote_rule(::Type{$B₁}, ::Type{$B₂}) = $B
end

_sub_basis = Dict(
    Map        => Dict("I"=>Map),
    Fourier    => Dict("I"=>Fourier),
    QUMap      => Dict("Q"=>Map,     "U"=>Map,     "P"=>QUMap),
    EBMap      => Dict("E"=>Map,     "B"=>Map,     "P"=>EBMap),
    QUFourier  => Dict("Q"=>Fourier, "U"=>Fourier, "P"=>QUFourier),
    EBFourier  => Dict("E"=>Fourier, "B"=>Fourier, "P"=>EBFourier),
    IQUMap     => Dict("I"=>Map,     "Q"=>Map,     "U"=>Map,     "P"=>QUMap,     "IP"=>IQUMap),
    IEBMap     => Dict("I"=>Map,     "E"=>Map,     "B"=>Map,     "P"=>EBMap,     "IP"=>IEBMap),
    IQUFourier => Dict("I"=>Fourier, "Q"=>Fourier, "U"=>Fourier, "P"=>QUFourier, "IP"=>IQUFourier),
    IEBFourier => Dict("I"=>Fourier, "E"=>Fourier, "B"=>Fourier, "P"=>EBFourier, "IP"=>IEBFourier)
)
sub_basis(B,pol) = _sub_basis[B][pol]


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
Base.isempty(::FieldOp) = true
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
    simulate(Σ; rng=global_rng_for(Σ), seed=nothing)
    
Draw a simulation from the covariance matrix `Σ`, i.e. draw a random vector
$\xi$ such that the covariance $\langle \xi \xi^\dagger \rangle = \Sigma$. 

The random number generator `rng` will be used and advanced in the proccess, and
is by default the appropriate one depending on if `Σ` is backed by `Array` or
`CuArray`.

The `seed` argument can also be used to seed the `rng`.
"""
function simulate(Σ, args...; rng=global_rng_for(Σ), seed=nothing)
    isnothing(seed) || Random.seed!(rng, seed)
    simulate(rng, Σ, args...)
end
function white_noise(Σ; rng=global_rng_for(Σ), seed=nothing)
    isnothing(seed) || Random.seed!(rng, seed)
    white_noise(rng, Σ)
end
function fixed_white_noise(Σ; rng=global_rng_for(Σ), seed=nothing)
    isnothing(seed) || Random.seed!(rng, seed)
    fixed_white_noise(rng, Σ)
end
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

≈(a::Field, b::Field) = ≈(promote(a,b)...)

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