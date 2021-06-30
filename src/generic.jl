

### abstract type hierarchy

abstract type Basis end
abstract type Basislike <: Basis end

# the basis in which derivatives are sparse for a given field
struct DerivBasis <: Basislike end
const Ð! = DerivBasis
const Ð  = DerivBasis

# the basis in which lensing is a pixel remapping for a given field
struct LenseBasis <: Basislike end
const Ł! = LenseBasis
const Ł  = LenseBasis

# the nearest harmonic basis, e.g. for anything QU its QUFourier and
# for anything EB its EBFourier
abstract type HarmonicBasis <: Basislike end


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
abstract type BasisProd{Bs} <: Basis end

struct Basis2Prod{B₁,B₂}    <: BasisProd{Tuple{B₁,B₂}} end
struct Basis3Prod{B₁,B₂,B₃} <: BasisProd{Tuple{B₁,B₂,B₃}} end

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

# handy for picking out anything Map/Fourier
const SpatialBasis{B,I,P} = Union{B, Basis2Prod{P,B}, Basis3Prod{I,P,B}}


# handy aliases
basis_aliases = [
    "Map"        => Map,
    "Fourier"    => Fourier,
    "QUMap"      => QUMap,
    "QUFourier"  => QUFourier,
    "EBMap"      => EBMap,
    "EBFourier"  => EBFourier,
    "IQUMap"     => IQUMap,
    "IQUFourier" => IQUFourier,
    "IEBMap"     => IEBMap,
    "IEBFourier" => IEBFourier,
    "S0"         => Union{Map,Fourier},
    "QU"         => Union{QUMap,QUFourier},
    "EB"         => Union{EBMap,EBFourier},
    "S2Map"      => Union{QUMap,EBMap},
    "S2Fourier"  => Union{QUFourier,EBFourier},
    "S2"         => Union{QUMap,QUFourier,EBMap,EBFourier},
    "IQU"        => Union{IQUMap,IQUFourier},
    "IEB"        => Union{IEBMap,IEBFourier},
    "S02Map"     => Union{IQUMap,IEBMap},
    "S02Fourier" => Union{IQUFourier,IEBFourier},
    "S02"        => Union{IQUMap,IQUFourier,IEBMap,IEBFourier},
    "Field"      => Any
]

# Enumerates all the fields types like FlatMap, FlatFourier, etc...
# for a field_name like "Flat" and a bound on the M type parameter in
# BaseField{B,M,T,A}. Note: the seemingly-redundant <:AbstractArray{T}
# in the argument (which is enforced in BaseField anyway) is there to
# help prevent method ambiguities
function make_field_aliases(field_root, M_bound; export_names=true)
    for (basis_alias, B) in basis_aliases
        F = Symbol(field_root, basis_alias)
        if isconcretetype(B)
            @eval const $F{       M<:$M_bound, T, A<:AbstractArray{T}} = BaseField{$B, M, T, A}
        else
            @eval const $F{B<:$B, M<:$M_bound, T, A<:AbstractArray{T}} = BaseField{B,  M, T, A}
        end
        if export_names
            @eval export $F
        end
    end
end

# for printing
for (alias,B) in basis_aliases
    if isconcretetype(B)
        @eval typealias(::Type{$B}) = $alias
    end
end


## generic promotion rules which might change basis
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
    @eval promote_basis_generic_rule(::$B₁, ::$B₂) = $B()
end
promote_basis_generic_rule(::B, ::B) where {B<:Basis} = B()
promote_basis_generic_rule(::Any, ::Any) = Unknown()
promote_basis_generic(x, y) = select_known_rule(promote_basis_generic_rule, x, y)
unknown_rule_error(::typeof(promote_basis_generic_rule), ::B₁, ::B₂) where {B₁, B₂} = 
    error("Can't promote fields in $(typealias(B₁)) and $(typealias(B₂)) bases.")


## stricter rules used only in broadcasting
promote_basis_strict_rule(::B,   ::B )        where {B <:Basis}                                                    = B()
promote_basis_strict_rule(::B,   ::Basislike) where {B <:Basis}                                                    = B()
promote_basis_strict_rule(::B₂,  ::B₀)        where {B₀<:Union{Map,Fourier}, B₂ <:Basis2Prod{  <:Union{𝐐𝐔,𝐄𝐁},B₀}} = B₂()
promote_basis_strict_rule(::B₀₂, ::B₀)        where {B₀<:Union{Map,Fourier}, B₀₂<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁},B₀}} = B₀₂()
promote_basis_strict_rule(::Any, ::Any)                                                                            = Unknown()
promote_basis_strict(x, y) = select_known_rule(promote_basis_strict_rule, x, y)
unknown_rule_error(::typeof(promote_basis_strict_rule), ::B₁, ::B₂) where {B₁, B₂} = 
    error("Can't broadcast fields in $(typealias(B₁)) and $(typealias(B₂)) bases.")



## applying bases

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
basis(::AbstractVector) = Basis

### printing
typealias(::Type{B}) where {B<:Basis} = string(B.name.name)
Base.show_datatype(io::IO, t::Type{<:Union{Field,FieldOp}}) = print(io, typealias(t))
Base.isempty(::ImplicitOp) = true
Base.isempty(::ImplicitField) = true
Base.isempty(::Diagonal{<:Any,<:ImplicitField}) = true
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

# without this, *sometimes* IJulia doesnt print the field types right, but I dont really understand it
@init @require IJulia="7073ff75-c697-5162-941a-fcdaad2a7d2a" begin
    Base.show(io::IOContext{IOBuffer}, t::Type{<:Union{Field,FieldOp}}) = print(io, typealias(t))
end


### logdet

"""
    logdet(L::FieldOp, θ)
    
If `L` depends on `θ`, evaluates `logdet(L(θ))` offset by its fiducial value at
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
function simulate(Σ; rng=global_rng_for(Σ), seed=nothing, kwargs...)
    isnothing(seed) || seed!(rng, seed)
    simulate(rng, Σ; kwargs...)
end
function white_noise(ξ; rng=global_rng_for(ξ), seed=nothing, kwargs...)
    isnothing(seed) || seed!(rng, seed)
    white_noise(ξ, rng; kwargs...)
end
function fixed_white_noise(ξ; rng=global_rng_for(Σ), seed=nothing, kwargs...)
    isnothing(seed) || seed!(rng, seed)
    fixed_white_noise(ξ, rng; kwargs...)
end


global_rng_for(x::T) where {T<:AbstractArray} = global_rng_for(T)
global_rng_for(::Type{<:Array}) = Random.default_rng()



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

function ≈(x::Field{<:Any,T}, y::Field{<:Any,S}; atol=0, rtol=Base.rtoldefault(T,S,atol)) where {T,S}
    _norm(x) = norm(unbatch(norm(x)))
    _norm(x - y) <= max(atol, rtol*max(_norm(x), _norm(y)))
end

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
print_array(io::IO, f::Field) = !isempty(f) && print_array(io, f[:])
show_vector(io::IO, f::Field) = !isempty(f) && show_vector(io, f[:])
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


# package code should use batch_index(f, 1), but for interactive work its
# convenient to be able to do f[!,1]
getindex(x::Union{Real,Field,FieldOp}, ::typeof(!), I) = batch_index(x, I)


# adapt_structure(to, x::Union{ImplicitOp,ImplicitField}) = x
# adapt_structure(to, arr::Array{<:Field}) = map(f -> adapt(to,f), arr)


one(f::Field) = fill!(similar(f), one(eltype(f)))
norm(f::Field) = sqrt(dot(f,f))
# sum_kbn(f::Field) = sum_kbn(f[:])

@init @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
    # never try to auto-convert Fields or FieldOps to Python arrays
    PyCall.PyObject(x::Union{FieldOp,Field}) = PyCall.pyjlwrap_new(x)
end