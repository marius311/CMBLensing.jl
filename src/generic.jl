

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

abstract type S0Basis <: Basis end
abstract type PolBasis <: Basis end

struct Map     <: S0Basis end
struct Fourier <: S0Basis end
struct 𝐈       <: PolBasis end
struct 𝐐𝐔      <: PolBasis end
struct 𝐄𝐁      <: PolBasis end

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

# used in implmenting things like f.Ix, f.Q, etc.. for fields
@generated function pol_index(::B, ::Val{k}) where {B,k}
    i = @match B() begin
        _::Map         => @match k begin (:Ix||:I) => 1 end
        _::Fourier     => @match k begin (:Il||:I) => 1 end
        _::QUMap       => @match k begin (:Qx||:Q) => 1; (:Ux||:U) => 2 end
        _::QUFourier   => @match k begin (:Ql||:Q) => 1; (:Ul||:U) => 2 end
        _::EBMap       => @match k begin (:Ex||:E) => 1; (:Bx||:B) => 2 end
        _::EBFourier   => @match k begin (:El||:E) => 1; (:Bl||:B) => 2 end
        _::IQUMap      => @match k begin (:Ix||:I) => 1; (:Qx||:Q) => 2; (:Ux||:U) => 3 end
        _::IQUFourier  => @match k begin (:Il||:I) => 1; (:Ql||:Q) => 2; (:Ul||:U) => 3 end
        _::IEBMap      => @match k begin (:Ix||:I) => 1; (:Ex||:E) => 2; (:Bx||:B) => 3 end
        _::IEBFourier  => @match k begin (:Il||:I) => 1; (:El||:E) => 2; (:Bl||:B) => 3 end
    end
    i == nothing ? error("A $B doesn't have a $k property") : i
end

## applying and converting bases

LenseBasis(                            ::S0Basis)  = Map
LenseBasis(::Basis2Prod{   <:PolBasis, <:S0Basis}) = QUMap
LenseBasis(::Basis3Prod{𝐈, <:PolBasis, <:S0Basis}) = IQUMap
DerivBasis(                            ::S0Basis)  = Fourier
DerivBasis(::Basis2Prod{   <:PolBasis, <:S0Basis}) = QUFourier
DerivBasis(::Basis3Prod{𝐈, <:PolBasis, <:S0Basis}) = IQUFourier
HarmonicBasis(                         ::S0Basis)  = Fourier
HarmonicBasis(::Basis2Prod{𝐐𝐔,         <:S0Basis}) = QUFourier
HarmonicBasis(::Basis2Prod{𝐄𝐁,         <:S0Basis}) = EBFourier
HarmonicBasis(::Basis3Prod{𝐈, 𝐐𝐔,      <:S0Basis}) = IQUFourier
HarmonicBasis(::Basis3Prod{𝐈, 𝐄𝐁,      <:S0Basis}) = IEBFourier

# B(::Basis) converts the basis type, e.g. Map(QUFourier()) = QUMap
(::Type{B})(::B′) where {B<:S0Basis, B′<:S0Basis} = B
(::Type{B})(::Basis2Prod{  Pol,B′}) where {Pol, B<:S0Basis, B′<:S0Basis} = Basis2Prod{  Pol,B}
(::Type{B})(::Basis3Prod{𝐈,Pol,B′}) where {Pol, B<:S0Basis, B′<:S0Basis} = Basis3Prod{𝐈,Pol,B}


# B(f::Field) where B is a basis converts f to that basis. This is the
# fallback if the field is already in the right basis.
(::Type{B})(f::Field{B}) where {B<:Basis} = f
# This is the fallback if no conversion is explicilty defined, which
# tries to first convert the basis, e.g. Map(f::BaseQUFourier) becomes
# Map(QUFourier())(f) which calls QUMap(f). This is also used for
# Basislike objects like LenseBasis(...)
(::Type{B})(f::Field{B′}) where {B′<:Basis,B<:Basis} = B(B′())(f)

# In-place version of the above
(::Type{B})(dst::Field, src::Field{B′}) where {B′<:Basis,B<:Basis} = B(B′())(dst,src)
# And the case where its already in the right basis (but note, we
# never actually set f′ in this case, which is more efficient, but
# necessitates some care when using this construct)
(::Type{B})(dst::Field{B}, src::Field{B}) where {B<:Basis} = src


# Basis conversion automatically maps over arrays
(::Type{B})(a::AbstractArray{<:Field}) where {B<:Basis} = B.(a)
(::Type{B})(dst::AbstractArray{<:Field}, src::AbstractArray{<:Field}) where {B<:Basis} = B.(dst,src)

# The abstract `Basis` type means "any basis", hence this conversion rule:
Basis(f) = f


# used in make_field_aliases below
basis_aliases = OrderedDict(
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
    "S0"         => S0Basis,
    "QU"         => Basis2Prod{   𝐐𝐔        , <:S0Basis},
    "EB"         => Basis2Prod{   𝐄𝐁        , <:S0Basis},
    "S2Map"      => Basis2Prod{   <:PolBasis, Map},
    "S2Fourier"  => Basis2Prod{   <:PolBasis, Fourier},
    "S2"         => Basis2Prod{   <:PolBasis, <:S0Basis},
    "IQU"        => Basis3Prod{𝐈, 𝐐𝐔        , <:S0Basis},
    "IEB"        => Basis3Prod{𝐈, 𝐄𝐁        , <:S0Basis},
    "S02Map"     => Basis3Prod{𝐈, <:PolBasis, Map},
    "S02Fourier" => Basis3Prod{𝐈, <:PolBasis, Fourier},
    "S02"        => Basis3Prod{𝐈, <:PolBasis, <:S0Basis},
    "Field"      => Any
)

# Enumerates all the fields types like FlatMap, FlatFourier, etc...
# for a field_name like "Flat" and a bound on the M type parameter in
# BaseField{B,M,T,A}. Note: the seemingly-redundant <:AbstractArray{T}
# in the argument (which is enforced in BaseField anyway) is there to
# help prevent method ambiguities
function make_field_aliases(field_root, M_bound; export_names=true, extra_aliases=Dict())
    for (basis_alias, B) in merge(basis_aliases, extra_aliases)
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


## generic promotion rules
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



# convenience "getter" functions for the B basis type parameter
basis(f::F) where {F<:Field} = basis(F)
basis(::Type{<:Field{B}}) where {B<:Basis} = B
basis(::Type{<:Field}) = Basis
basis(::Union{Number,AbstractVector}) = Basis # allows them to be in FieldTuple


### printing
typealias(::Type{B}) where {B<:Basis} = string(B)
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


### logdet

"""
    logdet(L::FieldOp, θ)
    
If `L` depends on `θ`, evaluates `logdet(L(θ))` offset by its fiducial value at
`L()`. Otherwise, returns 0.
"""
logdet(L::FieldOp, θ) = depends_on(L,θ) ? logdet(L()\L(θ)) : 0
logdet(L::Union{Int,UniformScaling{Bool}}, θ) = 0 # the default returns Float64 which unwantedly poisons the backprop to Float64
logdet(L, θ) = logdet(L)



### Simulation

@doc doc"""
    simulate([rng], Σ)
    
Draw a simulation from the covariance matrix `Σ`, i.e. draw a random vector
$\xi$ such that the covariance $\langle \xi \xi^\dagger \rangle = \Sigma$. 

The random number generator `rng` will be used and advanced in the proccess, and
defaults to `Random.default_rng()`.
"""
simulate(Σ::FieldOp, args...) = simulate(Random.default_rng(), Σ, args...)


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