# ### Diagonal ops

# we use Base.Diagonal(f) for diagonal operators so very little specific code is
# actually needed here. 

simulate(rng::AbstractRNG, D::DiagOp; Nbatch=nothing) = sqrt(D) * randn!(rng, similar(diag(D), Nbatch))

# automatic basis conversion (and NaN-zeroing)
(*)(D::DiagOp{<:Field{B}}, f::Field) where {B} = diag(D) .* B(f)
(\)(D::DiagOp{<:Field{B}}, f::Field) where {B} = nan2zero.(diag(D) .\ B(f))

# the generic versions of these kind of suck so we need these specializized
# versions:
(*)(x::Adjoint{<:Any,<:Field}, D::DiagOp) = (D*parent(x))'
(*)(x::Adjoint{<:Any,<:Field}, D::DiagOp, y::Field) = x*(D*y)
diag(D::DiagOp) = D.diag
(^)(D::DiagOp, p::Integer) = Diagonal(diag(D).^p)
pinv(D::DiagOp) = Diagonal(pinv.(diag(D)))
(≈)(D₁::DiagOp, D₂::DiagOp) = diag(D₁) ≈ diag(D₂)


# use getproperty here to ensure no basis conversion is done
getindex(D::DiagOp, s::Symbol) = Diagonal(getproperty(diag(D),s))

# the generic version of this is prohibitively slow so we need this
hash(D::DiagOp, h::UInt64) = foldr(hash, (typeof(D), D.diag), init=h)

# adapting
get_storage(D::DiagOp) = get_storage(diag(D))

# allows broadcasting over DiagOps
struct DiagOpStyle{S} <: AbstractArrayStyle{2} end
BroadcastStyle(::Type{Diagonal{T,F}}) where {T,F<:Field} = DiagOpStyle{typeof(BroadcastStyle(F))}()
BroadcastStyle(::DiagOpStyle{S₁}, ::DiagOpStyle{S₂}) where {S₁,S₂} = DiagOpStyle{typeof(result_style(S₁(), S₂()))}()
materialize(bc::Broadcasted{DiagOpStyle{S}}) where {S} = Diagonal(materialize(convert(Broadcasted{S}, preprocess(DiagOpStyle{S}(), bc))))
materialize!(dest::DiagOp, bc::Broadcasted{DiagOpStyle{S}}) where {S} = 
    (materialize!(diag(dest), convert(Broadcasted{S}, preprocess(DiagOpStyle{S}(), bc))); dest)
preprocess(::DiagOpStyle, D::Diagonal) = diag(D)

if VERSION >= v"1.7-beta"
    # https://github.com/JuliaLang/julia/pull/37898#issuecomment-877755178
    LinearAlgebra.mat_mat_scalar(A::DiagOp, B::DiagOp, γ::Number) = (A * B) * γ
end



# ### BlockDiagIEB
# 
# A TEB covariance of the form:
# 
#    [ΣTT ΣTE  ⋅
#     ΣTE ΣEE  ⋅
#      ⋅   ⋅  ΣBB]
# 
# We store the 2x2 block as a 2x2 SMatrix, ΣTE, so that we can easily
# call sqrt/inv on it, and the ΣBB block separately as ΣB. This type
# is generic with regards to the field type, F.
struct BlockDiagIEB{T,F} <: ImplicitOp{T}
    ΣTE :: SMatrix{2,2,Diagonal{T,F},4}
    ΣB :: Diagonal{T,F}
end
# applying
*(L::BlockDiagIEB, f::BaseS02) =       L * IEBFourier(f)
\(L::BlockDiagIEB, f::BaseS02) = pinv(L) * IEBFourier(f)
function *(L::BlockDiagIEB, f::BaseIEBFourier)
    (i,e),b = (L.ΣTE * [f.I, f.E]), L.ΣB * f.B
    BaseIEBFourier(i,e,b)
end
# manipulating
size(L::BlockDiagIEB) = 3 .* size(L.ΣB)
adjoint(L::BlockDiagIEB) = L
sqrt(L::BlockDiagIEB) = BlockDiagIEB(sqrt(L.ΣTE), sqrt(L.ΣB))
pinv(L::BlockDiagIEB) = BlockDiagIEB(pinv(L.ΣTE), pinv(L.ΣB))
diag(L::BlockDiagIEB) = BaseIEBFourier(L.ΣTE[1,1].diag, L.ΣTE[2,2].diag, L.ΣB.diag)
similar(L::BlockDiagIEB) = BlockDiagIEB(similar.(L.ΣTE), similar(L.ΣB))
get_storage(L::BlockDiagIEB) = get_storage(L.ΣB)
simulate(rng::AbstractRNG, L::BlockDiagIEB; Nbatch=nothing) = sqrt(L) * randn!(rng, similar(diag(L), Nbatch))
# arithmetic
*(L::BlockDiagIEB, D::DiagOp{<:BaseIEBFourier}) = BlockDiagIEB(SMatrix{2,2}(L.ΣTE * [[D[:I]] [0]; [0] [D[:E]]]), L.ΣB * D[:B])
+(L::BlockDiagIEB, D::DiagOp{<:BaseIEBFourier}) = BlockDiagIEB(@SMatrix[L.ΣTE[1,1]+D[:I] L.ΣTE[1,2]; L.ΣTE[2,1] L.ΣTE[2,2]+D[:E]], L.ΣB + D[:B])
*(La::F, Lb::F) where {F<:BlockDiagIEB} = F(La.ΣTE * Lb.ΣTE, La.ΣB * Lb.ΣB)
+(La::F, Lb::F) where {F<:BlockDiagIEB} = F(La.ΣTE + Lb.ΣTE, La.ΣB + Lb.ΣB)
+(L::BlockDiagIEB, U::UniformScaling{<:Scalar}) = BlockDiagIEB(@SMatrix[(L.ΣTE[1,1]+U) L.ΣTE[1,2]; L.ΣTE[2,1] (L.ΣTE[2,2]+U)], L.ΣB+U)
*(L::BlockDiagIEB, λ::Scalar) = BlockDiagIEB(L.ΣTE * λ, L.ΣB * λ)
*(D::DiagOp{<:BaseIEBFourier}, L::BlockDiagIEB) = L * D
+(U::UniformScaling{<:Scalar}, L::BlockDiagIEB) = L + U
*(λ::Scalar, L::BlockDiagIEB) = L * λ
# indexing
function getindex(L::BlockDiagIEB, k::Symbol)
    @match k begin
        :IP => L
        :I => L.ΣTE[1,1]
        :E => L.ΣTE[2,2]
        :B => L.ΣB
        :P => Diagonal(CartesianEBFourier(L[:E].diag, L[:B].diag))
        (:QQ || :UU || :QU || :UQ) => getindex(L[:P], k)
        _ => throw(ArgumentError("Invalid BlockDiagIEB index: $k"))
    end
end



### Derivative ops

# These ops just store the coordinate with respect to which a derivative is
# being taken, and each Field type F implements how to actually take the
# derivative.
# 
# ∇diag(coord, covariance, prefactor) is a Field which represents the diagonal
# entries of the Fourier representation of the gradient operation, with `coord`
# specifiying the coordinate for the gradient direction (1 or 2), `covariance`
# being COVARIANT or CONTRAVARIANT, and multiplied by a `prefactor` (generally
# 1 or -1 to keep track of if an adjoint has been taken). ∇[i] returns one of
# these wrapped in a Diagonal so that multipliying by these does the necessary
# basis conversion. ∇ (along with ∇ⁱ, and ∇ᵢ) are StaticVectors which can also
# be used with FieldVector algebra, e.g. ∇*f. 
# 
# Also note: We define the components of vectors, including ∇, to be with respect to
# the _unnormalized_ covariant or contravariant basis vectors, hence ∇ⁱ = d/dxᵢ
# and ∇ᵢ = d/dxⁱ. This is different than with respect to the _normalized)
# covariant basis vectors, which, e.g., in spherical coordinates, gives the more
# familiar ∇ = (d/dθ, 1/sinθ d/dϕ), (but whose components are neither covariant
# nor contravariant). 

Base.@enum Covariance COVARIANT CONTRAVARIANT

struct ∇diag <: ImplicitField{DerivBasis,Complex{Bottom}}
    coord :: Int
    covariance :: Covariance
    prefactor :: Int
end

struct ∇²diag <: Field{DerivBasis,Complex{Bottom}} end

get_g_metadata(::∇diag) = nothing



# adjoint(D::Diagonal{<:∇diag}) calls conj(D.diag) and here we keep
# track of that a conjugate was taken
conj(d::∇diag) = ∇diag(d.coord, d.covariance, -d.prefactor)
-(d::DiagOp{∇diag}) = Diagonal(conj(d))

# Gradient vector which can be used with FieldVector algebra. 
# struct ∇Op <: StaticVector{2,Diagonal{Float32,∇diag}} end
# getindex(::∇Op, i::Int) = Diagonal(∇diag{i,covariance,prefactor}())
# -(::∇Op{covariance,prefactor}) where {covariance,prefactor} = ∇Op{covariance,-prefactor}()
const ∇ⁱ = @SVector[Diagonal(∇diag(coord, CONTRAVARIANT, 1)) for coord=1:2]
const ∇ᵢ = @SVector[Diagonal(∇diag(coord, COVARIANT,     1)) for coord=1:2]
const ∇ = ∇ⁱ # ∇ is contravariant by default if not specified
const ∇² = Diagonal(∇²diag())


@doc doc"""
    gradhess(f)
    
Compute the gradient $g^i = \nabla^i f$, and the hessian, $H_j^{\,i} = \nabla_j \nabla^i f$.
"""
function gradhess(f)
    g = ∇ⁱ*f
    H = @SMatrix[∇ᵢ[1]*g[1] ∇ᵢ[2]*g[1]; ∇ᵢ[1]*g[2] ∇ᵢ[2]*g[2]]
    (;g, H)
end


# ### FuncOp

# An Op which applies some arbitrary function to its argument.
# Transpose and/or inverse operations which are not specified will return an error.
@kwdef struct FuncOp <: ImplicitOp{Bottom}
    op   = nothing
    opᴴ  = nothing
    op⁻¹ = nothing
    op⁻ᴴ = nothing
end
SymmetricFuncOp(;op=nothing, op⁻¹=nothing) = FuncOp(op,op,op⁻¹,op⁻¹)
FuncOp(op::Function) = FuncOp(op=op)
SymmetricFuncOp(op::Function) = SymmetricFuncOp(op=op)
*(L::FuncOp, f::Field) = L.op   != nothing ? L.op(f)   : error("op*f not implemented")
\(L::FuncOp, f::Field) = L.op⁻¹ != nothing ? L.op⁻¹(f) : error("op\\f not implemented")
*(f::Adjoint{<:Any,Field}, L::FuncOp) = L.opᴴ  != nothing ? L.opᴴ(f)  : error("opᴴ*f not implemented")
adjoint(L::FuncOp) = FuncOp(L.opᴴ,L.op,L.op⁻ᴴ,L.op⁻¹)
inv(L::FuncOp) = FuncOp(L.op⁻¹,L.op⁻ᴴ,L.op,L.opᴴ)
adapt_structure(to, L::FuncOp) = FuncOp(adapt(to, fieldvalues(L))...)
hash(L::FuncOp, h::UInt64) = foldr(hash, (typeof(L), fieldvalues(L)...), init=h)


### BandPassOp

# An op which applies some bandpass, like a high or low-pass filter. This object
# stores the bandpass weights, Wℓ, and each Field type F should implement
# preprocess((b,metadata), ::BandPassOp) to describe how this is actually
# applied. 

struct BandPass{W<:InterpolatedCℓs} <: ImplicitField{HarmonicBasis,Bottom}
    Wℓ::W
end
BandPassOp(ℓ,Wℓ) = Diagonal(BandPass(InterpolatedCℓs(promote(collect(ℓ),collect(Wℓ))...)))
BandPassOp(Wℓ::InterpolatedCℓs) = Diagonal(BandPass(Wℓ))
cos_ramp_up(length) = @. (cos($range(π,0,length=length))+1)/2
cos_ramp_down(length) = 1 .- cos_ramp_up(length)
HighPass(ℓ; Δℓ=50) = BandPassOp(ℓ:20000, [cos_ramp_up(Δℓ); ones(20000-ℓ-Δℓ+1)])
LowPass(ℓ; Δℓ=50) = BandPassOp(0:ℓ, [ones(ℓ-Δℓ+1); cos_ramp_down(Δℓ)])
MidPass(ℓmin, ℓmax; Δℓ=50) = BandPassOp(ℓmin:ℓmax, [cos_ramp_up(Δℓ); ones(ℓmax-ℓmin-2Δℓ+1); cos_ramp_down(Δℓ)])
MidPasses(ℓedges; Δℓ=10) = [MidPass(ℓmin-Δℓ÷2,ℓmax+Δℓ÷2; Δℓ=Δℓ) for (ℓmin,ℓmax) in zip(ℓedges[1:end-1],ℓedges[2:end])]
hash(b::BandPass, h::UInt64) = foldr(hash, (typeof(b), b.Wℓ), init=h)


### ParamDependentOp

# A FieldOp which depends on some parameters, θ. 
# L(;θ...) recomputes the operator at a given set of parameters, but the
# operator can also be used as-is in which case it is evaluated at a fiducial θ
# (which is stored inside the operator when it is first constructed). 


@doc doc"""
    ParamDependentOp(recompute_function::Function)
    
Creates an operator which depends on some parameters $\theta$ and can be
evaluated at various values of these parameters. 

`recompute_function` should be a function which accepts keyword arguments for
$\theta$ and returns the operator. Each keyword must have a default value; the
operator will act as if evaluated at these defaults unless it is explicitly
evaluated at other parameters. 

Example:

```julia
Cϕ₀ = Diagonal(...) # some fixed Diagonal operator
Cϕ = ParamDependentOp((;Aϕ=1)->Aϕ*Cϕ₀) # create ParamDependentOp

Cϕ(Aϕ=1.1) * ϕ   # Cϕ(Aϕ=1.1) is equal to 1.1*Cϕ₀
Cϕ * ϕ           # Cϕ alone will act like Cϕ(Aϕ=1) because that was the default above
```

Note: if you are doing parallel work, global variables referred to in the
`recompute_function` need to be distributed to all workers. A more robust
solution is to avoid globals entirely and instead ensure all variables are
"closed" over (and hence will automatically get distributed). This will happen
by default if defining the `ParamDependentOp` inside any function, or can be
forced at the global scope by wrapping everything in a `let`-block, e.g.:

```julia
Cϕ = let Cϕ₀=Cϕ₀
    ParamDependentOp((;Aϕ=1)->Aϕ*Cϕ₀)
end
```

After executing the code above, `Cϕ` is now ready to be (auto-)shipped to any workers
and will work regardless of what global variables are defined on these workers. 
"""
struct ParamDependentOp{T, L<:FieldOp{T}, F<:Function} <: ImplicitOp{T}
    op :: L
    recompute_function :: F
    parameters :: Vector{Symbol}
end
function ParamDependentOp(recompute_function::Function)
    kwarg_names = filter(!=(Symbol("_...")), get_kwarg_names(recompute_function))
    if endswith(string(kwarg_names[end]), "...")
        # an "kwargs..."-like keyword argument (except for "_..."
        # which is filtered out above) indicates this operator depends
        # on everything, which is indicated by an empty kwarg_names
        empty!(kwarg_names)
    end
    # invokelatest here allows creating a ParamDependent op which calls a
    # BinRescaledOp (eg this is the case for the mixing matrix G which depends
    # on Cϕ) from inside function. this would otherwise fail due to
    # BinRescaledOp eval'ed function being too new
    ParamDependentOp(Base.invokelatest(recompute_function), recompute_function, kwarg_names)
end
function (L::ParamDependentOp)(θ::NamedTuple)
    if depends_on(L,θ)
        # if L got adapt'ed to CuArray since this op was created,
        # L.op will be GPU-backed, but depending on how
        # recompute_function is written, recompute_function may
        # still return something CPU-backed. in that case, copy it
        # to GPU here
        storage = get_storage(L.op)
        Lθ = L.recompute_function(;θ...)
        basetype(storage) == basetype(get_storage(Lθ)) ? Lθ : adapt(storage, Lθ)
    else
        L.op
    end
end
(L::ParamDependentOp)(;θ...) = L((;θ...))
@init @require ComponentArrays="b0b7db55-cfe3-40fc-9ded-d10e2dbeff66" begin
    using .ComponentArrays
    (L::ParamDependentOp)(θ::ComponentArray) = L(convert(NamedTuple, θ))
    (L::Union{FieldOp,UniformScaling})(::ComponentArray) = L
end

@auto_adjoint *(L::ParamDependentOp, f::Field) = L.op * f
@auto_adjoint \(L::ParamDependentOp, f::Field) = L.op \ f
for F in (:inv, :pinv, :sqrt, :adjoint, :Diagonal, :diag, :simulate, :zero, :one, :logdet)
    @eval $F(L::ParamDependentOp) = $F(L.op)
end
getindex(L::ParamDependentOp, x) = getindex(L.op, x)
simulate(rng::AbstractRNG, L::ParamDependentOp) = simulate(rng, L.op)
depends_on(L::ParamDependentOp, θ) = depends_on(L, keys(θ))
depends_on(L::ParamDependentOp, θ::Tuple) = isempty(L.parameters) || any(L.parameters .∈ Ref(θ))
depends_on(L,                   θ) = false

typealias_def(::Type{<:ParamDependentOp{T,L}}) where {T,L} = "ParamDependentOp{$(typealias(L))}"
function Base.summary(io::IO, L::ParamDependentOp)
    dependent_params = isempty(L.parameters) ? "..." : join(map(string, L.parameters), ",")
    print(io, join(size(L.op), "×"), " (", dependent_params, ")-dependent ")
    Base.showarg(io, L, true)
end

# we have to include recompute_function in the hash, but its hash
# might change if shipped to a distributed worker, despite being the
# same function. not sure if there's any way around this...
hash(L::ParamDependentOp, h::UInt64) = foldr(hash, (typeof(L), L.op, L.recompute_function), init=h)

adapt_structure(to, L::ParamDependentOp) = 
    ParamDependentOp(adapt(to, L.op), adapt(to, L.recompute_function), L.parameters)


### LazyBinaryOp

# we use LazyBinaryOps to create new operators composed from other
# operators which don't actually evaluate anything until they've been
# multiplied by a field. 
# L::LazyBinaryOp{λ} lazily represents λ(L.X,L.Y)
struct LazyBinaryOp{λ} <: ImplicitOp{Bottom}
    X
    Y
    LazyBinaryOp(λ, X, Y) = new{λ}(X, Y)
end

# creating LazyBinaryOps
for λ in (:+, :-, :*)
    @eval function ($λ)(
        X :: Union{ImplicitOp, Adjoint{<:Any,<:ImplicitOp}, DiagOp{<:Field{B₁}}},
        Y :: Union{ImplicitOp, Adjoint{<:Any,<:ImplicitOp}, DiagOp{<:Field{B₂}}}
    ) where {B₁,B₂}
        LazyBinaryOp($λ, X, Y)
    end
    @eval function ($λ)(
        X :: DiagOp{<:Field{B}},
        Y :: DiagOp{<:Field{B}}
    ) where {B}
        Diagonal(broadcast($λ, diag(X), diag(Y)))
    end
end
for λ in (:+, :-) 
    @eval ($λ)(X::UniformScaling, Y::ImplicitOp) = LazyBinaryOp($λ, X, Y)
    @eval ($λ)(X::ImplicitOp, Y::UniformScaling) = LazyBinaryOp($λ, X, Y)
end
(*)(X::ImplicitOp,              Y::Scalar)     = LazyBinaryOp(*, X, Y)
(*)(X::Scalar,                  Y::ImplicitOp) = LazyBinaryOp(*, X, Y)
(/)(X::ImplicitOp,              Y::Real)       = LazyBinaryOp(/, X, Y)
(^)(X::ImplicitOp,              Y::Integer)    = LazyBinaryOp(^, X, Y)
(^)(X::DiagOp{<:ImplicitField}, Y::Integer)    = LazyBinaryOp(^, X, Y)
(-)(L::ImplicitOp)                             = LazyBinaryOp(*, -1, L)
pinv(L::LazyBinaryOp{*})                       = LazyBinaryOp(*, pinv(L.Y), pinv(L.X))
adjoint(L::LazyBinaryOp{*})                    = LazyBinaryOp(*, adjoint(L.Y), adjoint(L.X))

# evaluating LazyBinaryOps
for λ in (:+, :-)
    @eval (*)(L::LazyBinaryOp{$λ}, f::Field) = ($λ)(L.X * f, L.Y * f)
    @eval diag(L::LazyBinaryOp{$λ}) = ($λ)(diag(L.X), diag(L.Y))
    @eval adjoint(L::LazyBinaryOp{$λ}) = LazyBinaryOp(($λ), adjoint(L.X), adjoint(L.Y))
end
@auto_adjoint (*)(L::LazyBinaryOp{/}, f::Field) = (L.X * f) / L.Y
@auto_adjoint (*)(L::LazyBinaryOp{*}, f::Field) = L.X * (L.Y * f)
@auto_adjoint (\)(L::LazyBinaryOp{*}, f::Field) = L.Y \ (L.X \ f)
@auto_adjoint (*)(L::LazyBinaryOp{^}, f::Field) = foldr((L.Y>0 ? (*) : (\)), fill(L.X, abs(L.Y::Integer)), init=f)

# misc
getindex(L::LazyBinaryOp, i::Int) = i==1 ? L.X : i==2 ? L.Y : error("Can only index LazyBinaryOp by 1 or 2")
adapt_structure(to, L::LazyBinaryOp{λ}) where {λ} = LazyBinaryOp(λ, adapt(to,L.X), adapt(to,L.Y))
hash(L::LazyBinaryOp, h::UInt64) = foldr(hash, (typeof(L), L.X, L.Y), init=h)
