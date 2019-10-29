### Diagonal ops

# we use Base.Diagonal(f) for diagonal operators so very little specific code is
# actually needed here. 

simulate(D::DiagOp{F}) where {F<:Field} = sqrt(D) * white_noise(F)

# automatic basis conversion (and NaN-zeroing)
(*)(D::DiagOp{<:Field{B}}, f::Field) where {B} = D.diag .* B(f)
(\)(D::DiagOp{<:Field{B}}, f::Field) where {B} = nan2zero.(D.diag .\ B(f))

# broadcasting
BroadcastStyle(::StructuredMatrixStyle{<:DiagOp{F}}, ::StructuredMatrixStyle{<:DiagOp{<:ImplicitField}}) where {F<:Field} = StructuredMatrixStyle{DiagOp{F}}()
BroadcastStyle(::StructuredMatrixStyle{<:DiagOp{<:ImplicitField}}, ::StructuredMatrixStyle{<:DiagOp{F}}) where {F<:Field} = Base.Broadcast.Unknown()
function similar(bc::Broadcasted{<:StructuredMatrixStyle{<:DiagOp{F}}}, ::Type{T}) where {F<:Field,T}
    Diagonal(similar(typeof(BroadcastStyle(F)),T))
end
diag_data(D::Diagonal) = D.diag
diag_data(x) = x
function copyto!(dest::DiagOp, bc::Broadcasted{<:StructuredMatrixStyle})
    bc′ = flatten(bc)
    copyto!(dest.diag, Broadcasted{Nothing}(bc′.f, map(diag_data, bc′.args)))
    dest
end

# the generic versions of these kind of suck so we need these specializized
# versions:
(*)(x::Adjoint{<:Any,<:Field}, D::Diagonal) = (D*parent(x))'
(*)(x::Adjoint{<:Any,<:Field}, D::Diagonal, y::Field) = x*(D*y)
diag(L::DiagOp) = L.diag

# use getproperty here to ensure no basis conversion is done
getindex(D::DiagOp, s::Symbol) = Diagonal(getproperty(D.diag,s))

# for printing
size(::DiagOp{<:ImplicitField}) = ()
axes(::DiagOp{<:ImplicitField}, I) = OneTo(0)

# the generic version of this is prohibitively slow so we need this
hash(D::DiagOp, h::UInt64) = hash(D.diag, h)


### Derivative ops

# These ops just store the coordinate with respect to which a derivative is
# being taken, and each Field type F implements how to actually take the
# derivative.
# 
# ∇diag{coord, covariance, prefactor} is a Field which represents the diagonal
# entries of the Fourier representation of the gradient operation, with `coord`
# specifiying the coordinate for the gradient direction (1 or 2), `covariance`
# being :covariant or :contravariant, and multiplied by a `prefactor` (generally
# 1 or -1 to keep track of if an adjoint has been taken). ∇[i] returns one of
# these wrapped in a Diagonal so that multipliying by these does the necessary
# basis conversion. ∇ (along with ∇ⁱ, and ∇ᵢ) are StaticVectors which can also
# be used with FieldVector algebra, e.g. ∇*f. 
# 
# Note: we are cheating a bit, because while ∇ is diagonal when a derivative is
# taken in Fourier space via FFT's, its tri-diagonal if taking a map-space
# derivative, so conceptually these should not be wrapped in a Diagonal. Its a
# bit ugly and I'd like to fix it, but since the auto-basis conversion mechanism
# is based on Diagonal, keep things this way for now. Individual field types
# just need to intercept the broadcast machinery at copyto! to implement the
# map-space derivative, e.g. see: flat_generic.jl.
# 
# Also note: We define the components of vectors, including ∇, to be with respect to
# the _unnormalized_ covariant or contravariant basis vectors, hence ∇ⁱ = d/dxᵢ
# and ∇ᵢ = d/dxⁱ. This is different than with respect to the _normalized)
# covariant basis vectors, which, e.g., in spherical coordinates, gives the more
# familiar ∇ = (d/dθ, 1/sinθ d/dϕ), (but whose components are neither covariant
# nor contravariant). 

struct ∇diag{coord, covariance, prefactor} <: ImplicitField{DerivBasis,Spin,Pix} end
struct ∇²diag <: ImplicitField{DerivBasis,Spin,Pix} end

# adjoint(D::Diagonal{<:∇diag}) calls conj(D.diag) and here lazily we keep track
# of that a conjugate was taken
conj(::∇diag{coord,covariance,prefactor}) where {coord,covariance,prefactor} = ∇diag{coord,covariance,-prefactor}()
-(::DiagOp{∇diag{coord,covariance,prefactor}}) where {coord,covariance,prefactor} = Diagonal(∇diag{coord,covariance,-prefactor}())

# Gradient vector which can be used with FieldVector algebra. 
struct ∇Op{covariance,prefactor} <: StaticVector{2,Diagonal{Float32,∇diag{<:Any,covariance,prefactor}}} end 
getindex(::∇Op{covariance,prefactor}, i::Int) where {covariance,prefactor} = Diagonal(∇diag{i,covariance,prefactor}())
const ∇ⁱ = ∇Op{:contravariant,1}()
const ∇ᵢ = ∇Op{:covariant,1}()
const ∇ = ∇ⁱ # ∇ is contravariant by default if not specified
const ∇² = Diagonal(∇²diag())


@doc doc"""
    gradhess(f)
    
Compute the gradient $g^i = \nabla^i f$, and the hessian, $H_j^{\,i} = \nabla_j \nabla^i f$.
"""
function gradhess(f)
    g = ∇ⁱ*f
    (g=g, H=SMatrix{2,2}([permutedims(∇ᵢ * g[1]); permutedims(∇ᵢ * g[2])]))
end

# this is not strictly true (∇[1] is generically a gradient w.r.t. the first
# coordinate, e.g. ∂θ), but this is useful shorthand to have for the flat-sky:
const ∂x = ∇[1]
const ∂y = ∇[2]


### FuncOp

# An Op which applies some arbitrary function to its argument.
# Transpose and/or inverse operations which are not specified will return an error.
@kwdef struct FuncOp <: ImplicitOp{Basis,Spin,Pix}
    op   = nothing
    opᴴ  = nothing
    op⁻¹ = nothing
    op⁻ᴴ = nothing
end
SymmetricFuncOp(;op=nothing, op⁻¹=nothing) = FuncOp(op,op,op⁻¹,op⁻¹)
*(op::FuncOp, f::Field) = op.op   != nothing ? op.op(f)   : error("op*f not implemented")
*(f::Field, op::FuncOp) = op.opᴴ  != nothing ? op.opᴴ(f)  : error("f*op not implemented")
\(op::FuncOp, f::Field) = op.op⁻¹ != nothing ? op.op⁻¹(f) : error("op\\f not implemented")
adjoint(op::FuncOp) = FuncOp(op.opᴴ,op.op,op.op⁻ᴴ,op.op⁻¹)
const IdentityOp = FuncOp(@repeated(identity,4)...)
inv(op::FuncOp) = FuncOp(op.op⁻¹,op.op⁻ᴴ,op.op,op.opᴴ)



### BandPassOp

# An op which applies some bandpass, like a high or low-pass filter. This object
# stores the bandpass weights, Wℓ, and each Field type F should implement
# broadcast_data(::Type{F}, ::BandPassOp) to describe how this is actually
# applied. 

abstract type HarmonicBasis <: Basislike end

struct BandPass{W<:InterpolatedCℓs} <: ImplicitField{HarmonicBasis,Spin,Pix}
    Wℓ::W
end
BandPassOp(ℓ,Wℓ) = Diagonal(BandPass(InterpolatedCℓs(promote(collect(ℓ),collect(Wℓ))...)))
BandPassOp(Wℓ::InterpolatedCℓs) = Diagonal(BandPass(Wℓ))
HighPass(ℓ;Δℓ=50) = BandPassOp(0:10000, [zeros(ℓ-Δℓ); @.((cos($range(π,0,length=2Δℓ))+1)/2); ones(10001-ℓ-Δℓ)])
LowPass(ℓ;Δℓ=50) = BandPassOp(0:(ℓ+Δℓ-1), [ones(ℓ-Δℓ); @.(cos($range(0,π,length=2Δℓ))+1)/2])
MidPass(ℓmin,ℓmax;Δℓ=50)  = BandPassOp(0:(ℓmax+Δℓ-1), [zeros(ℓmin-Δℓ);  @.(cos($range(π,0,length=2Δℓ))+1)/2; ones(ℓmax-ℓmin-2Δℓ); @.((cos($range(0,π,length=2Δℓ))+1)/2)])


### ParamDependentOp

# A LinOp which depends on some parameters, θ. 
# L(;θ...) recomputes the operator at a given set of parameters, but the
# operator can also be used as-is in which case it is evaluated at a fiducial θ
# (which is stored inside the operator when it is first constructed). 


@doc doc"""
    ParamDependentOp(recompute_function::Function)
    ParamDependentOp(recompute_function!::Function, mem)
    
Creates an operator which depends on some parameters $\theta$ and can be
evaluated at various values of these parameters. 

There are two forms to construct this operator. In the first form,
`recompute_function` should be a function which accepts keyword arguments for
$\theta$ and returns the operator. Each keyword must have a default value; the
operator will act as if evaluated at these defaults unless it is explicitly
evaluated at other parameters. In the second form, we can preallocate some
memory for the results `mem`, in which case `recompute_function!` should
additionally accept a single positional argument holding this memory, which
should then be assigned in-place. 

Example:

```julia
Cϕ₀ = Diagonal(...) # some fixed Diagonal operator
Cϕ = ParamDependentOp((;Aϕ=1)->Aϕ*Cϕ₀) # create ParamDependentOp

Cϕ(Aϕ=1.1) * ϕ   # Cϕ(Aϕ=1.1) is equal to 1.1*Cϕ₀
Cϕ * ϕ           # Cϕ alone will act like Cϕ(Aϕ=1) because that was the default above

# a version which preallocates the memory:
Cϕ = ParamDependentOp((mem;Aϕ=1)->(@. mem = Aϕ*Cϕ₀), similar(Cϕ₀))
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

After executing the code above, `Cϕ` is now ready to be shipped to any workers
and will work regardless of what global variables are defined on these workers. 
"""
struct ParamDependentOp{B, S, P, L<:LinOp{B,S,P}, F<:Function} <: ImplicitOp{B,S,P}
    op::L
    recompute_function::F
    parameters::Vector{Symbol}
    inplace::Bool
end
function ParamDependentOp(recompute_function::Function)
    ParamDependentOp(recompute_function(), recompute_function, get_kwarg_names(recompute_function), false)
end
function ParamDependentOp(recompute_function!::Function, mem)
    op = recompute_function!(similar(mem))
    ParamDependentOp(op, (mem=mem;θ...)->(recompute_function!(mem;θ...);mem), get_kwarg_names(recompute_function), true)
end
function (L::ParamDependentOp)(mem=nothing;θ...) 
    if (mem!=nothing)
        L.inplace || throw(ArgumentError("Can't pass preallocated memory to non-inplace ParamDependentOp."))
        mem isa typeof(L.op) || throw(ArgumentError("Preallocated memory passed to ParamDependentOp should be $(typeof(L.op)), not $(typeof(mem))."))
    end
    if depends_on(L,θ)
        dependent_θ = filter(((k,_),)->k in L.parameters, pairs(θ))
        # type annotation here for if any Core.Box'ed variables slipped into our recompute_function:
        L.recompute_function((mem==nothing ? () : (mem,))...; dependent_θ...) :: typeof(L.op)
    else
        L.op
    end 
end
(L::ParamDependentOp)(θ::NamedTuple) = L(;θ...)
*(L::ParamDependentOp, f::Field) = L.op * f
\(L::ParamDependentOp, f::Field) = L.op \ f
for F in (:inv, :pinv, :sqrt, :adjoint, :Diagonal, :simulate, :zero, :one, :logdet)
    @eval $F(L::ParamDependentOp) = $F(L.op)
end
depends_on(L::ParamDependentOp, θ) = depends_on(L, keys(θ))
depends_on(L::ParamDependentOp, θ::Tuple) = any(L.parameters .∈ Ref(θ))
depends_on(L,                   θ) = false


### LazyBinaryOp

# we use LazyBinaryOps to create new operators composed from other operators
# which don't actually evaluate anything until they've been multiplied by a
# field
struct LazyBinaryOp{F,A<:Union{LinOrAdjOp,Scalar},B<:Union{LinOrAdjOp,Scalar}} <: ImplicitOp{Basis,Spin,Pix}
    a::A
    b::B
    LazyBinaryOp(op,a::A,b::B) where {A,B} = new{op,A,B}(a,b)
end
# creating LazyBinaryOps
for op in (:+, :-, :*)
    @eval ($op)(a::ImplicitOrAdjOp,          b::ImplicitOrAdjOp)          = LazyBinaryOp($op,a,b)
    @eval ($op)(a::Union{LinOrAdjOp,Scalar}, b::ImplicitOrAdjOp)          = LazyBinaryOp($op,a,b)
    @eval ($op)(a::ImplicitOrAdjOp,          b::Union{LinOrAdjOp,Scalar}) = LazyBinaryOp($op,a,b)
    # explicit vs. lazy binary operations on Diagonals:
    @eval ($op)(D1::DiagOp{<:Field{B}},  D2::DiagOp{<:Field{B}})  where {B}     = Diagonal(broadcast($op,D1.diag,D2.diag))
    @eval ($op)(D1::DiagOp{<:Field{B1}}, D2::DiagOp{<:Field{B2}}) where {B1,B2} = LazyBinaryOp($op,D1,D2)
end
/(op::ImplicitOrAdjOp, n::Real) = LazyBinaryOp(/,op,n)
literal_pow(::typeof(^), op::ImplicitOrAdjOp, ::Val{-1}) = inv(op)
literal_pow(::typeof(^), op::ImplicitOrAdjOp, ::Val{n}) where {n} = LazyBinaryOp(^,op,n)
^(op::ImplicitOrAdjOp, n::Int) = LazyBinaryOp(^,op,n)
inv(op::ImplicitOrAdjOp) = LazyBinaryOp(^,op,-1)
-(op::ImplicitOrAdjOp) = -1 * op
# evaluating LazyBinaryOps
for op in (:+, :-)
    @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
end
*(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
*(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)
\(lz::LazyBinaryOp{*}, f::Field) = lz.b \ (lz.a \ f)
*(lz::LazyBinaryOp{^}, f::Field) = foldr((lz.b>0 ? (*) : (\)), fill(lz.a,abs(lz.b)), init=f)
adjoint(lz::LazyBinaryOp{F}) where {F} = LazyBinaryOp(F,adjoint(lz.b),adjoint(lz.a))
ud_grade(lz::LazyBinaryOp{op}, args...; kwargs...) where {op} = LazyBinaryOp(op,ud_grade(lz.a,args...;kwargs...),ud_grade(lz.b,args...;kwargs...))


### OuterProdOp

# an operator L which represents L = M*M'
# this could also be represented by a LazyBinaryOp, but this allows us to 
# do simulate(L) = M * whitenoise
struct OuterProdOp{TM<:LinOp} <: ImplicitOp{Basis,Spin,Pix}
    M::TM
end
simulate(L::OuterProdOp{<:DiagOp{F}}) where {F} = L.M * white_noise(F)
simulate(L::OuterProdOp{<:LazyBinaryOp{*}}) = L.M.a * sqrt(L.M.b) * simulate(L.M.b)
pinv(L::OuterProdOp{<:LazyBinaryOp{*}}) = OuterProdOp(pinv(L.M.a)' * pinv(L.M.b)')
*(L::OuterProdOp, f::Field) = L.M * (L.M' * f)
\(L::OuterProdOp, f::Field) = L.M' \ (L.M \ f)
adjoint(L::OuterProdOp) = L
