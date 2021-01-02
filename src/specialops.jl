# ### Diagonal ops

# we use Base.Diagonal(f) for diagonal operators so very little specific code is
# actually needed here. 

simulate(rng::AbstractRNG, D::DiagOp{F}) where {F<:Field} = sqrt(D) * white_noise(rng, F)
global_rng_for(::Type{<:DiagOp{F}}) where {F} = global_rng_for(F)

# automatic basis conversion (and NaN-zeroing)
(*)(D::DiagOp{<:Field{B}}, f::Field) where {B} = diag(D) .* B(f)
(\)(D::DiagOp{<:Field{B}}, f::Field) where {B} = nan2zero.(diag(D) .\ B(f))

# # broadcasting
# struct DiagOpStyle{FS} <: AbstractArrayStyle{2} end
# BroadcastStyle(::Type{D}) where {F<:Field,D<:DiagOp{F}} = DiagOpStyle{typeof(BroadcastStyle(F))}()
# BroadcastStyle(::DiagOpStyle{FS1}, ::DiagOpStyle{FS2}) where {FS1,FS2} = DiagOpStyle{typeof(result_style(FS1(),FS2()))}()
# BroadcastStyle(S::DiagOpStyle, ::DefaultArrayStyle{0}) = S
# similar(bc::Broadcasted{DiagOpStyle{FS}}, ::Type{T}) where {FS,T} = Diagonal(similar(FS,T))
# instantiate(bc::Broadcasted{<:DiagOpStyle}) = bc
# diag_data(D::Diagonal) = D.diag
# diag_data(x) = x
# function copyto!(dest::DiagOp, bc::Broadcasted{Nothing})
#     copyto!(dest.diag, map_bc_args(diag_data, bc))
#     dest
# end

# the generic versions of these kind of suck so we need these specializized
# versions:
(*)(x::Adjoint{<:Any,<:Field}, D::Diagonal) = (D*parent(x))'
(*)(x::Adjoint{<:Any,<:Field}, D::Diagonal, y::Field) = x*(D*y)
diag(L::DiagOp) = L.diag

# use getproperty here to ensure no basis conversion is done
getindex(D::DiagOp, s::Symbol) = Diagonal(getproperty(diag(D),s))

# # for printing
# size(::DiagOp{<:Field}) = ()
# axes(::DiagOp{<:Field}, I) = OneTo(0)

# the generic version of this is prohibitively slow so we need this
hash(D::DiagOp, h::UInt64) = hash(D.diag, h)

get_storage(L::DiagOp) = get_storage(diag(L))


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


Base.@enum Covariance COVARIANT CONTRAVARIANT

struct ∇diag <: Field{DerivBasis,Bottom}
    coord :: Int
    covariance :: Covariance
    prefactor :: Int
end
size(::∇diag) = ()

struct ∇²diag <: Field{DerivBasis,Bottom} end

get_g_metadata(::∇diag) = nothing

Base.require_one_based_indexing(::∇diag) = nothing # allows Diagonal(::∇ᵢdiag)



# # adjoint(D::Diagonal{<:∇diag}) calls conj(D.diag) and here lazily we keep track
# # of that a conjugate was taken
# conj(::∇diag{coord,covariance,prefactor}) where {coord,covariance,prefactor} = ∇diag{coord,covariance,-prefactor}()
# -(::DiagOp{∇diag{coord,covariance,prefactor}}) where {coord,covariance,prefactor} = Diagonal(∇diag{coord,covariance,-prefactor}())

# Gradient vector which can be used with FieldVector algebra. 
# struct ∇Op <: StaticVector{2,Diagonal{Float32,∇diag}} end
# getindex(::∇Op, i::Int) = Diagonal(∇diag{i,covariance,prefactor}())
# -(::∇Op{covariance,prefactor}) where {covariance,prefactor} = ∇Op{covariance,-prefactor}()
const ∇ⁱ = @SVector[Diagonal(∇diag(coord, CONTRAVARIANT, 1)) for coord=1:2]
const ∇ᵢ = @SVector[Diagonal(∇diag(coord, COVARIANT,     1)) for coord=1:2]
const ∇ = ∇ⁱ # ∇ is contravariant by default if not specified
# const ∇² = Diagonal(∇²diag())


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

# # An Op which applies some arbitrary function to its argument.
# # Transpose and/or inverse operations which are not specified will return an error.
# @kwdef struct FuncOp <: ImplicitOp{Basis,Spin,Pix}
#     op   = nothing
#     opᴴ  = nothing
#     op⁻¹ = nothing
#     op⁻ᴴ = nothing
# end
# SymmetricFuncOp(;op=nothing, op⁻¹=nothing) = FuncOp(op,op,op⁻¹,op⁻¹)
# FuncOp(op::Function) = FuncOp(op=op)
# SymmetricFuncOp(op::Function) = SymmetricFuncOp(op=op)
# *(L::FuncOp, f::Field) = 
#     L.op   != nothing ? L.op(f)   : error("op*f not implemented")
# \(L::FuncOp, f::Field) = 
#     L.op⁻¹ != nothing ? L.op⁻¹(f) : error("op\\f not implemented")
# *(f::Adjoint{<:Any,Field}, L::FuncOp) = 
#     L.opᴴ  != nothing ? L.opᴴ(f)  : error("opᴴ*f not implemented")
# adjoint(L::FuncOp) = FuncOp(L.opᴴ,L.op,L.op⁻ᴴ,L.op⁻¹)
# inv(L::FuncOp) = FuncOp(L.op⁻¹,L.op⁻ᴴ,L.op,L.opᴴ)
# adapt_structure(to, L::FuncOp) = FuncOp(adapt(to, fieldvalues(L))...)


# ### BandPassOp

# # An op which applies some bandpass, like a high or low-pass filter. This object
# # stores the bandpass weights, Wℓ, and each Field type F should implement
# # broadcast_data(::Type{F}, ::BandPassOp) to describe how this is actually
# # applied. 

# abstract type HarmonicBasis <: Basislike end

# struct BandPass{W<:InterpolatedCℓs} <: ImplicitField{HarmonicBasis,Spin,Pix}
#     Wℓ::W
# end
# BandPassOp(ℓ,Wℓ) = Diagonal(BandPass(InterpolatedCℓs(promote(collect(ℓ),collect(Wℓ))...)))
# BandPassOp(Wℓ::InterpolatedCℓs) = Diagonal(BandPass(Wℓ))
# cos_ramp_up(length) = @. (cos($range(π,0,length=length))+1)/2
# cos_ramp_down(length) = 1 .- cos_ramp_up(length)
# HighPass(ℓ; Δℓ=50) = BandPassOp(ℓ:20000, [cos_ramp_up(Δℓ); ones(20000-ℓ-Δℓ+1)])
# LowPass(ℓ; Δℓ=50) = BandPassOp(0:ℓ, [ones(ℓ-Δℓ+1); cos_ramp_down(Δℓ)])
# MidPass(ℓmin, ℓmax; Δℓ=50) = BandPassOp(ℓmin:ℓmax, [cos_ramp_up(Δℓ); ones(ℓmax-ℓmin-2Δℓ+1); cos_ramp_down(Δℓ)])
# MidPasses(ℓedges; Δℓ=10) = [MidPass(ℓmin-Δℓ÷2,ℓmax+Δℓ÷2; Δℓ=Δℓ) for (ℓmin,ℓmax) in zip(ℓedges[1:end-1],ℓedges[2:end])]


# ### ParamDependentOp

# # A LinOp which depends on some parameters, θ. 
# # L(;θ...) recomputes the operator at a given set of parameters, but the
# # operator can also be used as-is in which case it is evaluated at a fiducial θ
# # (which is stored inside the operator when it is first constructed). 


# @doc doc"""
#     ParamDependentOp(recompute_function::Function)
    
# Creates an operator which depends on some parameters $\theta$ and can be
# evaluated at various values of these parameters. 

# `recompute_function` should be a function which accepts keyword arguments for
# $\theta$ and returns the operator. Each keyword must have a default value; the
# operator will act as if evaluated at these defaults unless it is explicitly
# evaluated at other parameters. 

# Example:

# ```julia
# Cϕ₀ = Diagonal(...) # some fixed Diagonal operator
# Cϕ = ParamDependentOp((;Aϕ=1)->Aϕ*Cϕ₀) # create ParamDependentOp

# Cϕ(Aϕ=1.1) * ϕ   # Cϕ(Aϕ=1.1) is equal to 1.1*Cϕ₀
# Cϕ * ϕ           # Cϕ alone will act like Cϕ(Aϕ=1) because that was the default above
# ```

# Note: if you are doing parallel work, global variables referred to in the
# `recompute_function` need to be distributed to all workers. A more robust
# solution is to avoid globals entirely and instead ensure all variables are
# "closed" over (and hence will automatically get distributed). This will happen
# by default if defining the `ParamDependentOp` inside any function, or can be
# forced at the global scope by wrapping everything in a `let`-block, e.g.:

# ```julia
# Cϕ = let Cϕ₀=Cϕ₀
#     ParamDependentOp((;Aϕ=1)->Aϕ*Cϕ₀)
# end
# ```

# After executing the code above, `Cϕ` is now ready to be (auto-)shipped to any workers
# and will work regardless of what global variables are defined on these workers. 
# """
# struct ParamDependentOp{B, S, P, L<:LinOp{B,S,P}, F<:Function} <: ImplicitOp{B,S,P}
#     op::L
#     recompute_function::F
#     parameters::Vector{Symbol}
# end
# function ParamDependentOp(recompute_function::Function)
#     # invokelatest here allows creating a ParamDependent op which calls a
#     # BinRescaledOp (eg this is the case for the mixing matrix G which depends
#     # on Cϕ) from inside function. this would otherwise fail due to
#     # BinRescaledOp eval'ed function being too new
#     kwarg_names = get_kwarg_names(recompute_function)
#     if endswith(string(kwarg_names[end]), "...") && !startswith(string(kwarg_names[end]),"_")
#         kwarg_decl = empty!(kwarg_names) # to indicate it depends on anything
#     end
#     ParamDependentOp(Base.invokelatest(recompute_function), recompute_function, kwarg_names)
# end
# function (L::ParamDependentOp)(θ::NamedTuple)
#     if depends_on(L,θ)
#         # filtering out non-dependent parameters disabled until I can find a fix to:
#         # https://discourse.julialang.org/t/can-zygote-do-derivatives-w-r-t-keyword-arguments-which-get-captured-in-kwargs/34553/8
#         # dependent_θ = filter(((k,_),)->k in L.parameters, pairs(θ))
        
#         # if L got adapt'ed to CuArray since this op was created,
#         # L.op will be GPU-backed, but depending on how
#         # recompute_function is written, recompute_function may
#         # still return something CPU-backed. in that case, copy it
#         # to GPU here
#         storage = get_storage(L.op)
#         Lθ = L.recompute_function(;θ...)
#         storage == get_storage(Lθ) ? Lθ : adapt(storage, Lθ)
#     else
#         L.op
#     end 
# end
# (L::ParamDependentOp)(;θ...) = L((;θ...))
# *(L::ParamDependentOp, f::Field) = L.op * f
# \(L::ParamDependentOp, f::Field) = L.op \ f
# for F in (:inv, :pinv, :sqrt, :adjoint, :Diagonal, :diag, :simulate, :zero, :one, :logdet, :global_rng_for)
#     @eval $F(L::ParamDependentOp) = $F(L.op)
# end
# getindex(L::ParamDependentOp, x) = getindex(L.op, x)
# simulate(rng::AbstractRNG, L::ParamDependentOp) = simulate(rng, L.op)
# depends_on(L::ParamDependentOp, θ) = depends_on(L, keys(θ))
# depends_on(L::ParamDependentOp, θ::Tuple) = isempty(L.parameters) || any(L.parameters .∈ Ref(θ))
# depends_on(L,                   θ) = false

# adapt_structure(to, L::ParamDependentOp) = 
#     ParamDependentOp(adapt(to, L.op), adapt(to, L.recompute_function), L.parameters)


# ### LazyBinaryOp

# # we use LazyBinaryOps to create new operators composed from other operators
# # which don't actually evaluate anything until they've been multiplied by a
# # field
# struct LazyBinaryOp{F,A<:Union{LinOrAdjOp,Scalar},B<:Union{LinOrAdjOp,Scalar}} <: ImplicitOp{Basis,Spin,Pix}
#     a::A
#     b::B
#     LazyBinaryOp(op,a::A,b::B) where {A,B} = new{op,A,B}(a,b)
# end
# # creating LazyBinaryOps
# for op in (:+, :-, :*)
#     @eval ($op)(a::ImplicitOrAdjOp,          b::ImplicitOrAdjOp)          = LazyBinaryOp($op,a,b)
#     @eval ($op)(a::Union{LinOrAdjOp,Scalar}, b::ImplicitOrAdjOp)          = LazyBinaryOp($op,a,b)
#     @eval ($op)(a::ImplicitOrAdjOp,          b::Union{LinOrAdjOp,Scalar}) = LazyBinaryOp($op,a,b)
#     # explicit vs. lazy binary operations on Diagonals:
#     @eval ($op)(D1::DiagOp{<:Field{B}},  D2::DiagOp{<:Field{B}})  where {B}     = Diagonal(broadcast($op,D1.diag,D2.diag))
#     @eval ($op)(D1::DiagOp{<:Field{B1}}, D2::DiagOp{<:Field{B2}}) where {B1,B2} = LazyBinaryOp($op,D1,D2)
# end
# /(op::ImplicitOrAdjOp, n::Real) = LazyBinaryOp(/,op,n)
# ^(op::Union{ImplicitOrAdjOp,DiagOp{<:ImplicitField}}, n::Int) = LazyBinaryOp(^,op,n)
# inv(op::Union{ImplicitOrAdjOp,DiagOp{<:ImplicitField}}) = LazyBinaryOp(^,op,-1)
# -(op::ImplicitOrAdjOp) = -1 * op
# pinv(op::LazyBinaryOp{*}) = pinv(op.b) * pinv(op.a)
# # evaluating LazyBinaryOps
# for op in (:+, :-)
#     @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
#     @eval diag(lz::LazyBinaryOp{$op}) = ($op)(diag(lz.a), diag(lz.b))
#     @eval adjoint(lz::LazyBinaryOp{$op}) = LazyBinaryOp(($op),adjoint(lz.a),adjoint(lz.b))
# end
# *(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
# *(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)
# \(lz::LazyBinaryOp{*}, f::Field) = lz.b \ (lz.a \ f)
# *(lz::LazyBinaryOp{^}, f::Field) = foldr((lz.b>0 ? (*) : (\)), fill(lz.a,abs(lz.b)), init=f)
# adjoint(lz::LazyBinaryOp{*}) = LazyBinaryOp(*,adjoint(lz.b),adjoint(lz.a))
# ud_grade(lz::LazyBinaryOp{op}, args...; kwargs...) where {op} = LazyBinaryOp(op,ud_grade(lz.a,args...;kwargs...),ud_grade(lz.b,args...;kwargs...))
# adapt_structure(to, lz::LazyBinaryOp{op}) where {op} = LazyBinaryOp(op, adapt(to,lz.a), adapt(to,lz.b))
# function diag(lz::LazyBinaryOp{*}) 
#     _diag(x) = diag(x)
#     _diag(x::Int) = x
#     da, db = _diag(lz.a), _diag(lz.b)
#     # if basis(da)!=basis(db)
#     #     error("Can't take diag(A*B) where A::$(typeof(lz.a)) and B::$(typeof(lz.b)).")
#     # end
#     da .* db
# end


# ### OuterProdOp

# # an operator L which represents L = V*W'
# # this could also be represented by a LazyBinaryOp, but this allows us to 
# # define a few extra functions like simulate or diag, which get used in various places 
# struct OuterProdOp{TV,TW} <: ImplicitOp{Basis,Spin,Pix}
#     V::TV
#     W::TW
# end
# OuterProdOp(V) = OuterProdOp(V,V)
# _check_sym(L::OuterProdOp) = L.V === L.W ? L : error("Can't do this operation on non-symmetric OuterProdOp.")
# pinv(L::OuterProdOp{<:LazyBinaryOp{*}}) = (_check_sym(L); OuterProdOp(pinv(L.V.a)' * pinv(L.V.b)'))
# *(L::OuterProdOp, f::Field) = L.V * (L.W' * f)
# \(L::OuterProdOp{<:LinOp,<:LinOp}, f::Field) = L.W' \ (L.V \ f)
# adjoint(L::OuterProdOp) = OuterProdOp(L.W,L.V)
# adapt_structure(to, L::OuterProdOp) = OuterProdOp((V′=adapt(to,L.V);), (L.V === L.W ? V′ : adapt(to,L.W)))
# diag(L::OuterProdOp{<:Field{B},<:Field}) where {B} = L.V .* conj.(B(L.W))
# *(D::DiagOp{<:Field{B}}, L::OuterProdOp{<:Field{B},<:Field{B}}) where {B} = OuterProdOp(diag(D) .* L.V, L.W)
# *(L::OuterProdOp{<:Field{B},<:Field{B}}, D::DiagOp{<:Field{B}}) where {B} = OuterProdOp(L.V, L.W .* diag(D))
# tr(L::OuterProdOp{<:Field{B},<:Field{B}}) where {B} = dot(L.V, L.W)




# ### BinRescaledOp

# """
#     BinRescaledOp(C₀, Cbins, θname::Symbol)
    
# Create a [`ParamDependentOp`](@ref) which has a parameter named `θname` which is
# an array that controls the amplitude of bandpowers in bins given by `Cbins`. 

# For example, `BinRescaledOp(C₀, [Cbin1, Cbin2], :A)` creates the operator: 

#     ParamDependentOp( (;A=[1,1], _...) -> C₀ + (A[1]-1) * Cbin1 + (A[2]-1) * Cbin2 )

# where `C₀`, `Cbin1`, and `Cbin2` should be some `LinOp`s. Note `Cbins` are
# directly the power which is added, rather than a mask. 

# The resulting operator is differentiable in `θname`.

# """
# function BinRescaledOp(C₀, Cbins, θname::Symbol)
#     # for some reason, if I eval this into CMBLensing as opposed to Main, it
#     # doesn't get shipped to workers correctly. 
#     # see also: https://discourse.julialang.org/t/closure-not-shipping-to-remote-workers-except-from-main
#     @eval Main begin
#         # ensure Cbins is a tuple and not Array so that `adapt` works recursively through it
#         let C₀=$C₀, T=$(real(eltype(C₀))), Cbins=$(tuple(Cbins...)) 
#             $ParamDependentOp(function (;($θname)=$(ones(Int,length(Cbins))), _...)
#                 $(Expr(:call, :+, :C₀, [:(T($θname[$i] - 1) * Cbins[$i]) for i=1:length(Cbins)]...))
#             end)
#         end
#     end
# end
