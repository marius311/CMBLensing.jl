### FullDiagOp

function showarg(io::IO, D::Diagonal{<:Any,<:Field}, toplevel)
    print(io, "Diagonal{")
    showarg(io, D.diag, toplevel)
    print(io, "}")
end
function getindex(D::Diagonal{<:Any,<:FieldTuple}, i::Int, j::Int)
    i==j ? CatView(map(x->view(x,:), D.diag.fs)...)[i] : 0
end




### Derivative ops

# These ops just store the coordinate with respect to which a derivative is
# being taken, and each Field type F should implement broadcast_data(::Type{F},
# ::∂) to describe how this is actually applied. 

# Note: We define the components of vectors, including ∇, to be with respect to
# the unnormalized covariant or contravariant basis vectors, hence ∇ⁱ = d/dxᵢ
# and ∇ᵢ = d/dxⁱ. This is different than with respect to the normalized
# covariant basis vectors, which, e.g., in spherical coordinates, gives the more
# familiar ∇ = (d/dθ, 1/sinθ d/dϕ), (but whose components are neither covariant
# nor contravariant). 

abstract type DerivBasis <: Basislike end
const Ð = DerivBasis
Ð!(args...) = Ð(args...)

# bit of a hack: ∇i is marked as Float32 here so that it can be widened to
# whatever its broadcast with by combine_eltypes in Broadcast.copy
struct ∇i{coord, covariant} <: Field{DerivBasis,Spin,Pix,Float32} end

# ∇i doesn't have a set size, it will take the size of the fields its broadcasted with:
Base.axes(::∇i) = ()

# gives the gradient gⁱ = ∇ⁱf, and the hessian, Hⁱⱼ = ∇ⱼ∇ⁱf
const ∇⁰, ∇¹, ∇₀, ∇₁ = Diagonal(∇i{0,false}()), Diagonal(∇i{1,false}()), Diagonal(∇i{0,true}()), Diagonal(∇i{1,true}())

# printing
show(io::IO, ::MIME"text/plain", ∇i::Diagonal{<:Any,<:∇i}) = show(io, ∇i)
function show(io::IO, ::Diagonal{<:Any,∇i{coord, covariant}}) where {coord, covariant}
    print(io, "LinearAlgebra.Diagonal{∇", covariant ? (coord==0 ? "₀" : "₁") : (coord==0 ? "⁰" : "¹"),"}()")
end


function gradhess(f)
    g = ∇ⁱ*f
    g, SMatrix{2,2}([permutedims(∇ᵢ * g[1]); permutedims(∇ᵢ * g[2])])
end
# struct ∇Op{covariant} <: StaticArray{Tuple{2}, ∇i, 1} end 
# getindex(::∇Op{covariant}, i::Int) where {covariant} = ∇i{i-1,covariant}()
# const ∇ⁱ = ∇Op{false}()
# const ∇ᵢ = ∇Op{true}()
# const ∇ = ∇ⁱ # ∇ is contravariant by default unless otherwise specified
# allocate_result(::∇Op, f::Field) = @SVector[similar(f), similar(f)]
# allocate_result(::typeof(∇ⁱ'),f) = allocate_result(∇,f)
# allocate_result(::typeof(∇ᵢ'),f) = allocate_result(∇,f)
# mul!(f′::FieldVector, ∇Op::∇Op, f::Field) = @SVector[mul!(f′[1],∇Op[1],f), mul!(f′[2],∇Op[2],f)]
# struct ∇²Op <: LinOp{Basis,Spin,Pix} end
# *(::∇²Op, f::Field) = sum(diag(gradhess(f)[2]))
# const ∇² = ∇²Op()
# # this is not strictly true (∇[1] is generically a gradient w.r.t. the first
# # coordinate, e.g. ∂θ), but this is useful shorthand for the flat-sky:
# const ∂x = ∇[1]
# const ∂y = ∇[2]

    

# ### FuncOp
# 
# # An Op which applies some arbitrary function to its argument.
# # Transpose and/or inverse operations which are not specified will return an error.
# @with_kw struct FuncOp <: LinOp{Basis,Spin,Pix}
#     op   = nothing
#     opᴴ  = nothing
#     op⁻¹ = nothing
#     op⁻ᴴ = nothing
# end
# SymmetricFuncOp(;op=nothing, op⁻¹=nothing) = FuncOp(op,op,op⁻¹,op⁻¹)
# *(op::FuncOp, f::Field) = op.op   != nothing ? op.op(f)   : error("op*f not implemented")
# *(f::Field, op::FuncOp) = op.opᴴ  != nothing ? op.opᴴ(f)  : error("f*op not implemented")
# \(op::FuncOp, f::Field) = op.op⁻¹ != nothing ? op.op⁻¹(f) : error("op\\f not implemented")
# adjoint(op::FuncOp) = FuncOp(op.opᴴ,op.op,op.op⁻ᴴ,op.op⁻¹)
# const IdentityOp = FuncOp(@repeated(identity,4)...)
# inv(op::FuncOp) = FuncOp(op.op⁻¹,op.op⁻ᴴ,op.op,op.opᴴ)
# 
# 
# 
# ### BandPassOp
# 
# # An op which applies some bandpass, like a high or low-pass filter. This object
# # stores the bandpass weights, Wℓ, and each Field type F should implement
# # broadcast_data(::Type{F}, ::BandPassOp) to describe how this is actually
# # applied. 
# 
# abstract type HarmonicBasis <: Basislike end
# 
# struct BandPassOp{T<:Vector} <: LinDiagOp{HarmonicBasis,Spin,Pix}
#     ℓ::T
#     Wℓ::T
# end
# BandPassOp(ℓ,Wℓ) = BandPassOp(promote(collect(ℓ),collect(Wℓ))...)
# HighPass(ℓ,Δℓ=50) = BandPassOp(0:10000,    [zeros(ℓ-Δℓ); @.((cos($linspace(π,0,2Δℓ))+1)/2); ones(10001-ℓ-Δℓ)])
# LowPass(ℓ,Δℓ=50)  = BandPassOp(0:(ℓ+Δℓ-1), [ones(ℓ-Δℓ);  @.(cos($linspace(0,π,2Δℓ))+1)/2])
# MidPass(ℓmin,ℓmax,Δℓ=50)  = BandPassOp(0:(ℓmax+Δℓ-1), [zeros(ℓmin-Δℓ);  @.(cos($linspace(π,0,2Δℓ))+1)/2; ones(ℓmax-ℓmin-2Δℓ); @.((cos($linspace(0,π,2Δℓ))+1)/2)])
# ud_grade(b::BandPassOp, args...; kwargs...) = b
# adjoint(L::BandPassOp) = L
# 
# # An Op which turns all NaN's to zero
# const Squash = SymmetricFuncOp(op=x->broadcast(nan2zero,x))
# 
# 
# ### ParamDependentOp
# 
# # A LinOp which depends on some parameters, θ. 
# # L(;θ...) recomputes the operator at a given set of parameters, but the
# # operator can also be used as-is in which case it is evaluated at a fiducial θ
# # (which is stored inside the operator when it is first constructed). 
# struct ParamDependentOp{B, S, P, L<:LinOp{B,S,P}, F<:Function} <: LinOp{B,S,P}
#     op::L
#     recompute_function::F
#     parameters::Vector{Symbol}
# end
# function ParamDependentOp(recompute_function::Function)
#     parameters = Vector{Symbol}(Base.kwarg_decl(first(methods(recompute_function)), typeof(methods(recompute_function).mt.kwsorter)))
#     ParamDependentOp(recompute_function(), recompute_function, parameters)
# end
# (L::ParamDependentOp)(θ::NamedTuple) = L.recompute_function(;θ...)
# (L::ParamDependentOp)(;θ...) = L.recompute_function(;θ...)
# *(L::ParamDependentOp, f::Field) = L.op * f
# \(L::ParamDependentOp, f::Field) = L.op \ f
# for F in (:inv, :sqrt, :adjoint, :Diagonal, :simulate, :zero, :one, :logdet)
#     @eval $F(L::ParamDependentOp) = $F(L.op)
# end
# # the following could be changed to calling ::LinOp directly pending
# # https://github.com/JuliaLang/julia/issues/14919
# evaluate(L::Union{LinOp,Real}; θ...) = L
# evaluate(L::ParamDependentOp; θ...) = L(;θ...)
# depends_on(L::ParamDependentOp, θ) = depends_on(L, keys(θ))
# depends_on(L::ParamDependentOp, θ::Tuple) = any(L.parameters .∈ Ref(θ))
# depends_on(L,                   θ) = false
# 
# 
# ### LazyBinaryOp
# 
# # we use LazyBinaryOps to create new operators composed from other operators
# # which don't actually evaluate anything until they've been multiplied by a
# # field
# struct LazyBinaryOp{F,A<:Union{LinOp,Scalar},B<:Union{LinOp,Scalar}} <: LinOp{Basis,Spin,Pix}
#     a::A
#     b::B
#     LazyBinaryOp(op,a::A,b::B) where {A,B} = new{op,A,B}(a,b)
# end
# # creating LazyBinaryOps
# for op in (:+, :-, :*)
#     @eval ($op)(a::LinOp,  b::LinOp)  = LazyBinaryOp($op,a,b)
#     @eval ($op)(a::LinOp,  b::Scalar) = LazyBinaryOp($op,a,b)
#     @eval ($op)(a::Scalar, b::LinOp)  = LazyBinaryOp($op,a,b)
# end
# /(op::LinOp, n::Real) = LazyBinaryOp(/,op,n)
# literal_pow(::typeof(^), op::LinOp, ::Val{-1}) = inv(op)
# literal_pow(::typeof(^), op::LinOp, ::Val{n}) where {n} = LazyBinaryOp(^,op,n)
# inv(op::LinOp) = LazyBinaryOp(^,op,-1)
# -(op::LinOp) = -1 * op
# # evaluating LazyBinaryOps
# for op in (:+, :-)
#     @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
# end
# *(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
# *(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)
# \(lz::LazyBinaryOp{*}, f::Field) = lz.b \ (lz.a \ f)
# *(lz::LazyBinaryOp{^}, f::Field) = foldr((lz.b>0 ? (*) : (\)), fill(lz.a,abs(lz.b)), init=f)
# adjoint(lz::LazyBinaryOp{F}) where {F} = LazyBinaryOp(F,adjoint(lz.b),adjoint(lz.a))
# ud_grade(lz::LazyBinaryOp{op}, args...; kwargs...) where {op} = LazyBinaryOp(op,ud_grade(lz.a,args...;kwargs...),ud_grade(lz.b,args...;kwargs...))
# 
# 
# ### OuterProdOp
# 
# # an operator L which represents L = M*M'
# # this could also be represented by a LazyBinaryOp, but this allows us to 
# # do simulate(L) = M * whitenoise
# struct OuterProdOp{TM<:LinOp} <: LinOp{Basis,Spin,Pix}
#     M::TM
# end
# simulate(L::OuterProdOp{<:FullDiagOp{F}}) where {F} = L.M * white_noise(F)
# simulate(L::OuterProdOp{<:LazyBinaryOp{*}}) = L.M.a * sqrt(L.M.b) * simulate(L.M.b)
# *(L::OuterProdOp, f::Field) = L.M * (L.M' * f)
# \(L::OuterProdOp, f::Field) = L.M' \ (L.M \ f)
