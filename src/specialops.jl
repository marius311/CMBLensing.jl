### FullDiagOp

# A FullDiagOp is a LinDiagOp which is stored explicitly as all of its diagonal
# coefficients in the basis in which it's diagonal. It also has the convenient
# `unsafe_invert` option which, if true, makes it so we don't have to worry
# about inverting such operators which have zero-entries. 
struct FullDiagOp{F<:Field,B,S,P} <: LinDiagOp{B,S,P}
    f::F
    unsafe_invert::Bool
    FullDiagOp(f::F, unsafe_invert=true) where {B,S,P,F<:Field{B,S,P}} = new{F,B,S,P}(f,unsafe_invert)
end

*(L::FullDiagOp{F}, f::Field) where {F} = L.unsafe_invert ? nan2zero.(L.f .* F(f)) : L.f .* F(f)
\(L::FullDiagOp{F}, f::Field) where {F} = L.unsafe_invert ? nan2zero.(L.f .\ F(f)) : L.f .\ F(f)


# non-broadcasted algebra on FullDiagOps
for op in (:+,:-,:*,:\,:/)
    @eval ($op)(L::FullDiagOp, s::Scalar)           = FullDiagOp($op(L.f, s))
    @eval ($op)(s::Scalar,     L::FullDiagOp)       = FullDiagOp($op(s,   L.f))
    # ops with FullDiagOps only produce a FullDiagOp if the two are in the same basis
    @eval ($op)(La::F, Lb::F) where {F<:FullDiagOp} = FullDiagOp($op(La.f,Lb.f))
    # if they're not, we will fall back to creating a LazyBinaryOp (see algebra.jl)
end
sqrt(f::FullDiagOp) = sqrt.(f)
adjoint(f::FullDiagOp) = conj.(f)
simulate(L::FullDiagOp{F}) where {F} = sqrt(L) .* F(white_noise(F))
broadcast_data(::Type{F}, L::FullDiagOp{F}) where {F} = broadcast_data(F,L.f)
containertype(L::FullDiagOp) = containertype(L.f)
inv(L::FullDiagOp) = FullDiagOp(L.unsafe_invert ? nan2inf.(1 ./ L.f) : 1 ./ L.f, L.unsafe_invert)
ud_grade(L::FullDiagOp{<:Field{B}}, θnew) where {B<:Union{Fourier,EBFourier,QUFourier}} = 
    FullDiagOp(B(ud_grade((L.unsafe_invert ? nan2zero.(L.f) : L.f),θnew,mode=:fourier,deconv_pixwin=false,anti_aliasing=false)))
ud_grade(L::FullDiagOp{<:Field{B}}, θnew) where {B<:Union{Map,EBMap,QUMap}} = 
    FullDiagOp(B(ud_grade((L.unsafe_invert ? nan2zero.(L.f) : L.f),θnew,mode=:map,    deconv_pixwin=false,anti_aliasing=false)))


### Derivative ops

# These ops just store the coordinate with respect to which a derivative is
# being taken, and each Field type F should implement broadcast_data(::Type{F},
# ::∂) to describe how this is actually applied. 
abstract type DerivBasis <: Basislike end
const Ð = DerivBasis
Ð!(args...) = Ð(args...)
struct ∇i{coord, covariant} <: LinOp{DerivBasis,Spin,Pix} end
function gradhess(f)
    g = ∇ⁱ*f
    g, SMatrix{2,2}([permutedims(∇ᵢ * g[1]); permutedims(∇ᵢ * g[1])])
end
const ∇⁰, ∇¹, ∇₀, ∇₁ = ∇i{0,false}(), ∇i{1,false}(), ∇i{0,true}(), ∇i{1,true}()
struct ∇Op{covariant} <: StaticArray{Tuple{2}, ∇i, 1} end 
getindex(::∇Op{covariant}, i::Int) where {covariant} = ∇i{i-1,covariant}()
const ∇ⁱ = ∇Op{false}()
const ∇ᵢ = ∇Op{true}()
const ∇ = ∇ⁱ # ∇ is contravariant by default unless otherwise specified
allocate_result(::∇Op, f::Field) = @SVector[similar(f), similar(f)]
allocate_result(::typeof(∇ⁱ'),f) = allocate_result(∇,f)
allocate_result(::typeof(∇ᵢ'),f) = allocate_result(∇,f)
mul!(f′, ∇Op::∇Op, f::Field) = @SVector[mul!(f′[1],∇Op[1],f), mul!(f′[1],∇Op[1],f)]


### FuncOp

# An Op which applies some arbitrary function to its argument.
# Transpose and/or inverse operations which are not specified will return an error.
@with_kw struct FuncOp <: LinOp{Basis,Spin,Pix}
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

struct BandPassOp{T<:Vector} <: LinDiagOp{HarmonicBasis,Spin,Pix}
    ℓ::T
    Wℓ::T
end
BandPassOp(ℓ,Wℓ) = BandPassOp(promote(collect(ℓ),collect(Wℓ))...)
HP(ℓ,Δℓ=50) = BandPassOp(0:10000,    [zeros(ℓ-Δℓ); @.((cos($linspace(π,0,2Δℓ))+1)/2); ones(10001-ℓ-Δℓ)])
LP(ℓ,Δℓ=50) = BandPassOp(0:(ℓ+Δℓ-1), [ones(ℓ-Δℓ);  @.(cos($linspace(0,π,2Δℓ))+1)/2])
ud_grade(b::BandPassOp, args...; kwargs...) = b
adjoint(L::BandPassOp) = L

# An Op which turns all NaN's to zero
const Squash = SymmetricFuncOp(op=x->broadcast(nan2zero,x))
