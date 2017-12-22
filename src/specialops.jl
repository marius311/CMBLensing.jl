### FullDiagOp

# A LinDiagOp which is stored explicitly as all of its diagonal coefficients in
# the basis in which it's diagonal. Additionally, `unsafe_invert=true` makes it
# so we don't have to worry about inverting such ops which have zero-entries. 
struct FullDiagOp{F<:Field,P,S,B} <: LinDiagOp{P,S,B}
    f::F
    unsafe_invert::Bool
    FullDiagOp(f::F, unsafe_invert=true) where {P,S,B,F<:Field{P,S,B}} = new{F,P,S,B}(f,unsafe_invert)
end
for op=(:*,:\)
    @eval @∷ ($op)(O::FullDiagOp{<:Field{∷,∷,B}}, f::Field{∷,∷,B}) where {B} = O.unsafe_invert ? nan2zero.($(Symbol(:.,op))(O.f,f)) : $(Symbol(:.,op))(O.f,f)
end
sqrtm(f::FullDiagOp) = sqrt.(f)
simulate(Σ::FullDiagOp{F}) where {F} = sqrtm(Σ) .* F(white_noise(F))
broadcast_data(::Type{F}, op::FullDiagOp{F}) where {F} = broadcast_data(F,op.f)
containertype(op::FullDiagOp) = containertype(op.f)
inv(op::FullDiagOp) = FullDiagOp(op.unsafe_invert ? nan2inf.(1./op.f) : 1./op.f, op.unsafe_invert)



### Derivative ops

# These ops just store the coordinate with respect to which a derivative is
# being taken, and each Field type F should implement broadcast_data(::Type{F},
# ::∂) to describe how this is actually applied. 

abstract type DerivBasis <: Basislike end
const Ð = DerivBasis
struct ∂{s} <: LinDiagOp{Pix,Spin,DerivBasis} end
const ∂x,∂y= ∂{:x}(),∂{:y}()
const ∇ = @SVector [∂x,∂y]
*(∂::∂, f::Field) = ∂ .* Ð(f)
function gradhess(f)
    (∂xf,∂yf)=∇*Ð(f)
    ∂xyf = ∂x*∂yf
    @SVector([∂xf,∂yf]), @SMatrix([∂x*∂xf ∂xyf; ∂xyf ∂y*∂yf])
end
shortname(::Type{∂{s}}) where {s} = "∂$s"
struct ∇²Op <: LinDiagOp{Pix,Spin,DerivBasis} end
const ∇² = ∇²Op()
*(∇²::∇²Op, f::Field) = ∇² .* Ð(f)


### FuncOp

# An Op which applies some arbitrary function to its argument.
# Transpose and/or inverse operations which are not specified will return an error.
@with_kw struct FuncOp <: LinOp{Pix,Spin,Basis}
    op   = nothing
    opᴴ  = nothing
    op⁻¹ = nothing
    op⁻ᴴ = nothing
end
SymmetricFuncOp(;op=nothing, op⁻¹=nothing) = FuncOp(op,op,op⁻¹,op⁻¹)
@∷ *(op::FuncOp, f::Field) = op.op   != nothing ? op.op(f)   : error("op*f not implemented")
@∷ *(f::Field, op::FuncOp) = op.opᴴ  != nothing ? op.opᴴ(f)  : error("f*op not implemented")
@∷ \(op::FuncOp, f::Field) = op.op⁻¹ != nothing ? op.op⁻¹(f) : error("op\\f not implemented")
ctranspose(op::FuncOp) = FuncOp(op.opᴴ,op.op,op.op⁻ᴴ,op.op⁻¹)
const IdentityOp = FuncOp(repeated(identity,4)...)
inv(op::FuncOp) = FuncOp(op.op⁻¹,op.op⁻ᴴ,op.op,op.opᴴ)



### BandPassOp

# An op which applies some bandpass, like a high or low-pass filter. This object
# stores the bandpass weights, Wℓ, and each Field type F should implement
# broadcast_data(::Type{F}, ::BandPassOp) to describe how this is actually
# applied. 

struct BandPassOp{T<:Vector} <: LinDiagOp{Pix,Spin,DerivBasis}
    ℓ::T
    Wℓ::T
end
BandPassOp(ℓ,Wℓ) = BandPassOp(promote(collect(ℓ),collect(Wℓ))...)
HP(ℓ,Δℓ=50) = BandPassOp(0:10000,    [zeros(ℓ-Δℓ); @.((cos($linspace(π,0,2Δℓ))+1)/2); ones(10001-ℓ-Δℓ)])
LP(ℓ,Δℓ=50) = BandPassOp(0:(ℓ+Δℓ-1), [ones(ℓ-Δℓ);  @.(cos($linspace(0,π,2Δℓ))+1)/2])
*(op::BandPassOp,f::Field) = op .* Ð(f)
(::Type{FullDiagOp{F}})(b::BandPassOp) where {F<:Field} = FullDiagOp(F(broadcast_data(F,b)...))


# An Op which turns all NaN's to zero
const Squash = SymmetricFuncOp(op=x->broadcast(nan2zero,x))
