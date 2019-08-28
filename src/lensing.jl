
# Abstract type for lensing operators
abstract type LenseOp <: ImplicitOp{Basis,Spin,Pix} end


const FΦTuple = FieldTuple{<:Basis,<:NamedTuple{(:f,:ϕ)}}

# 
# δfϕₛ_δfϕₜ is an operator which computes the lensing jacobian between time s
# and time t,
# 
# [δfₛ/δfₜ  δfₛ/δϕ;
#  δϕ/δfₜ  δϕ/δϕ]
# 
# The operator acts on a FieldTuple containing (f,ϕ). 
# 
# Individual lensing algorithms need to implement how to apply this operator,
# inverse, and/or transpose. Different algorithms may need the field at time s,
# time t, or both, hence both are stored here
# 
# Note, the bottom row is trivially [0,1], but included to make the Jacobian
# square and easier to reason about
# 
struct δfϕₛ_δfϕₜ{s, t, L<:LenseOp, Fₛ<:Field, Fₜ<:Field} <: ImplicitOp{Basis,Spin,Pix}
    L::L
    fₛ::Fₛ
    fₜ::Fₜ
    δfϕₛ_δfϕₜ{s,t}(l::L,fₛ::Fₛ,fₜ::Fₜ) where {s,t,L,Fₛ,Fₜ} = new{s,t,L,Fₛ,Fₜ}(l,fₛ,fₜ)
end
# convenience constructors which use f̃ to mean f_(t=1) and/or f to mean f_(t=0)
δf̃ϕ_δfϕₜ(L,f̃,fₜ,::Val{t}) where {t} = δfϕₛ_δfϕₜ{1.,t}(L,f̃,fₜ)
δf̃ϕ_δfϕₜ(L,f̃,fₜ,::Val{1}) = IdentityOp
δfϕ_δfϕₜ(L,f,fₜ,::Val{t}) where {t} = δfϕₛ_δfϕₜ{0.,t}(L,f,fₜ)
δfϕ_δfϕₜ(L,f,fₜ,::Val{0}) = IdentityOp
δfϕ_δf̃ϕ(L,f,f̃) = δfϕₛ_δfϕₜ{0,1}(L,f,f̃)
δf̃ϕ_δfϕ(L,f̃,f) = δfϕₛ_δfϕₜ{1,0}(L,f̃,f)
# inverse Jacobians are the same as switching time t and s
\(J::δfϕₛ_δfϕₜ{s,t}, f::Field) where {s,t} = δfϕₛ_δfϕₜ{t,s}(J.L,J.fₜ,J.fₛ) * f


# operator for [δ/δϕ L(ϕ)*f]
# (this just involves picking out one block of δf̃ϕ_δfϕ)
δLf_δϕ(f, ϕ, ::Type{L}=LenseFlow) where {L} = δLf_δϕ(f, L(ϕ))
δLf_δϕ(f, L::LenseOp) = FuncOp(
    op  = g -> (δf̃ϕ_δfϕ(L,f,f)  * FieldTuple(g,zero(L)))[2],
    opᴴ = g -> (δf̃ϕ_δfϕ(L,f,f)' * FieldTuple(g,zero(L)))[2]
)


# some syntactic sugar for making lensing operators that lense from time t1 to t2
struct →{t1,t2} end
→(t1::Real,t2::Real) = →{float(t1),float(t2)}()
getindex(L::LenseOp, ::→{t,t}) where {t} = 1
getindex(L::LenseOp, ::→{0,1}) = L
getindex(L::LenseOp, i::→)  = _getindex(L,i)
_getindex(L::LenseOp, ::→{t1,t2}) where {t1,t2} = error("Lensing from time $t1 to $t2 with $(typeof(L)) is not implemented.")

struct NoLensing <: LenseOp end
NoLensing(ϕ) = NoLensing()
cache(::NoLensing, ::Field) = NoLensing()
cache!(::NoLensing, ::Field) = NoLensing()
*(::NoLensing, f::Field) = f
\(::NoLensing, f::Field) = f
adjoint(L::NoLensing) = L
_getindex(L::NoLensing, i::→) = L
δfϕₛ_δfϕₜ{t₀,t₁}(L::NoLensing,::Any,::Any) where {t₀,t₁} = IdentityOp
