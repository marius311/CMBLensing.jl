
# For each Field type, lensing algorithms needs to know the basis in which lensing is a
# remapping. E.g. for FlatS0 and FlatS2 this is Map and QUMap, respectively.
# Fields should implement their own LenseBasis(::Type{F}) to specify. 
abstract type LenseOp <: LinOp{Pix,Spin,Basis} end
abstract type LenseBasis <: Basislike end
const Ł = LenseBasis

""" 
Operator which computes the lensing jacobian between time s and time t,

[δfₛ/δfₜ  δfₛ/δϕ;
 δϕ/δfₜ  δϕ/δϕ]

The operator acts on a Field2Tuple containing (f,ϕ). Depending on the underlying
lensing algorithm, you can transpose and/or invert this operator. 
 
(Note: the bottom row is trivially [0,1], but included in the computation of
the operator for logical consistency)
"""
# different algorithms may need the field at time s, time t, or both, hence both
# are stored here (e.g. LenseFlow only needs fₛ but PowerLens only needs fₜ)
struct δfϕₛ_δfϕₜ{s, t, L<:LenseOp} <: LinOp{Pix,Spin,Basis}
    L::L
    fₛ::Field
    fₜ::Field
    δfϕₛ_δfϕₜ{s,t}(L,fₛ,fₜ) where {s,t} = new{s,t,typeof(L)}(L,fₛ,fₜ)
end
δf̃_δfₜϕ(L,f̃,fₜ,::Type{Val{t}}) where {t} = δfϕₛ_δfϕₜ{1.,t}(L,f̃,fₜ)
δf̃_δfₜϕ(L,f̃,fₜ,::Type{Val{1.}}) = FuncOp(identity)
δf_δfₜϕ(L,f,fₜ,::Type{Val{t}}) where {t} = δfϕₛ_δfϕₜ{0.,t}(L,f,fₜ)
δf_δfₜϕ(L,f,fₜ,::Type{Val{0.}}) = FuncOp(identity)
δfϕ_δf̃ϕ(L,f,f̃) = δfϕₛ_δfϕₜ{0.,1.}(L,f,f̃)
δf̃ϕ_δfϕ(L,f̃,f) = δfϕₛ_δfϕₜ{1.,0.}(L,f̃,f)

@∷ const FΦTuple = Field2Tuple{<:Field,<:Field{∷,<:S0}}

# some syntactic sugar for making lensing operators that lense from time t1 to t2
struct →{t1,t2} end
→(t1::Real,t2::Real) = →{float(t1),float(t2)}()
getindex(L::LenseOp, ::→{t,t}) where {t} = 1
getindex(L::LenseOp, ::→{0.,1.}) = L
getindex(L::LenseOp, i::→)  = _getindex(L,i)
_getindex(L::LenseOp, ::→{t1,t2}) where {t1,t2} = error("Lensing from time $t1 to $t2 with $(typeof(L)) is not implemented.")


include("powerlens.jl")
include("lenseflow.jl")
