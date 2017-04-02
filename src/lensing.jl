
# For each Field type, lensing algorithms needs to know the basis in which lensing is a
# remapping. E.g. for FlatS0 and FlatS2 this is Map and QUMap, respectively.
# Fields should implement their own LenseBasis(::Type{F}) to specify. 
abstract type LenseOp <: LinOp{Pix,Spin,Basis} end
abstract type LenseBasis <: Basislike end
const Ł = LenseBasis

# operator which right-multiplies a field by the lensing jacobian between time s and time t
# individual lensing algorithms will implement *(::Field, ::δf_δfϕ)
# different algorithms may need the field at time s, time t, or both, hence both
# are stored here (e.g. LenseFlow only needs fₛ but PowerLense needs fₜ)
struct δfₛ_δfₜϕ{s, t, L<:LenseOp} <: LinOp{Pix,Spin,Basis}
    L::L
    fₛ::Field
    fₜ::Field
    δfₛ_δfₜϕ{s,t}(L,fₛ,fₜ) where {s,t} = new{s,t,typeof(L)}(L,fₛ,fₜ)
end
struct δfₜ_δfₜϕ <: LinOp{Pix,Spin,Basis} end
*(f::Field, ::δfₜ_δfₜϕ) = (f,0)
δf̃_δfₜϕ(L,f̃,fₜ,::Type{Val{t}}) where {t} = δfₛ_δfₜϕ{1.,t}(L,f̃,fₜ)
δf̃_δfₜϕ(L,f̃,fₜ,::Type{Val{1.}}) = δfₜ_δfₜϕ()
δf_δfₜϕ(L,f,fₜ,::Type{Val{t}}) where {t} = δfₛ_δfₜϕ{0.,t}(L,f,fₜ)
δf_δfₜϕ(L,f,fₜ,::Type{Val{0.}}) = δfₜ_δfₜϕ()


# some syntactic sugar for making lensing operators that lense from time t1 to t2
struct →{t1,t2} end
→(t1::Real,t2::Real) = →{float(t1),float(t2)}()
getindex(L::LenseOp, ::→{t,t}) where {t} = 1
getindex(L::LenseOp, ::→{0.,1.}) = L
getindex(L::LenseOp, i::→)  = _getindex(L,i)
_getindex(L::LenseOp, ::→{t1,t2}) where {t1,t2} = error("Lensing from time $t1 to $t2 with $(typeof(L)) is not implemented.")


include("powerlens.jl")
include("lenseflow.jl")
