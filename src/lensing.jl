
# For each Field type, lensing algorithms needs to know the basis in which lensing is a
# remapping. E.g. for FlatS0 and FlatS2 this is Map and QUMap, respectively.
# Fields should implement their own LenseBasis(::Type{F}) to specify. 
abstract type LenseOp <: LinOp{Pix,Spin,Basis} end
LenseBasis(f::F) where {F<:Field} = LenseBasis(F)(f)
LenseBasis(a::AbstractArray{<:Field}) = map(x->LenseBasis(x),a)
LenseBasis(::Type{F}) where {F<:Field} = error("""To lense a field of type $F, LenseBasis(f::$F) needs to be implemented.""")
const Ł = LenseBasis

# operator which multiplies a field by the transpose lensing jacobian
# individual lensing algorithms will implement *(::δf̃_δfϕᵀ,::Field)
struct δf̃_δfϕᵀ{L<:LenseOp} <: LinOp{Pix,Spin,Basis}
    L::L
    f::Field
end

include("powerlens.jl")
include("lenseflow.jl")
