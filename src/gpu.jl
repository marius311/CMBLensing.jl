using .CuArrays
using Serialization
import Serialization: serialize

const CuFlatS0{P,T,M<:CuArray} = FlatS0{P,T,M}

### broadcasting
preprocess(dest::F, bc::Broadcasted) where {F<:CuFlatS0} = 
    Broadcasted{Nothing}(CuArrays.cufunc(bc.f), preprocess_args(dest, bc.args), map(OneTo,size_2d(F)))
preprocess(dest::F, arg) where {M,F<:CuFlatS0{<:Any,<:Any,M}} = 
    adapt(M,broadcastable(F, arg))
function copyto!(dest::F, bc::Broadcasted{Nothing}) where {F<:CuFlatS0}
    bc′ = preprocess(dest, bc)
    copyto!(firstfield(dest), bc′)
    return dest
end
BroadcastStyle(::FlatS0Style{F,Array}, ::FlatS0Style{F,CuArray}) where {P,F<:FlatS0{P}} = 
    FlatS0Style{basetype(F){P},CuArray}()


# always adapt to Array storage when serializing since we may deserialize in an
# environment that does not have CuArrays loaded 
serialize(s::AbstractSerializer, f::CuFlatS0) = serialize(s, adapt(Array,f))


### misc
# the generic versions of these trigger scalar indexing of CuArrays, so provide
# specialized versions: 

function copyto!(dst::F, src::F) where {F<:CuFlatS0}
    copyto!(firstfield(dst),firstfield(src))
    dst
end
pinv(D::Diagonal{T,<:CuFlatS0}) where {T} = Diagonal(@. ifelse(isfinite(inv(D.diag)), inv(D.diag), $zero(T)))
inv(D::Diagonal{T,<:CuFlatS0}) where {T} = any(Array((D.diag.==0)[:])) ? throw(SingularException(-1)) : Diagonal(inv.(D.diag))
fill!(f::CuFlatS0, x) = (fill!(firstfield(f),x); f)
dot(a::CuFlatS0, b::CuFlatS0) = sum_kbn(Array(Map(a).Ix .* Map(b).Ix)) * fieldinfo(a).Δx^2
≈(a::CuFlatS0, b::CuFlatS0) = (firstfield(a) ≈ firstfield(b))

# some pretty low-level hacks to get a few thing broadcasting correctly for
# Complex arguments that don't currently work in CuArrays
CuArrays.CUDAnative.isfinite(x::Complex) = Base.isfinite(x)
CuArrays.CUDAnative.sqrt(x::Complex) = CuArrays.CUDAnative.sqrt(CuArrays.CUDAnative.abs(x)) * CuArrays.CUDAnative.exp(im*CuArrays.CUDAnative.angle(x)/2)
CuArrays.culiteral_pow(::typeof(^), x::Complex, ::Val{2}) = x * x
