using .CuArrays
using .CuArrays.CUDAnative
using .CuArrays.CUDAdrv: devices
using .CuArrays.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSC
using .CuArrays.CUSOLVER: CuQR

const CuFlatS0{P,T,M<:CuArray} = FlatS0{P,T,M}

# a function version of @cuda which can be referenced before CUDAnative is
# loaded as long as it exists by run-time (unlike the macro @cuda which must
# exist at compile-time)
function cuda(f, args...; threads=256)
    @cuda threads=threads f(args...)
end

is_gpu_backed(f::FlatField) = fieldinfo(f).M <: CuArray

seed_for_storage!(::Type{<:CuArray}, seed=nothing) = 
    CuArrays.CURAND.seed!((seed == nothing ? (rand(0:typemax(Int),)) : seed)...)

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
dot(a::CuFlatS0, b::CuFlatS0) = sum_kbn(Array(Map(a).Ix .* Map(b).Ix))
≈(a::CuFlatS0, b::CuFlatS0) = (firstfield(a) ≈ firstfield(b))
sum(f::CuFlatS0; dims=:) = (dims == :) ? sum(firstfield(f)) : error("Not implemented")


# some pretty low-level hacks to get a few thing broadcasting correctly for
# Complex arguments that don't currently work in CuArrays
CuArrays.CUDAnative.isfinite(x::Complex) = Base.isfinite(x)
CuArrays.CUDAnative.sqrt(x::Complex) = CuArrays.CUDAnative.sqrt(CuArrays.CUDAnative.abs(x)) * CuArrays.CUDAnative.exp(im*CuArrays.CUDAnative.angle(x)/2)
CuArrays.culiteral_pow(::typeof(^), x::Complex, ::Val{2}) = x * x


# this makes cu(::SparseMatrixCSC) return a CuSparseMatrixCSC rather than a
# dense CuArray
adapt_structure(::Type{<:CuArray}, L::SparseMatrixCSC) = CuSparseMatrixCSC(L)

# CuArrays somehow missing this one
# see https://github.com/JuliaGPU/CuArrays.jl/issues/103
# and https://github.com/JuliaGPU/CuArrays.jl/pull/580
ldiv!(qr::CuQR, x::CuVector) = qr.R \ (CuMatrix(qr.Q)' * x)

# bug in CuArrays for this one
# see https://github.com/JuliaGPU/CuArrays.jl/pull/637
mul!(C::CuVector{T},adjA::Adjoint{<:Any,<:CuSparseMatrix},B::CuVector) where {T} = mv!('C',one(T),parent(adjA),B,zero(T),C,'O')
