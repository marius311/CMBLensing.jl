
using CUDA
using CUDA: cufunc, curand_rng
using CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSC, mv!
using CUDA.CUSOLVER: CuQR

const CuFlatS0{P,T,M<:CuArray} = FlatS0{P,T,M}

# a function version of @cuda which can be referenced before CUDAnative is
# loaded as long as it exists by run-time (unlike the macro @cuda which must
# exist at compile-time)
function cuda(f, args...; threads=256)
    @cuda threads=threads f(args...)
end

is_gpu_backed(f::FlatField) = fieldinfo(f).M <: CuArray
global_rng_for(::Type{<:CuArray}) = curand_rng()
seed_for_storage!(::Type{<:CuArray}, seed=nothing) = 
    Random.seed!(global_rng_for(CuArray), seed)



### broadcasting
preprocess(dest::F, bc::Broadcasted) where {F<:CuFlatS0} = 
    Broadcasted{Nothing}(cufunc(bc.f), preprocess_args(dest, bc.args), map(OneTo,_size(F)))
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
# the generic versions of these trigger scalar indexing of CUDA, so provide
# specialized versions: 

function copyto!(dst::F, src::F) where {F<:CuFlatS0}
    copyto!(firstfield(dst),firstfield(src))
    dst
end
pinv(D::Diagonal{T,<:CuFlatS0}) where {T} = Diagonal(@. ifelse(isfinite(inv(D.diag)), inv(D.diag), $zero(T)))
inv(D::Diagonal{T,<:CuFlatS0}) where {T} = any(Array((D.diag.==0)[:])) ? throw(SingularException(-1)) : Diagonal(inv.(D.diag))
fill!(f::CuFlatS0, x) = (fill!(firstfield(f),x); f)
dot(a::CuFlatS0{<:Flat{N,θ}}, b::CuFlatS0{<:Flat{N,θ}}) where {N,θ} = dot(adapt(Array,Map(a)), adapt(Array,Map(b)))
≈(a::CuFlatS0, b::CuFlatS0) = (firstfield(a) ≈ firstfield(b))
sum(f::CuFlatS0; dims=:) = (dims == :) ? sum(firstfield(f)) : error("Not implemented")


# these only work for Reals in CUDA
# with these definitions, they work for Complex as well
CUDA.isfinite(x::Complex) = Base.isfinite(x)
CUDA.sqrt(x::Complex) = CUDA.sqrt(CUDA.abs(x)) * CUDA.exp(im*CUDA.angle(x)/2)
CUDA.culiteral_pow(::typeof(^), x::Complex, ::Val{2}) = x * x


# this makes cu(::SparseMatrixCSC) return a CuSparseMatrixCSC rather than a
# dense CuArray
adapt_structure(::Type{<:CuArray}, L::SparseMatrixCSC) = CuSparseMatrixCSC(L)

# CUDA somehow missing this one
# see https://github.com/JuliaGPU/CuArrays.jl/issues/103
# and https://github.com/JuliaGPU/CuArrays.jl/pull/580
ldiv!(qr::CuQR, x::CuVector) = qr.R \ (CuMatrix(qr.Q)' * x)

# bug in CUDA for this one
# see https://github.com/JuliaGPU/CuArrays.jl/pull/637
mul!(C::CuVector{T},adjA::Adjoint{<:Any,<:CuSparseMatrix},B::CuVector) where {T} = 
    mv!('C',one(T),parent(adjA),B,zero(T),C,'O')

# some Random API which CUDA doesn't implement yet
Random.randn(rng::CUDA.CURAND.RNG, T::Random.BitFloatType) = 
    adapt(Array,randn!(rng, CuVector{T}(undef,1)))[1]
Random.seed!(rng::CUDA.CURAND.RNG, ::Nothing) = Random.seed!(rng)
