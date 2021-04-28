
using CUDA
using CUDA: curand_rng
using CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSR, CuSparseMatrixCOO
using CUDA.CUSOLVER: CuQR

export cuda_gc, gpu, @gpu!, @cu!

const CuBaseField{B,M,T,A<:CuArray} = BaseField{B,M,T,A}

typealias(::Type{CuArray{T,N}}) where {T,N} = "CuArray{$T,$N}"

# a function version of @cuda which can be referenced before CUDA is
# loaded as long as it exists by run-time (unlike the macro @cuda which must
# exist at compile-time)
function cuda(f, args...; threads=256)
    @cuda threads=threads f(args...)
end

is_gpu_backed(::BaseField{B,M,T,A}) where {B,M,T,A<:CuArray} = true
global_rng_for(::Type{<:CuArray}) = curand_rng()

# handy conversion functions and macros
@doc doc"""

    @gpu! x y

Equivalent to `x = gpu(x)`, `y = gpu(y)`, etc... for any number of
listed variables. See [`gpu`](@ref).
"""
macro gpu!(vars...)
    :(begin; $((:($(esc(var)) = gpu($(esc(var)))) for var in vars)...); nothing; end)
end
@doc doc"""

    gpu(x)

Recursively moves x to GPU, but unlike `CUDA.cu`, doesn't also convert
to Float32. Equivalent to `adapt_structure(CuArray, x)`. Returns nothing.
"""
gpu(x) = adapt_structure(CuArray, x)

function adapt_structure(::Type{Mem.Unified}, x::Union{Array{T,N},CuArray{T,N}}) where {T,N}
    buf = Mem.alloc(Mem.Unified, sizeof(T) * prod(size(x)))
    y = unsafe_wrap(CuArray{T,N}, convert(CuPtr{T}, buf), size(x); own=false)
    # TODO: need to write finalizer, right now this just leaks the unified memory
    copyto!(y, x)
    return y
end
unified_gpu(x) = adapt(Mem.Unified, x)



@doc doc"""

    @cu! x y
    
Equivalent to `x = cu(x)`, `y = cu(y)`, etc... for any number of
listed variables. See `CUDA.cu`. Returns nothing.
"""
macro cu!(vars...)
    :(begin; $((:($(esc(var)) = cu($(esc(var)))) for var in vars)...); nothing; end)
end



adapt_structure(::CUDA.Float32Adaptor, proj::ProjLambert) = adapt_structure(CuArray{Float32}, proj)


_deviceid(::Type{<:CuArray}) = deviceid()


function Cℓ_to_2D(Cℓ, proj::ProjLambert{T,<:CuArray}) where {T}
    # todo: remove needing to go through cpu here:
    gpu(Complex{T}.(nan2zero.(Cℓ.(cpu(proj.ℓmag)))))
end


### misc
# the generic versions of these trigger scalar indexing of CUDA, so provide
# specialized versions: 
pinv(D::Diagonal{T,<:CuBaseField}) where {T} = Diagonal(@. ifelse(isfinite(inv(D.diag)), inv(D.diag), $zero(T)))
inv(D::Diagonal{T,<:CuBaseField}) where {T} = any(Array((D.diag.==0)[:])) ? throw(SingularException(-1)) : Diagonal(inv.(D.diag))
fill!(f::CuBaseField, x) = (fill!(f.arr,x); f)
==(a::CuBaseField, b::CuBaseField) = (==)(promote(a.arr, b.arr)...)
sum(f::CuBaseField; dims=:) = (dims == :) ? sum(f.arr) : (1 in dims) ? error("Sum over invalid dims of CuFlatS0.") : f

# adapting of SparseMatrixCSC ↔ CuSparseMatrixCSR (otherwise dense arrays created)
adapt_structure(::Type{<:CuArray}, L::SparseMatrixCSC)   = CuSparseMatrixCSR(L)
adapt_structure(::Type{<:Array},   L::CuSparseMatrixCSR) = SparseMatrixCSC(L)
adapt_structure(::Type{<:CuArray}, L::CuSparseMatrixCSR) = L
adapt_structure(::Type{<:Array},   L::SparseMatrixCSC)   = L


# CUDA somehow missing this one
# see https://github.com/JuliaGPU/CuArrays.jl/issues/103
# and https://github.com/JuliaGPU/CuArrays.jl/pull/580
ldiv!(qr::CuQR, x::CuVector) = qr.R \ (CuMatrix(qr.Q)' * x)

# some Random API which CUDA doesn't implement yet
Random.randn(rng::CUDA.CURAND.RNG, T::Random.BitFloatType) = 
    cpu(randn!(rng, CuVector{T}(undef,1)))[1]

# perhaps minor type-piracy, but this lets us simulate into a CuArray using the
# CPU random number generator
Random.randn!(rng::MersenneTwister, A::CuArray) = 
    (A .= adapt(CuArray, randn!(rng, adapt(Array, A))))

# CUDA makes some copies here as a workaround for JuliaGPU/CuArrays.jl#345 &
# NVIDIA/cuFFT#2714055 but it doesn't appear to be needed in the R2C case, and
# in the C2R case we pre-allocate the memory only once (via memoization) as well
# as doing the copy asynchronously
import CUDA.CUFFT: unsafe_execute!
using CUDA.CUFFT: rCuFFTPlan, cufftReal, cufftComplex, CUFFT_R2C, cufftExecR2C, 
    cufftExecC2R, CUFFT_C2R, unsafe_copyto!, pointer, stream

plan_buffer(x) = plan_buffer(eltype(x),size(x))
@memoize plan_buffer(T, dims, dev=deviceid()) = CuArray{T}(undef,dims...)

## might want to bring this back but need to adapt to newer CUDA verions

# function unsafe_execute!(plan::rCuFFTPlan{cufftReal,K,false,N},
#                             x::CuArray{cufftReal,N}, y::CuArray{cufftComplex,N}
#                             ) where {K,N}
#     @assert plan.xtype == CUFFT_R2C
#     cufftExecR2C(plan, x, y)
# end
# function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,false,N},
#                             x::CuArray{cufftComplex,N}, y::CuArray{cufftReal}
#                             ) where {K,N}
#     @assert plan.xtype == CUFFT_C2R
#     cufftExecC2R(plan, unsafe_copyto!(pointer(plan_buffer(x)),pointer(x),length(x),async=true,stream=stream()), y)
# end


"""
    cuda_gc()

Gargbage collect and reclaim GPU memory (technically should never be
needed to do this by hand, but sometimes helps with GPU OOM errors)
"""
function cuda_gc()
    GC.gc(true)
    CUDA.reclaim()
end
