
using CUDA
using CUDA: cufunc, curand_rng
using CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSR, CuSparseMatrixCOO
using CUDA.CUSOLVER: CuQR

const CuFlatS0{P,T,M<:CuArray} = FlatS0{P,T,M}

# a function version of @cuda which can be referenced before CUDA is
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
    Broadcasted{Nothing}(cufunc(bc.f), preprocess_args(dest, bc.args), map(OneTo,content_size(F)))
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

pinv(D::Diagonal{T,<:CuFlatS0}) where {T} = Diagonal(@. ifelse(isfinite(inv(D.diag)), inv(D.diag), $zero(T)))
inv(D::Diagonal{T,<:CuFlatS0}) where {T} = any(Array((D.diag.==0)[:])) ? throw(SingularException(-1)) : Diagonal(inv.(D.diag))
fill!(f::CuFlatS0, x) = (fill!(firstfield(f),x); f)
==(a::CuFlatS0, b::CuFlatS0) = (==)(firstfield.(promote(a,b))...)
≈(a::CuFlatS0, b::CuFlatS0) = (≈)(firstfield.(promote(a,b))...)
≈(a::Diagonal{<:Any,<:CuFlatS0}, b::Diagonal{<:Any,<:CuFlatS0}) = basis(diag(a)) == basis(diag(b)) && diag(a) ≈ diag(b)
sum(f::CuFlatS0; dims=:) = (dims == :) ? sum(firstfield(f)) : (1 in dims) ? error("Sum over invalid dims of CuFlatS0.") : f


# these only work for Reals in CUDA
# with these definitions, they work for Complex as well
CUDA.isfinite(x::Complex) = Base.isfinite(x)
CUDA.sqrt(x::Complex) = CUDA.sqrt(CUDA.abs(x)) * CUDA.exp(im*CUDA.angle(x)/2)
CUDA.culiteral_pow(::typeof(^), x::Complex, ::Val{2}) = x * x


# this makes cu(::SparseMatrixCSC) return a CuSparseMatrixCSR rather than a
# dense CuArray
adapt_structure(::Type{<:CuArray}, L::SparseMatrixCSC) = CuSparseMatrixCSR(L)

# CUDA somehow missing this one
# see https://github.com/JuliaGPU/CuArrays.jl/issues/103
# and https://github.com/JuliaGPU/CuArrays.jl/pull/580
ldiv!(qr::CuQR, x::CuVector) = qr.R \ (CuMatrix(qr.Q)' * x)

# some Random API which CUDA doesn't implement yet
Random.randn(rng::CUDA.CURAND.RNG, T::Random.BitFloatType) = 
    adapt(Array,randn!(rng, CuVector{T}(undef,1)))[1]

# perhaps minor type-piracy, but this lets us simulate into a CuArray using the
# CPU random number generator
Random.randn!(rng::MersenneTwister, A::CuArray{T}) where {T} = 
    (A .= adapt(CuArray{T}, randn!(rng, adapt(Array{T},A))))

# CUDA makes some copies here as a workaround for JuliaGPU/CuArrays.jl#345 &
# NVIDIA/cuFFT#2714055 but it doesn't appear to be needed in the R2C case, and
# in the C2R case we pre-allocate the memory only once (via memoization) as well
# as doing the copy asynchronously
import CUDA.CUFFT: unsafe_execute!
using CUDA.CUFFT: rCuFFTPlan, cufftReal, cufftComplex, CUFFT_R2C, cufftExecR2C, cufftExecC2R, CUFFT_C2R, unsafe_copyto!, CuDefaultStream, pointer

plan_buffer(x) = plan_buffer(eltype(x),size(x))
@memoize plan_buffer(T,dims) = CuArray{T}(undef,dims...)

function unsafe_execute!(plan::rCuFFTPlan{cufftReal,K,false,N},
                            x::CuArray{cufftReal,N}, y::CuArray{cufftComplex,N}
                            ) where {K,N}
    @assert plan.xtype == CUFFT_R2C
    cufftExecR2C(plan, x, y)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,false,N},
                            x::CuArray{cufftComplex,N}, y::CuArray{cufftReal}
                            ) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    cufftExecC2R(plan, unsafe_copyto!(pointer(plan_buffer(x)),pointer(x),length(x),async=true,stream=CuDefaultStream()), y)
end

# monkey-patched version of https://github.com/JuliaGPU/CUDA.jl/pull/436
# until it hits a release
using CUDA.CURAND: curandSetPseudoRandomGeneratorSeed, curandSetGeneratorOffset, 
    CURAND_STATUS_ALLOCATION_FAILED, CURAND_STATUS_PREEXISTING_FAILURE, CURAND_STATUS_SUCCESS,
    unsafe_curandGenerateSeeds, throw_api_error, @retry_reclaim, RNG

function Random.seed!(rng::RNG, seed=Base.rand(UInt64), offset=0)
    curandSetPseudoRandomGeneratorSeed(rng, seed)
    curandSetGeneratorOffset(rng, offset)
    res = @retry_reclaim err->isequal(err, CURAND_STATUS_ALLOCATION_FAILED) ||
                              isequal(err, CURAND_STATUS_PREEXISTING_FAILURE) begin
        unsafe_curandGenerateSeeds(rng)
    end
    if res != CURAND_STATUS_SUCCESS
        throw_api_error(res)
    end
    return
end



gc = () -> (GC.gc(true); CUDA.reclaim())


"""
    assign_GPU_workers()

Assuming you submitted a SLURM job and got several GPUs, possibly across several
nodes, this assigns each Julia worker process a unique GPU using `CUDA.device!`.
"""
function assign_GPU_workers()
    @everywhere @eval Main using CUDA, Distributed
    accessible_gpus = @eval Main Dict(pmap(workers()) do _
        ds = CUDA.devices()
        myid() => Dict(CUDA.deviceid.(ds) .=> CUDA.uuid.(ds))
    end)
    claimed = Set()
    assignments = Dict(map(workers()) do myid
        for (gpu_id, gpu_uuid) in accessible_gpus[myid]
            if !(gpu_uuid in claimed)
                push!(claimed, gpu_uuid)
                return myid => gpu_id
            end
        end
    end)
    @everywhere workers() device!($assignments[myid()])
end