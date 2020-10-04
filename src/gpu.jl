
using CUDA
using CUDA: cufunc, curand_rng
using CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSR, mv!, switch2csr
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

function copyto!(dst::F, src::F) where {F<:CuFlatS0}
    copyto!(firstfield(dst),firstfield(src))
    dst
end
pinv(D::Diagonal{T,<:CuFlatS0}) where {T} = Diagonal(@. ifelse(isfinite(inv(D.diag)), inv(D.diag), $zero(T)))
inv(D::Diagonal{T,<:CuFlatS0}) where {T} = any(Array((D.diag.==0)[:])) ? throw(SingularException(-1)) : Diagonal(inv.(D.diag))
fill!(f::CuFlatS0, x) = (fill!(firstfield(f),x); f)
dot(a::CuFlatS0{<:Flat{N,θ}}, b::CuFlatS0{<:Flat{N,θ}}) where {N,θ} = dot(adapt(Array,Map(a)), adapt(Array,Map(b)))
≈(a::CuFlatS0, b::CuFlatS0) = (firstfield(a) ≈ firstfield(b))
sum(f::CuFlatS0; dims=:) = ((dims == :) || (1 in dims)) ? sum(firstfield(f)) : f


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
Assumes the SLURM variables `SLURM_STEP_GPUS` and `GPU_DEVICE_ORDINAL` are
defined on the workers.
"""
function assign_GPU_workers()
    topo = pmap(workers()) do i
        hostname = gethostname()
        virtgpus = parse.(Int,split(ENV["GPU_DEVICE_ORDINAL"],","))
        if "SLURM_STEP_GPUS" in keys(ENV)
            physgpus = parse.(Int,split(ENV["SLURM_STEP_GPUS"],","))
        else
            # SLURM_STEP_GPUS seems not correctly set on all systems. this
            # will work if you requested a full node's worth of GPUs at least
            physgpus = virtgpus
        end
        (i=i, hostname=hostname, virtgpus=virtgpus, physgpus=physgpus)
    end
    claimed = Set()
    assignments = Dict(map(topo) do (i,hostname,physgpus,virtgpus)
        for (physgpu,virtgpu) in zip(physgpus,virtgpus)
            if !((hostname,physgpu) in claimed)
                push!(claimed,(hostname,physgpu))
                return i => virtgpu
            end
        end
    end)
    @everywhere workers() device!($assignments[myid()])
end