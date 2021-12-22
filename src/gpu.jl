
using CUDA
using CUDA: curand_rng
using CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSR, CuSparseMatrixCOO
using CUDA.CUSOLVER: CuQR

export cuda_gc, gpu

const CuBaseField{B,M,T,A<:CuArray} = BaseField{B,M,T,A}

# printing
typealias(::Type{CuArray{T,N}}) where {T,N} = "CuArray{$T,$N}"
Base.print_array(io::IO, X::Diagonal{<:Any,<:CuBaseField}) = Base.print_array(io, cpu(X))


# a function version of @cuda which can be referenced before CUDA is
# loaded as long as it exists by run-time (unlike the macro @cuda which must
# exist at compile-time)
function cuda(f, args...; threads=256)
    @cuda threads=threads f(args...)
end

is_gpu_backed(::BaseField{B,M,T,A}) where {B,M,T,A<:CuArray} = true

# handy conversion functions and macros
@doc doc"""

    gpu(x)

Recursively moves x to GPU, but unlike `CUDA.cu`, doesn't also convert
to Float32. Equivalent to `adapt_structure(CuArray, x)`. Returns nothing.
"""
gpu(x) = adapt_structure(CuArray, x)



adapt_structure(::CUDA.CuArrayAdaptor, proj::ProjLambert) = adapt_structure(CuArray{Float32}, proj)


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


"""
    cuda_gc()

Gargbage collect and reclaim GPU memory (technically should never be
needed to do this by hand, but sometimes helps with GPU OOM errors)
"""
function cuda_gc()
    isdefined(Main,:Out) && empty!(Main.Out)
    GC.gc(true)
    CUDA.reclaim()
end

@static if versionof(Zygote)>v"0.6.11"
    # https://github.com/JuliaGPU/CUDA.jl/issues/982
    dot(x::CuArray, y::CuArray) = sum(conj.(x) .* y)
end

# prevents unnecessary CuArray views in some cases
Base.view(arr::CuArray{T,3}, I, J, K, ::typeof(..)) where {T} = view(arr, I, J, K)
