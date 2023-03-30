
module CMBLensingCUDAExt

using Adapt
using AbstractFFTs
using CMBLensing
using CUDA
using CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSR, CuSparseMatrixCOO
using EllipsisNotation
using ForwardDiff
using ForwardDiff: Dual, Partials, value, partials
using LinearAlgebra
using Markdown
using Random
using SparseArrays
using Zygote

const CuBaseField{B,M,T,A<:CuArray} = BaseField{B,M,T,A}

# printing
CMBLensing.typealias(::Type{CuArray{T,N}}) where {T,N} = "CuArray{$T,$N}"
Base.print_array(io::IO, X::Diagonal{<:Any,<:CuBaseField}) = Base.print_array(io, cpu(X))


# a function version of @cuda which can be referenced before CUDA is
# loaded as long as it exists by run-time (unlike the macro @cuda which must
# exist at compile-time)
function CMBLensing.cuda(f, args...; threads=256)
    @cuda threads=threads f(args...)
end

CMBLensing.is_gpu_backed(::BaseField{B,M,T,A}) where {B,M,T,A<:CuArray} = true
CMBLensing.gpu(x) = Adapt.adapt_structure(CuArray, x)


function CMBLensing.Cℓ_to_2D(Cℓ, proj::ProjLambert{T,<:CuArray}) where {T}
    # todo: remove needing to go through cpu here:
    gpu(T.(nan2zero.(Cℓ.(cpu(proj.ℓmag)))))
end


### misc
# the generic versions of these trigger scalar indexing of CUDA, so provide
# specialized versions: 
LinearAlgebra.pinv(D::Diagonal{T,<:CuBaseField}) where {T} = Diagonal(@. ifelse(isfinite(inv(D.diag)), inv(D.diag), $zero(T)))
LinearAlgebra.inv(D::Diagonal{T,<:CuBaseField}) where {T} = any(Array((D.diag.==0)[:])) ? throw(SingularException(-1)) : Diagonal(inv.(D.diag))
Base.fill!(f::CuBaseField, x) = (fill!(f.arr,x); f)
Base.sum(f::CuBaseField; dims=:) = (dims == :) ? CMBLensing.sum_dropdims(f.arr) : (1 in dims) ? error("Sum over invalid dims of CuFlatS0.") : f

# adapting of SparseMatrixCSC ↔ CuSparseMatrixCSR (otherwise dense arrays created)
Adapt.adapt_structure(::Type{<:CuArray}, L::SparseMatrixCSC)   = CuSparseMatrixCSR(L)
Adapt.adapt_structure(::Type{<:Array},   L::CuSparseMatrixCSR) = SparseMatrixCSC(L)
Adapt.adapt_structure(::Type{<:CuArray}, L::CuSparseMatrixCSR) = L
Adapt.adapt_structure(::Type{<:Array},   L::SparseMatrixCSC)   = L

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
function CMBLensing.cuda_gc()
    isdefined(Main,:Out) && empty!(Main.Out)
    GC.gc(true)
    CUDA.reclaim()
end

CMBLensing.unsafe_free!(x::CuArray) = CUDA.unsafe_free!(x)

@static if CMBLensing.versionof(Zygote)>v"0.6.11"
    # https://github.com/JuliaGPU/CUDA.jl/issues/982
    dot(x::CuArray, y::CuArray) = sum(conj.(x) .* y)
end

# prevents unnecessary CuArray views in some cases
Base.view(arr::CuArray{T,2}, I, J, K, ::typeof(..)) where {T} = view(arr, I, J, K)
Base.view(arr::CuArray{T,3}, I, J, K, ::typeof(..)) where {T} = view(arr, I, J, K)


## ForwardDiff through FFTs 
# these definitions needed bc the CUDA.jl definitions supersede the
# AbstractArray ones in autodiff.jl

for P in [AbstractFFTs.Plan, AbstractFFTs.ScaledPlan]
    for op in [:(Base.:*), :(Base.:\)]
        @eval function ($op)(plan::$P, arr::CuArray{<:Union{Dual{T},Complex{<:Dual{T}}}}) where {T}
            arr_of_duals(T, apply_plan($op, plan, arr)...)
        end
    end
end

AbstractFFTs.plan_fft(arr::CuArray{<:Complex{<:Dual}}, region) = plan_fft(complex.(value.(real.(arr)), value.(imag.(arr))), region)
AbstractFFTs.plan_rfft(arr::CuArray{<:Dual}, region; kws...) = plan_rfft(value.(arr), region; kws...)

# until something like https://github.com/JuliaDiff/ForwardDiff.jl/pull/619
function ForwardDiff.extract_gradient!(::Type{T}, result::CuArray, dual::Dual) where {T}
    result[:] .= partials(T, dual)
    return result
end
function ForwardDiff.extract_gradient_chunk!(::Type{T}, result::CuArray, dual, index, chunksize) where {T}
    result[index:index+chunksize-1] .= partials.(T, dual, 1:chunksize)
    return result
end

# fix for https://github.com/jonniedie/ComponentArrays.jl/issues/193
function Base.reshape(a::CuArray{T,M}, dims::Tuple{}) where {T,M}
    if prod(dims) != length(a)
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(size(a))"))
    end
  
    if 0 == M && dims == size(a)
        return a
    end
  
    CUDA._derived_array(T, 0, a, dims)
end


end