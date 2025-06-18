
module CMBLensingCUDAExt

using CMBLensing

if isdefined(Base, :get_extension)
    using CUDA
    using CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSR, CuSparseMatrixCOO
else
    using ..CUDA
    using ..CUDA.CUSPARSE: CuSparseMatrix, CuSparseMatrixCSR, CuSparseMatrixCOO
end

using Adapt
using AbstractFFTs
using EllipsisNotation
using ForwardDiff
using ForwardDiff: Dual, Partials, value, partials
using GPUArrays
using LinearAlgebra
using Markdown
using Memoization
using Random
using SparseArrays
using StaticArrays
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
    cpu(randn!(rng, CUDA.CuVector{T}(undef,1)))[1]

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
            CMBLensing.arr_of_duals(T, CMBLensing.apply_plan($op, plan, arr)...)
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
# A method ambiguity between Base.reshape and ComponentArrays.reshape for 0-dim arrays
function Base.reshape(a::CuArray{T,M}, dims::Tuple{}) where {T,M}
    if prod(dims) != length(a)
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(size(a))"))
    end
  
    if 0 == M && dims == size(a)
        return a
    end
  
    GPUArrays.derive(T, 0, a, dims, 0)
end


function CMBLensing.BilinearLens(ϕ::FlatField{B1,M1,CT,AA}) where {B1,M1,CT,AA<:CuArray}
    
    # if ϕ == 0 then just return identity operator
    if norm(ϕ) == 0
        return BilinearLens(ϕ,I,I)
    end
    
    @unpack Nbatch,Nx,Ny,Δx = ϕ
    T = real(ϕ.T)
    Nbatch > 1 && error("BilinearLens with batched ϕ not implemented yet.")
    
    # the (i,j)-th pixel is deflected to (ĩs[i],j̃s[j])
    j̃s,ĩs = getindex.((∇*ϕ)./Δx, :Ix)
    ĩs .=  ĩs  .+ (1:Ny)
    j̃s .= (j̃s' .+ (1:Nx))'
    
    # sub2ind converts a 2D index to 1D index, including wrapping at edges
    indexwrap(i,N) = mod(i - 1, N) + 1
    sub2ind(i,j) = Base._sub2ind((Ny,Nx),indexwrap(i,Ny),indexwrap(j,Nx))

    # compute the 4 non-zero entries in L[I,:] (ie the Ith row of the sparse
    # lensing representation, L) and add these to the sparse constructor
    # matrices, M, and V, accordingly. this function is split off so it can be
    # called directly or used as a CUDA kernel
    function compute_row!(I, ĩ, j̃, M, V)

        # (i,j) indices of the 4 nearest neighbors
        left,right = floor(Int,ĩ) .+ (0, 1)
        top,bottom = floor(Int,j̃) .+ (0, 1)
        
        # 1-D indices of the 4 nearest neighbors
        M[4I-3:4I] .= @SVector[sub2ind(left,top), sub2ind(right,top), sub2ind(left,bottom), sub2ind(right,bottom)]
        
        # weights of these neighbors in the bilinear interpolation
        Δx⁻, Δx⁺ = ((left,right) .- ĩ)
        Δy⁻, Δy⁺ = ((top,bottom) .- j̃)
        A = @SMatrix[
            1 Δx⁻ Δy⁻ Δx⁻*Δy⁻;
            1 Δx⁺ Δy⁻ Δx⁺*Δy⁻;
            1 Δx⁻ Δy⁺ Δx⁻*Δy⁺;
            1 Δx⁺ Δy⁺ Δx⁺*Δy⁺
        ]
        V[4I-3:4I] .= inv(A)[1,:]

    end
    
    # a surprisingly large fraction of the computation for large Nside, so memoize it:
    @memoize getK(Nx,Ny) = Int32.((4:4*Nx*Ny+3) .÷ 4)

    K = CUDA.CuVector{Cint}(getK(Nx,Ny))
    M = similar(K)
    V = similar(K,T)
    CMBLensing.cuda(ĩs, j̃s, M, V; threads=256) do ĩs, j̃s, M, V
        index = CUDA.threadIdx().x
        stride = CUDA.blockDim().x
        for I in index:stride:length(ĩs)
            compute_row!(I, ĩs[I], j̃s[I], M, V)
        end
    end
    spr = CuSparseMatrixCSR(CuSparseMatrixCOO{T}(K,M,V,(Nx*Ny,Nx*Ny)))
    return CMBLensing.BilinearLens(ϕ, spr, nothing)
end



end