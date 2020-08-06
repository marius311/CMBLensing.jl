
export BilinearLens

@doc doc"""

    BilinearLens(ϕ)
    
BilinearLens is a lensing operator that computes lensing with bilinear
interpolation. The action of the operator, as well as its adjoint, inverse,
inverse-adjoint, and gradient w.r.t. ϕ can all be computed. The log-determinant
of the operation is non-zero and can't be computed. 

Internally, BilinearLens forms a sparse matrix with the interpolation weights,
which can be applied and adjoint-ed extremely fast (e.g. at least an order of
magnitude faster than LenseFlow). Inverse and inverse-adjoint lensing is
somewhat slower as it is implemented with several steps of the [preconditioned
generalized minimal residual](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)
algorithm, taking anti-lensing as the preconditioner.

!!! warning 

    Due to [this bug](https://github.com/JuliaLang/PackageCompiler.jl/issues/379)
    in PackageCompiler, currently you have to run `using SparseArrays` by hand
    in your Julia session before `BilinearLens` is available.

"""
mutable struct BilinearLens{Φ,S} <: ImplicitOp{Basis,Spin,Pix}
    ϕ :: Φ
    sparse_repr :: S
    anti_lensing_sparse_repr :: Union{S, Nothing}
end

function BilinearLens(ϕ::FlatS0)
    
    # if ϕ == 0 then just return identity operator
    if norm(ϕ) == 0
        return BilinearLens(ϕ,I,I)
    end
    
    @unpack Nside,Δx,T = fieldinfo(ϕ)
    
    # the (i,j)-th pixel is deflected to (ĩs[i],j̃s[j])
    j̃s,ĩs = getindex.((∇*ϕ)./Δx, :Ix)
    ĩs .=  ĩs  .+ (1:Nside)
    j̃s .= (j̃s' .+ (1:Nside))'
    
    # sub2ind converts a 2D index to 1D index, including wrapping at edges
    indexwrap(i) = mod(i - 1, Nside) + 1
    sub2ind(i,j) = Base._sub2ind((Nside,Nside),indexwrap(i),indexwrap(j))

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
    
    # CPU
    function compute_sparse_repr(is_gpu_backed::Val{false})
        K = Int32.(collect(flatten(repeated.(1:Nside^2,4))))
        M = similar(K)
        V = similar(K,Float32)
        for I in 1:length(ĩs)
            compute_row!(I, ĩs[I], j̃s[I], M, V)
        end
        sparse(K,M,V,Nside^2,Nside^2)
    end

    # GPU
    function compute_sparse_repr(is_gpu_backed::Val{true})
        K = CuVector{Cint}(collect(flatten(repeated.(1:Nside^2,4))))
        M = similar(K)
        V = similar(K,Float32)
        cuda(ĩs, j̃s, M, V; threads=256) do ĩs, j̃s, M, V
            index = threadIdx().x
            stride = blockDim().x
            for I in index:stride:length(ĩs)
                compute_row!(I, ĩs[I], j̃s[I], M, V)
            end
        end
        # remove once CuSparseMatrixCOO makes it into official CUDA.jl:
        if !Base.isdefined(CUSPARSE,:CuSparseMatrixCOO)
            error("To use BilinearLens on GPU, run `pkg> add https://github.com/marius311/CUDA.jl#coo` and restart Julia.")
        end
        switch2csr(CUSPARSE.CuSparseMatrixCOO{Float32}(K,M,V,(Nside^2,Nside^2)))
    end
    
    
    BilinearLens(ϕ, compute_sparse_repr(Val(is_gpu_backed(ϕ))), nothing)

end


# lazily computing the sparse representation for anti-lensing

function get_anti_lensing_sparse_repr!(Lϕ::BilinearLens)
    if Lϕ.anti_lensing_sparse_repr == nothing
        Lϕ.anti_lensing_sparse_repr = BilinearLens(-Lϕ.ϕ).sparse_repr
    end
    Lϕ.anti_lensing_sparse_repr
end


# applying various forms of the operator

function *(Lϕ::BilinearLens, f::FlatS0{P}) where {N,D,P<:Flat{N,<:Any,<:Any,D}}
    Łf = Ł(f)
    f̃ = similar(Łf)
    ds = (D == 1 ? ((),) : tuple.(1:D))
    for d in ds
        mul!(@views(f̃.Ix[:,:,d...][:]), Lϕ.sparse_repr, @views(f.Ix[:,:,d...][:]))
    end
    f̃
end

function *(Lϕ::Adjoint{<:Any,<:BilinearLens}, f::FlatS0{P}) where {N,D,P<:Flat{N,<:Any,<:Any,D}}
    Łf = Ł(f)
    f̃ = similar(Łf)
    ds = (D == 1 ? ((),) : tuple.(1:D))
    for d in ds
        mul!(@views(f̃.Ix[:,:,d...][:]), parent(Lϕ).sparse_repr', @views(f.Ix[:,:,d...][:]))
    end
    f̃
end

function \(Lϕ::BilinearLens, f::FlatS0{P}) where {N,P<:Flat{N}}
    FlatMap{P}(reshape(gmres(
        Lϕ.sparse_repr, view(f[:Ix],:),
        Pl = get_anti_lensing_sparse_repr!(Lϕ), maxiter = 5
    ), N, N))
end

function \(Lϕ::Adjoint{<:Any,<:BilinearLens}, f::FlatS0{P}) where {N,P<:Flat{N}}
    FlatMap{P}(reshape(gmres(
        parent(Lϕ).sparse_repr', view(f[:Ix],:),
        Pl = get_anti_lensing_sparse_repr!(parent(Lϕ))', maxiter = 5
    ), N, N))
end

# special cases for BilinearLens(0ϕ), which don't work with bicgstabl, 
# see https://github.com/JuliaMath/IterativeSolvers.jl/issues/271
function \(Lϕ::BilinearLens{<:Any,<:UniformScaling}, f::FlatS0{P}) where {N,P<:Flat{N}}
    Lϕ.sparse_repr \ f
end
function \(Lϕ::Adjoint{<:Any,<:BilinearLens{<:Any,<:UniformScaling}}, f::FlatS0{P}) where {N,P<:Flat{N}}
    parent(Lϕ).sparse_repr \ f
end

for op in (:*, :\)
    @eval function ($op)(Lϕ::Union{BilinearLens, Adjoint{<:Any,<:BilinearLens}}, f::FieldTuple)
        Łf = Ł(f)
        F = typeof(Łf)
        F(map(f->($op)(Lϕ,f), Łf.fs))
    end
end


# gradients

@adjoint BilinearLens(ϕ) = BilinearLens(ϕ), Δ -> (Δ,)

@adjoint function *(Lϕ::BilinearLens, f::Field{B}) where {B}
    f̃ = Lϕ * f
    function back(Δ)
        (∇' * (Ref(tuple_adjoint(Ł(Δ))) .* Ł(∇*f̃))), B(Lϕ*Δ)
    end
    f̃, back
end


# gpu

adapt_structure(storage, Lϕ::BilinearLens) = BilinearLens(adapt(storage, fieldvalues(Lϕ))...)
