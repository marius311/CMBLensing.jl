
export BilinearLens

@doc doc"""

    BilinearLens(ϕ)

`BilinearLens` is a lensing operator that computes lensing with
bilinear interpolation. The action of the operator, as well as its
adjoint, inverse, inverse-adjoint, and gradient w.r.t. `ϕ` can all be
computed. The log-determinant of the operation is non-zero and can't
be computed. 

Internally, `BilinearLens` forms a sparse matrix with the
interpolation weights, which can be applied and adjoint-ed extremely
fast (e.g. at least an order of magnitude faster than
[`LenseFlow`](@ref)). Inverse and inverse-adjoint lensing is somewhat
slower since it requires an iterative solve, here performed with the
[preconditioned generalized minimal
residual](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)
algorithm. 

"""
mutable struct BilinearLens{T,Φ<:Field{<:Any,T},S} <: ImplicitOp{T}
    ϕ :: Φ
    sparse_repr :: S
    anti_lensing_sparse_repr :: Union{S, Nothing}
end


function BilinearLens(ϕ::FlatField{B1,M1,CT,AA}) where {B1,M1,CT,AA<:AbstractArray}
    
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

    K = Vector{Int32}(getK(Nx,Ny))
    M = similar(K)
    V = similar(K,T)
    for I in 1:length(ĩs)
        compute_row!(I, ĩs[I], j̃s[I], M, V)
    end
    spr = sparse(K,M,V,Nx*Ny,Nx*Ny)
    return BilinearLens(ϕ, spr, nothing)
end


# lazily computing the sparse representation for anti-lensing

function get_anti_lensing_sparse_repr!(Lϕ::BilinearLens)
    if Lϕ.anti_lensing_sparse_repr == nothing
        Lϕ.anti_lensing_sparse_repr = BilinearLens(-Lϕ.ϕ).sparse_repr
    end
    Lϕ.anti_lensing_sparse_repr
end


getϕ(Lϕ::BilinearLens) = Lϕ.ϕ
(Lϕ::BilinearLens)(ϕ::FlatField) = BilinearLens(ϕ)
hash(L::BilinearLens, h::UInt64) = foldr(hash, (typeof(L), getϕ(L)), init=h)


# applying various forms of the operator

function *(Lϕ::BilinearLens, f::FlatField)
    Lϕ.sparse_repr===I && return f
    Łf = Ł(f)
    f̃ = similar(Łf)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        mul!(@views(f̃.arr[:,:,pol,batch][:]), Lϕ.sparse_repr, @views(Łf.arr[:,:,pol,batch][:]))
    end
    f̃
end

function *(Lϕ::Adjoint{<:Any,<:BilinearLens}, f::FlatField)
    parent(Lϕ).sparse_repr===I && return f
    Łf = Ł(f)
    f̃ = similar(Łf)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        mul!(@views(f̃.arr[:,:,pol,batch][:]), parent(Lϕ).sparse_repr', @views(Łf.arr[:,:,pol,batch][:]))
    end
    f̃
end

function \(Lϕ::BilinearLens, f̃::FlatField)
    Lϕ.sparse_repr===I && return f̃
    Łf̃ = Ł(f̃)
    f = similar(Łf̃)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        @views(f.arr[:,:,pol,batch][:]) .= gmres(
            Lϕ.sparse_repr, @views(Łf̃.arr[:,:,pol,batch][:]),
            Pl = get_anti_lensing_sparse_repr!(Lϕ), maxiter = 5
        )
    end
    f
end

function \(Lϕ::Adjoint{<:Any,<:BilinearLens}, f̃::FlatField)
    parent(Lϕ).sparse_repr===I && return f̃
    Łf̃ = Ł(f̃)
    f = similar(Łf̃)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        @views(f.arr[:,:,pol,batch][:]) .= gmres(
            parent(Lϕ).sparse_repr', @views(Łf̃.arr[:,:,pol,batch][:]),
            Pl = get_anti_lensing_sparse_repr!(parent(Lϕ))', maxiter = 5
        )
    end
    f
end


for op in (:*, :\)
    @eval function ($op)(Lϕ::Union{BilinearLens, Adjoint{<:Any,<:BilinearLens}}, f::FieldTuple)
        FieldTuple(map(f->($op)(Lϕ,f), f.fs))
    end
end


# gradients

@adjoint BilinearLens(ϕ) = BilinearLens(ϕ), Δ -> (Δ,)

@adjoint function *(Lϕ::BilinearLens, f::Field{B}) where {B}
    f̃ = Lϕ * f
    function back(Δ)
        (∇' * (Ref(spin_adjoint(Ł(Δ))) .* Ł(∇*f̃))), B(Lϕ' * Δ)
    end
    f̃, back
end


# gpu

adapt_structure(storage, Lϕ::BilinearLens) = BilinearLens(adapt(storage, fieldvalues(Lϕ))...)
