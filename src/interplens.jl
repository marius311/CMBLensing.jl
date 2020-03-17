
@doc doc"""

    InterpLens(ϕ)
    
InterpLens is a lensing operator that computes lensing with bilinear
interpolation. The action of the operator, as well as its adjoint, inverse,
inverse-adjoint, and gradient w.r.t. ϕ can all be computed. The log-determinant
of the operation is non-zero and can't be computed. 

Internally, InterpLens forms a SparseMatrix with the interpolation weights,
which can be applied and adjoint-ed extremely fast (e.g. at least an order of
magnitude faster than LenseFlow). Inverse and inverse-adjoint lensing is
somewhat slower as it is implemented with several steps of the [preconditioned
Biconjugate gradient stabilized
method](https://juliamath.github.io/IterativeSolvers.jl/dev/linear_systems/bicgstabl/),
taking anti-lensing as the
preconditioner.

"""
mutable struct InterpLens{Φ,S} <: ImplicitOp{Basis,Spin,Pix}
    ϕ :: Φ
    sparse_repr :: S
    anti_lensing_sparse_repr :: Union{S, Nothing}
end


function InterpLens(ϕ::FlatS0)
    
    if iszero(ϕ)
        return InterpLens(ϕ,I,I)
    end
    
    @unpack Nside,Δx,T = fieldinfo(ϕ)
    
    # the (i,j)-th pixel is deflected to (ĩ,j̃)
    j̃,ĩ = Map.((∇*ϕ)./Δx)
    ĩ.Ix .+= (1:Nside)
    j̃.Ix .+= (1:Nside)'
    
    indexwrap(i) = mod(i - 1, Nside) + 1
    
    K = collect(flatten(repeated.(1:Nside^2,4)))
    M,V = similar(K,Float32), similar(K,Float32)
    
    for I in eachindex(ĩ)
        
        let ĩ=ĩ[I], j̃=j̃[I]
            
            # (i,j) indices of the 4 nearest neighbors
            left_right = floor(Int,ĩ) .+ (0, 1)
            top_bottom = floor(Int,j̃) .+ (0, 1)
            
            # 1-D indices of the 4 nearest neighbors
            M[4I-3:4I] = [Base._sub2ind((Nside,Nside),indexwrap(i),indexwrap(j)) for i=left_right, j=top_bottom]
            
            # weights of these neighbors in the bilinear interpolation
            Δx⁻, Δx⁺ = (left_right .- ĩ)
            Δy⁻, Δy⁺ = (top_bottom .- j̃)
            A = @SMatrix[
                1 Δx⁻ Δy⁻ Δx⁻*Δy⁻;
                1 Δx⁺ Δy⁻ Δx⁺*Δy⁻;
                1 Δx⁻ Δy⁺ Δx⁻*Δy⁺;
                1 Δx⁺ Δy⁺ Δx⁺*Δy⁺
            ]
            # todo: I think there's a faster way than inverting the whole matrix
            # but need to work it out
            V[4I-3:4I] = inv(A)[1,:]
            
        end
        
    end
    
    InterpLens(ϕ, sparse(K, M, V), nothing)

end


# lazily computing the sparse representation for anti-lensing

function get_anti_lensing_sparse_repr!(Lϕ::InterpLens)
    if Lϕ.anti_lensing_sparse_repr == nothing
        Lϕ.anti_lensing_sparse_repr = InterpLens(-Lϕ.ϕ).sparse_repr
    end
    Lϕ.anti_lensing_sparse_repr
end


# applying various forms of the operator

function *(Lϕ::InterpLens, f::FlatS0{P}) where {N,P<:Flat{N}}
    FlatMap{P}(reshape(Lϕ.sparse_repr * view(f[:Ix],:), N, N))
end

function *(Lϕ::Adjoint{<:Any,<:InterpLens}, f::FlatS0{P}) where {N,P<:Flat{N}}
    FlatMap{P}(reshape(parent(Lϕ).sparse_repr' * view(f[:Ix],:), N, N))
end

function \(Lϕ::InterpLens, f::FlatS0{P}) where {N,P<:Flat{N}}
    FlatMap{P}(reshape(bicgstabl(
        get_anti_lensing_sparse_repr!(Lϕ) * Lϕ.sparse_repr, 
        get_anti_lensing_sparse_repr!(Lϕ) * view(f[:Ix],:),
        max_mv_products = 3
    ), N, N))
end

function \(Lϕ::Adjoint{<:Any,<:InterpLens}, f::FlatS0{P}) where {N,P<:Flat{N}}
    FlatMap{P}(reshape(bicgstabl(
        get_anti_lensing_sparse_repr!(parent(Lϕ))' * parent(Lϕ).sparse_repr', 
        get_anti_lensing_sparse_repr!(parent(Lϕ))' * view(f[:Ix],:),
        max_mv_products = 3
    ), N, N))
end

# special cases for InterpLens(0ϕ), which don't work with bicgstabl, 
# see https://github.com/JuliaMath/IterativeSolvers.jl/issues/271
function \(Lϕ::InterpLens{<:Any,<:UniformScaling}, f::FlatS0{P}) where {N,P<:Flat{N}}
    Lϕ.sparse_repr \ f
end
function \(Lϕ::Adjoint{<:Any,<:InterpLens{<:Any,<:UniformScaling}}, f::FlatS0{P}) where {N,P<:Flat{N}}
    parent(Lϕ).sparse_repr \ f
end

for op in (:*, :\)
    @eval function ($op)(Lϕ::Union{InterpLens, Adjoint{<:Any,<:InterpLens}}, f::FieldTuple)
        Łf = Ł(f)
        F = typeof(Łf)
        F(map(f->($op)(Lϕ,f), Łf.fs))
    end
end


# gradients

@adjoint InterpLens(ϕ) = InterpLens(ϕ), Δ -> (Δ,)

@adjoint function *(Lϕ::InterpLens, f::Field{B}) where {B}
    f̃ = Lϕ * f
    function back(Δ)
        (∇' * (Ref(tuple_adjoint(Ł(Δ))) .* Ł(∇*f̃))), B(Lϕ*Δ)
    end
    f̃, back
end


# gpu

adapt_structure(storage, Lϕ::InterpLens) = InterpLens(adapt(storage, fieldvalues(Lϕ))...)
