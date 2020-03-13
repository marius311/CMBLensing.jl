
struct InterpLens{S} <: ImplicitOp{Basis,Spin,Pix}
    sparse_repr :: S
end


function InterpLens(ϕ::FlatS0)
    
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
    
    InterpLens(sparse(K, M, V))

end

adjoint(Lϕ::InterpLens) = InterpLens(adjoint(Lϕ.sparse_repr))

for op in (:*, :\)

    @eval function ($op)(Lϕ::InterpLens, f::FlatS0{P}) where {N,P<:Flat{N}}
        FlatMap{P}(reshape(($op)(Lϕ.sparse_repr, view(f[:Ix],:)), N, N))
    end
    
    @eval function ($op)(Lϕ::InterpLens, f::FieldTuple)
        Łf = Ł(f)
        F = typeof(Łf)
        F(map(f->($op)(Lϕ,f), Łf.fs))
    end

end


@adjoint InterpLens(ϕ) = InterpLens(ϕ), Δ -> (Δ,)

@adjoint function *(Lϕ::InterpLens, f::Field{B}) where {B}
    f̃ = Lϕ * f
    function back(Δ)
        (∇' * (Ref(tuple_adjoint(Ł(Δ))) .* Ł(∇*f̃))), B(Lϕ*Δ)
    end
    f̃, back
end
