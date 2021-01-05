


# A Flat TEB covariance of the form:
# 
#    [ΣTT ΣTE  ⋅
#     ΣTE ΣEE  ⋅
#      ⋅   ⋅  ΣBB]
# 
# We store the 2x2 block as a 2x2 SMatrix, ΣTE, so that we can easily call sqrt/inv on
# it, and the ΣBB block separately as ΣB. 
struct FlatIEBCov{T,F} <: ImplicitOp{IEBFourier,S02,Pix}
    ΣTE :: SMatrix{2,2,Diagonal{T,F},4}
    ΣB :: Diagonal{T,F}
end

# contructing from Cℓs
function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S02}, CℓTT, CℓEE, CℓBB, CℓTE; kwargs...) where {T,P}
    ΣTT, ΣEE, ΣBB, ΣTE = [Cℓ_to_Cov(P,T,S0,Cℓ; kwargs...) for Cℓ in (CℓTT,CℓEE,CℓBB,CℓTE)]
    FlatIEBCov(@SMatrix([ΣTT ΣTE; ΣTE ΣEE]), ΣBB)
end

# applying the operator
*(L::FlatIEBCov, f::FlatS02) =       L * IEBFourier(f)
\(L::FlatIEBCov, f::FlatS02) = pinv(L) * IEBFourier(f)
function *(L::FlatIEBCov, f::FlatIEBFourier)
    (i,e),b = (L.ΣTE * [f.I, f.E]), L.ΣB * f.B
    FlatIEBFourier(i,e,b)
end

# manipulating the operator
adjoint(L::FlatIEBCov) = L
sqrt(L::FlatIEBCov) = FlatIEBCov(sqrt(L.ΣTE), sqrt(L.ΣB))
pinv(L::FlatIEBCov) = FlatIEBCov(pinv(L.ΣTE), pinv(L.ΣB))
simulate(rng::AbstractRNG, L::FlatIEBCov{Complex{T},FlatFourier{P,T,M}}) where {P,T,M} = sqrt(L) * white_noise(rng, FlatIEBFourier{P,T,M})
global_rng_for(::Type{FlatIEBCov{T,F}}) where {T,F} = global_rng_for(F)
diag(L::FlatIEBCov) = FlatIEBFourier(L.ΣTE[1,1].diag, L.ΣTE[2,2].diag, L.ΣB.diag)
similar(L::FlatIEBCov) = FlatIEBCov(similar.(L.ΣTE), similar(L.ΣB))

# getindex
function getindex(L::FlatIEBCov, k::Symbol)
    @match k begin
        :IP => L
        :I => L.ΣTE[1,1]
        :E => L.ΣTE[2,2]
        :B => L.ΣB
        :P => Diagonal(FlatEBFourier(L[:E].diag, L[:B].diag))
        (:QQ || :UU || :QU || :UQ) => getindex(L[:P], k)
        _ => throw(ArgumentError("Invalid FlatIEBCov index: $k"))
    end
end


# FlatIEBCov arithmetic
*(L::FlatIEBCov, D::DiagOp{<:FlatIEBFourier}) = FlatIEBCov(SMatrix{2,2}(L.ΣTE * [[D[:I]] [0]; [0] [D[:E]]]), L.ΣB * D[:B])
+(L::FlatIEBCov, D::DiagOp{<:FlatIEBFourier}) = FlatIEBCov(@SMatrix[L.ΣTE[1,1]+D[:I] L.ΣTE[1,2]; L.ΣTE[2,1] L.ΣTE[2,2]+D[:E]], L.ΣB + D[:B])
*(La::F, Lb::F) where {F<:FlatIEBCov} = F(La.ΣTE * Lb.ΣTE, La.ΣB * Lb.ΣB)
+(La::F, Lb::F) where {F<:FlatIEBCov} = F(La.ΣTE + Lb.ΣTE, La.ΣB + Lb.ΣB)
+(L::FlatIEBCov, U::UniformScaling{<:Scalar}) = FlatIEBCov(@SMatrix[(L.ΣTE[1,1]+U) L.ΣTE[1,2]; L.ΣTE[2,1] (L.ΣTE[2,2]+U)], L.ΣB+U)
*(L::FlatIEBCov, λ::Scalar) = FlatIEBCov(L.ΣTE * λ, L.ΣB * λ)
*(D::DiagOp{<:FlatIEBFourier}, L::FlatIEBCov) = L * D
+(U::UniformScaling{<:Scalar}, L::FlatIEBCov) = L + U
*(λ::Scalar, L::FlatIEBCov) = L * λ
copyto!(dst::Σ, src::Σ) where {Σ<:FlatIEBCov} = (copyto!(dst.ΣB, src.ΣB); copyto!.(dst.ΣTE, src.ΣTE); dst)

function get_Cℓ(f1::FlatS02, f2::FlatS02=f1; which=(:II,:EE,:BB,:IE,:IB,:EB), kwargs...)
    Cℓ = (;[Symbol(x1*x2) => get_Cℓ(getindex(f1,Symbol(x1)),getindex(f2,Symbol(x2)); kwargs...) for (x1,x2) in split.(string.(ensure1d(which)),"")]...)
    which isa Symbol ? Cℓ[1] : Cℓ
end
