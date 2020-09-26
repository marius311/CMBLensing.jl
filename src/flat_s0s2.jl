
### FlatS02 types
# (FlatS02's are just FieldTuple's of a spin-0 and a spin-2)
const FlatIQUMap{P,T,M}     = FieldTuple{IQUMap,     NamedTuple{(:I,:P),Tuple{FlatMap{P,T,M},    FlatQUMap{P,T,M}}},     T}
const FlatIQUFourier{P,T,M} = FieldTuple{IQUFourier, NamedTuple{(:I,:P),Tuple{FlatFourier{P,T,M},FlatQUFourier{P,T,M}}}, Complex{T}}
const FlatIEBMap{P,T,M}     = FieldTuple{IEBMap,     NamedTuple{(:I,:P),Tuple{FlatMap{P,T,M},    FlatEBMap{P,T,M}}},     T}
const FlatIEBFourier{P,T,M} = FieldTuple{IEBFourier, NamedTuple{(:I,:P),Tuple{FlatFourier{P,T,M},FlatEBFourier{P,T,M}}}, Complex{T}}
# some handy Unions
const FlatS02{P,T,M} = Union{FlatIQUMap{P,T,M},FlatIQUFourier{P,T,M},FlatIEBMap{P,T,M},FlatIEBFourier{P,T,M}}
const FlatIQU{P,T,M} = Union{FlatIQUMap{P,T,M},FlatIQUFourier{P,T,M}}
const FlatIEB{P,T,M} = Union{FlatIEBMap{P,T,M},FlatIEBFourier{P,T,M}}
const FlatS02Map{P,T,M} = Union{FlatIQUMap{P,T,M},FlatIEBMap{P,T,M}}
const FlatS02Fourier{P,T,M} = Union{FlatIQUFourier{P,T,M},FlatIEBFourier{P,T,M}}

spin(::Type{<:FlatS02}) = S02


### convenience constructors
for (F,F2,F0,(X,Y,Z),(X′,Y′,Z′),T) in [
        (:FlatIQUMap,     :FlatQUMap,     :FlatMap,     (:Ix,:Qx,:Ux), (:I,:Q,:U), :T),
        (:FlatIQUFourier, :FlatQUFourier, :FlatFourier, (:Il,:Ql,:Ul), (:I,:Q,:U), :(Complex{T})),
        (:FlatIEBMap,     :FlatEBMap,     :FlatMap,     (:Ix,:Ex,:Bx), (:I,:E,:B), :T),
        (:FlatIEBFourier, :FlatEBFourier, :FlatFourier, (:Il,:El,:Bl), (:I,:E,:B), :(Complex{T}))
    ]
    
    doc = """
        # main constructors:
        $F($X::AbstractMatrix, $Y::AbstractMatrix, $Z::AbstractMatrix[, θpix={resolution in arcmin}, ∂mode={fourier∂ or map∂})
        $F($X′::$F0, $Y′::$F0, $Z′::$F0)
        
        # more low-level:
        $F{P}($X::AbstractMatrix, $Y::AbstractMatrix, $Z::AbstractMatrix) # specify pixelization P explicilty
        $F{P,T}($X::AbstractMatrix, $Y::AbstractMatrix, $Z::AbstractMatrix) # additionally, convert elements to type $T
        $F{P,T,M<:AbstractMatrix{$T}}($X::M, $Y::M, $Z::M) # specify everything explicilty
        
    Construct a `$F` object. The top form of the constructors is most convenient
    for interactive work, while the others may be more useful for low-level code.
    """
    @eval begin
        @doc $doc $F
        $F($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array, $Z::AbstractRank2or3Array; kwargs...) =
            $F($F0($X; kwargs...), $F2($Y,$Z; kwargs...))
        $F{P}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array, $Z::AbstractRank2or3Array) where {P} =
            $F{P}($F0{P}($X), $F2{P}($Y,$Z))
        $F{P,T}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array, $Z::AbstractRank2or3Array) where {P,T} =
            $F{P,T}($F0{P,T}($X), $F2{P,T}($Y,$Z))
        $F{P,T,M}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array, $Z::AbstractRank2or3Array) where {P,T,M} =
            $F{P,T,M}($F0{P,T,M}($X), $F2{P,T,M}($Y,$Z))
        $F($X′::$F0, $Y′::$F0, $Z′::$F0) = $F($X′, $F2($Y′,$Z′))
    end
end


### properties
propertynames(f::FlatS02) = (sort([:I, :P, :IP, propertynames(f.I)..., propertynames(f.P)...])...,)
getproperty(f::FlatIQU, s::Val{:IP}) = f
getproperty(f::FlatIQU, s::Union{Val{:Q},Val{:U}}) = getproperty(getfield(f,:fs).P,s)
getproperty(f::FlatIEB, s::Union{Val{:E},Val{:B}}) = getproperty(getfield(f,:fs).P,s)
getproperty(f::FlatS02, s::Union{Val{:Qx},Val{:Ux},Val{:Ex},Val{:Bx},Val{:Ql},Val{:Ul},Val{:El},Val{:Bl}}) = getproperty(getfield(f,:fs).P,s)
getproperty(f::FlatS02, s::Union{Val{:Ix},Val{:Il}}) = getproperty(getfield(f,:fs).I,s)
function getindex(f::FlatS02, k::Symbol)
    @match k begin
        (:IP) => f
        (:I || :P) => getfield(f.fs,k)
        (:Q || :U || :E || :B) => getindex(f.P,k)
        (:Ix || :Il) => getindex(f.I,k)
        (:Qx || :Ux || :Ql || :Ul || :Ex || :Bx || :El || :Bl) => getindex(f.P,k)
        _ => throw(ArgumentError("Invalid FlatS02 index: $k"))
    end
end


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
