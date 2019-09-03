
### FlatS02 types
# (FlatS02's are just FieldTuple's of a spin-0 and a spin-2)
const FlatIQUMap{P,T,M}     = FieldTuple{BasisTuple{Tuple{Map,QUMap}},         NamedTuple{(:I,:P),Tuple{FlatMap{P,T,M},    FlatQUMap{P,T,M}}},     T}
const FlatIQUFourier{P,T,M} = FieldTuple{BasisTuple{Tuple{Fourier,QUFourier}}, NamedTuple{(:I,:P),Tuple{FlatFourier{P,T,M},FlatQUFourier{P,T,M}}}, Complex{T}}
const FlatIEBMap{P,T,M}     = FieldTuple{BasisTuple{Tuple{Map,EBMap}},         NamedTuple{(:I,:P),Tuple{FlatMap{P,T,M},    FlatEBMap{P,T,M}}},     T}
const FlatIEBFourier{P,T,M} = FieldTuple{BasisTuple{Tuple{Fourier,EBFourier}}, NamedTuple{(:I,:P),Tuple{FlatFourier{P,T,M},FlatEBFourier{P,T,M}}}, Complex{T}}
# some handy Unions
const FlatS02{P,T,M} = Union{FlatIQUMap{P,T,M},FlatIQUFourier{P,T,M},FlatIEBMap{P,T,M},FlatIEBFourier{P,T,M}}
const FlatIQU{P,T,M} = Union{FlatIQUMap{P,T,M},FlatIQUFourier{P,T,M}}
const FlatIEB{P,T,M} = Union{FlatIEBMap{P,T,M},FlatIEBFourier{P,T,M}}

### convenience constructors
for (F,F2,F0,(X,Y,Z),T) in [
        (:FlatIQUMap,     :FlatQUMap,     :FlatMap,     (:Ix,:Qx,:Ux), :T),
        (:FlatIQUFourier, :FlatQUFourier, :FlatFourier, (:Il,:Ql,:Ul), :(Complex{T})),
        (:FlatIEBMap,     :FlatEBMap,     :FlatMap,     (:Ix,:Ex,:Bx), :T),
        (:FlatIEBFourier, :FlatEBFourier, :FlatFourier, (:Il,:El,:Bl), :(Complex{T}))
    ]
    doc = """
        # main constructor:
        $F($X::AbstractMatrix, $Y::AbstractMatrix, $Z::AbstractMatrix[, θpix={resolution in arcmin}, ∂mode={fourier∂ or map∂})
        
        # more low-level:
        $F{P}($X::AbstractMatrix, $Y::AbstractMatrix, $Z::AbstractMatrix) # specify pixelization P explicilty
        $F{P,T}($X::AbstractMatrix, $Y::AbstractMatrix, $Z::AbstractMatrix) # additionally, convert elements to type $T
        $F{P,T,M<:AbstractMatrix{$T}}($X::M, $Y::M, $Z::M) # specify everything explicilty
        
    Construct a `$F` object. The top form of the constructor is most convenient
    for interactive work, while the others may be more useful for low-level code.
    """
    @eval begin
        @doc $doc $F
        $F($X, $Y, $Z; kwargs...) = $F($F0($X; kwargs...), $F2($Y,$Z; kwargs...))
        $F{P}($X, $Y, $Z) where {P} = $F{P}($F0{P}($X), $F2{P}($Y,$Z))
        $F{P,T}($X, $Y, $Z) where {P,T} = $F{P,T}($F0{P,T}($X), $F2{P,T}($Y,$Z))
    end
end


### properties
propertynames(f::FlatS02) = (sort([:I, :P, propertynames(f.I)..., propertynames(f.P)...])...,)
getproperty(f::FlatIQU, s::Union{Val{:Q},Val{:U}}) = getproperty(getfield(f,:fs).P,s)
getproperty(f::FlatIEB, s::Union{Val{:E},Val{:B}}) = getproperty(getfield(f,:fs).P,s)
getproperty(f::FlatS02, s::Union{Val{:Qx},Val{:Ux},Val{:Ex},Val{:Bx},Val{:Ql},Val{:Ul},Val{:El},Val{:Bl}}) = getproperty(getfield(f,:fs).P,s)
getproperty(f::FlatS02, s::Union{Val{:Ix},Val{:Il}}) = getproperty(getfield(f,:fs).I,s)
function getindex(f::FlatS02, k::Symbol)
    @match k begin
        (:IP) => f
        (:P) => f.P
        (:I) => f.I
        (:Q || :U || :E || :U) => getindex(f.P,k)
        (:Ix || :Il) => getindex(f.I,k)
        (:Qx || :Ux || :Ql || :Ul || :Ex || :Bx || :El || :Bl) => getindex(f.P,k)
        _ => throw(ArgumentError("Invalid FlatS02 index: $k"))
    end
end

#
# # a block TEB diagonal operator
# struct FlatTEBCov{T,P} <: LinOp{BasisTuple{Tuple{Fourier,EBFourier}},SpinTuple{Tuple{S0,S2}},P}
#     ΣTE :: SMatrix{2,2,Diagonal{T},4}
#     ΣB :: Matrix{T}
#     unsafe_invert :: Bool
#     FlatTEBCov{T,P}(ΣTE,ΣB,unsafe_invert=true) where {T,P} = new{T,P}(ΣTE,ΣB,unsafe_invert)
# end
# 
# # convenience constructor
# function FlatTEBCov{T,P}(ΣTT::AbstractMatrix, ΣTE::AbstractMatrix, ΣEE::AbstractMatrix, ΣBB::AbstractMatrix) where {T,P}
#     D(Σ) = Diagonal(Σ[:])
#     FlatTEBCov{T,P}(@SMatrix([D(ΣTT) D(ΣTE); D(ΣTE) D(ΣEE)]), ΣBB)
# end
# 
# # contructing from Cℓs
# function Cℓ_to_Cov(::Type{T}, ::Type{P}, ::Type{S0}, ::Type{S2}, CℓTT::InterpolatedCℓs, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs, CℓTE::InterpolatedCℓs; mask_nyquist=true) where {T,P}
#     _Mnyq = mask_nyquist ? Mnyq : identity
#     FlatTEBCov{T,P}((_Mnyq(T,P,Cℓ_2D(P, Cℓ.ℓ, Cℓ.Cℓ)) for Cℓ in (CℓTT,CℓTE,CℓEE,CℓBB))...)
# end
# 
# # applying the operator
# *(L::FlatTEBCov, f::FlatS02) =      L * BasisTuple{Tuple{Fourier,EBFourier}}(f)
# \(L::FlatTEBCov, f::FlatS02) = inv(L) * BasisTuple{Tuple{Fourier,EBFourier}}(f)
# function *(L::FlatTEBCov{T,P}, f::FlatTEBFourier{T,P}) where {T,N,P<:Flat{<:Any,N}} 
#     (t,e),b = (L.ΣTE * [@view(f.fs[1].Tl[:]), @view(f.fs[2].El[:])]), L.ΣB .* f.fs[2].Bl
#     FieldTuple(FlatS0Fourier{T,P}(reshape(t,N÷2+1,N)),FlatS2EBFourier{T,P}(reshape(e,N÷2+1,N),b))
# end
# adjoint(L::F) where {F<:FlatTEBCov} = F(L.ΣTE',L.ΣB)
# inv(L::F) where {F<:FlatTEBCov} = F((L.unsafe_invert ? (nan2zero.(inv(L.ΣTE)), nan2zero.(1 ./ L.ΣB)) : (inv(L.ΣTE), 1 ./ L.ΣB))...)
# sqrt(L::F) where {F<:FlatTEBCov} = F((L.unsafe_invert ? nan2zero.(sqrt(L.ΣTE)) : sqrt(L.ΣTE)), sqrt.(L.ΣB))
# simulate(L::FlatTEBCov{T,P}) where {T,P} = sqrt(L) * white_noise(FlatTEBFourier{T,P})
# function Diagonal(L::FlatTEBCov{T,P}) where {T,N,P<:Flat{<:Any,N}}
#     FullDiagOp(FlatTEBFourier{T,P}(reshape.(diag.([L.ΣTE[1,1], L.ΣTE[2,2]]),[(N÷2+1,N)])..., L.ΣB))
# end
# 
# 
# # multiplication by a Diag{TEB}
# function *(L::FlatTEBCov{T,P}, D::FullDiagOp{FlatTEBFourier{T,P}}) where {T,P}
#     t,e,b = Diagonal(real(D.f[:Tl][:])), Diagonal(real(D.f[:El][:])), real(D.f[:Bl])
#     FlatTEBCov{T,P}(L.ΣTE * [[t] [0]; [0] [e]], b .* L.ΣB)
# end
# *(D::FullDiagOp{FlatTEBFourier{T,P}}, L::FlatTEBCov{T,P}) where {T,P} = (adjoint(L)*D)'
# # multiplication of two FlatTEBCov
# *(La::F, Lb::F) where {F<:FlatTEBCov} = F(La.ΣTE * Lb.ΣTE, La.ΣB .* Lb.ΣB)
# # addition
# +(La::F, Lb::F) where {F<:FlatTEBCov} = F(La.ΣTE .+ Lb.ΣTE, La.ΣB .+ Lb.ΣB)
# 
# 
# # simple arithmetic with scalars
# +(s::UniformScaling{<:Scalar}, L::FlatTEBCov) = L + s
# +(L::F, s::UniformScaling{<:Scalar}) where {F<:FlatTEBCov} =
#     F([[Diagonal(diag(L.ΣTE[1,1]).+s.λ)] [L.ΣTE[1,2]]; [L.ΣTE[2,1]] [Diagonal(diag(L.ΣTE[2,2]).+s.λ)]], L.ΣB.+s.λ, L.unsafe_invert)
# *(s::Scalar, L::FlatTEBCov) = L * s
# *(L::F, s::Scalar) where {F<:FlatTEBCov} = F(L.ΣTE .* s, L.ΣB .* s, L.unsafe_invert)
# 
# function get_Cℓ(f::FlatS02{T,P}; which=(:TT,:TE,:EE,:BB), kwargs...) where {T,P}
#     [get_Cℓ((FlatS0Fourier{T,P}(f[Symbol(x,:l)]) for x=xs)...; kwargs...) for xs in string.(which)]
# end
