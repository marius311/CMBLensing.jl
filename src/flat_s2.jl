
# this file defines a flat-sky pixelized spin-2 map (like a polarization Q&U map)
# and operators on this map

export 
    QUMap, EBMap, QUFourier, EBFourier,
    FlatS2QUMap, FlatS2EBMap, FlatS2QUFourier, FlatS2EBFourier,
    FlatS2, FlatS2QU, FlatS2Map, FlatS2Fourier
    
abstract QUMap <: Basis
abstract EBMap <: Basis
abstract QUFourier <: Basis
abstract EBFourier <: Basis


immutable FlatS2EBMap{T<:Real,P<:Flat} <: Field{P,S2,EBMap}
    Ex::Matrix{T}
    Bx::Matrix{T}
    FlatS2EBMap(Ex, Bx) = new(checkmap(P,Ex),checkmap(P,Bx))
end


immutable FlatS2EBFourier{T<:Real,P<:Flat} <: Field{P,S2,EBFourier}
    El::Matrix{Complex{T}}
    Bl::Matrix{Complex{T}}
    FlatS2EBFourier(El, Bl) = new(checkfourier(P,El),checkfourier(P,Bl))
end

immutable FlatS2QUMap{T<:Real,P<:Flat} <: Field{P,S2,QUMap}
    Qx::Matrix{T}
    Ux::Matrix{T}
    FlatS2QUMap(Qx,Ux) = new(checkmap(P,Qx),checkmap(P,Ux))
end

immutable FlatS2QUFourier{T<:Real,P<:Flat} <: Field{P,S2,QUFourier}
    Ql::Matrix{Complex{T}}
    Ul::Matrix{Complex{T}}
    FlatS2QUFourier(Ql,Ul) = new(checkfourier(P,Ql),checkfourier(P,Ul))
end

typealias FlatS2{T,P} Union{FlatS2EBMap{T,P},FlatS2EBFourier{T,P},FlatS2QUMap{T,P},FlatS2QUFourier{T,P}}
typealias FlatS2QU{T,P} Union{FlatS2QUMap{T,P},FlatS2QUFourier{T,P}}
typealias FlatS2Map{T,P} Union{FlatS2QUMap{T,P},FlatS2EBMap{T,P}}
typealias FlatS2Fourier{T,P} Union{FlatS2QUFourier{T,P},FlatS2EBFourier{T,P}}

# convenience constructors
for (F,T) in [(:FlatS2EBMap,:T),(:FlatS2QUMap,:T),(:FlatS2EBFourier,:(Complex{T})),(:FlatS2QUFourier,:(Complex{T}))]
    @eval ($F){T}(a::Matrix{$T},b::Matrix{$T},Θpix=Θpix₀) = ($F){T,Flat{Θpix,size(a,2)}}(a,b)
end

LenseBasis{F<:FlatS2}(::Type{F}) = QUMap

QUFourier{T,P}(f::FlatS2QUMap{T,P}) = FlatS2QUFourier{T,P}(ℱ{P}*f.Qx, ℱ{P}*f.Ux)
QUFourier{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier
function QUFourier{T,P}(f::FlatS2EBFourier{T,P})
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    Ql = - f.El .*cos2ϕ + f.Bl .* sin2ϕ
    Ul = - f.El .*sin2ϕ - f.Bl .* cos2ϕ
    FlatS2QUFourier{T,P}(Ql,Ul)
end

QUMap{T,P}(f::FlatS2QUFourier{T,P}) = FlatS2QUMap{T,P}(ℱ{P}\f.Ql, ℱ{P}\f.Ul)
QUMap{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier |> QUMap
QUMap{T,P}(f::FlatS2EBFourier{T,P}) = f |> QUFourier |> QUMap

EBFourier{T,P}(f::FlatS2EBMap{T,P}) = FlatS2EBFourier{T,P}(ℱ{P}*f.Ex, ℱ{P}*f.Bx)
EBFourier{T,P}(f::FlatS2QUMap{T,P}) = f |> QUFourier |> EBFourier
function EBFourier{T,P}(f::FlatS2QUFourier{T,P}) 
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    El = - f.Ql .* cos2ϕ - f.Ul .* sin2ϕ
    Bl =   f.Ql .* sin2ϕ - f.Ul .* cos2ϕ
    FlatS2EBFourier{T,P}(El,Bl)
end

EBMap{T,P}(f::FlatS2EBFourier{T,P}) = FlatS2EBMap{T,P}(ℱ{P}\f.El, ℱ{P}\f.Bl)
EBMap{T,P}(f::FlatS2QUMap{T,P}) = f |> QUFourier |> EBFourier |> EBMap
EBMap{T,P}(f::FlatS2QUFourier{T,P}) = f |> EBFourier |> EBMap

# basically we always err on the side of keeping things in the QU and Map (could
# be further explored if this is the most optimal choice given the operations we
# tend to do)
rules = Dict(
    (FlatS2QUMap,     FlatS2QUFourier)  => FlatS2QUMap,
    (FlatS2EBMap,     FlatS2EBFourier)  => FlatS2EBMap,
    (FlatS2QUMap,     FlatS2EBMap)      => FlatS2QUMap,
    (FlatS2QUFourier, FlatS2EBFourier)  => FlatS2QUFourier,
    (FlatS2QUMap,     FlatS2EBFourier)  => FlatS2QUMap,
    (FlatS2QUFourier, FlatS2EBMap)      => FlatS2QUFourier
)

for ((F1,F2),Tout) in rules
    @eval @swappable promote_type{T,P}(::Type{$F1{T,P}},::Type{$F2{T,P}})=$Tout{T,P}
end

function white_noise{F<:FlatS2}(::Type{F})
    T,P = F.parameters #will be less hacky in 0.6
    FlatS2QUMap{T,P}((randn(Nside(P),Nside(P)) / FFTgrid(T,P).Δx for i=1:2)...)
end

function Cℓ_to_cov{T,P}(::Type{P}, ::Type{S2}, ℓ::Vector{T}, CℓEE::Vector{T}, CℓBB::Vector{T})
    g = FFTgrid(T,P)
    n = g.nside÷2+1
    LinearDiagOp(FlatS2EBFourier{T,P}(cls_to_cXXk(ℓ, CℓEE, g.r)[1:n,:], cls_to_cXXk(ℓ, CℓBB, g.r)[1:n,:]))
end


# define derivatives
∂Basis{F<:FlatS2QU}(::Type{F}) = QUFourier
function *{T,P,n}(::∂Op{:x,n}, f::FlatS2QUFourier{T,P})
    ikⁿ = (im .* FFTgrid(T,P).k).^n
    FlatS2QUFourier{T,P}(ikⁿ' .* f.Ql, ikⁿ' .* f.Ul)
end
function *{T,P,n}(::∂Op{:y,n}, f::FlatS2QUFourier{T,P})
    ikⁿ = (im .* FFTgrid(T,P).k[1:Nside(P)÷2+1]).^n
    FlatS2QUFourier{T,P}(ikⁿ .* f.Ql, ikⁿ .* f.Ul)
end

# dot products
dot{F<:FlatS2Map}(a::F,b::F) = a[:] ⋅ b[:] * FFTgrid(F.parameters...).Δx^2
function dot{F<:FlatS2Fourier}(a::F,b::F)
    F0 = FlatS0Fourier{F.parameters...}
    +(map((a,b)->F0(a)⋅F0(b),data(a),data(b))...)
end

# vector conversions
tovec(f::FlatS2) = vcat((v[:] for v in fieldvalues(f))...)
function fromvec{F<:Union{FlatS2QUMap,FlatS2EBMap}}(::Type{F}, vec::AbstractVector)
    nside = round(Int,√(length(vec)÷2))
    F(reshape(vec[1:end÷2],(nside,nside)), reshape(vec[end÷2+1:end],(nside,nside)))
end
function fromvec{F<:Union{FlatS2QUFourier,FlatS2EBFourier}}(::Type{F}, vec::AbstractVector)
    nside = round(Int,√(1+length(vec))-1)
    F(reshape(vec[1:end÷2],(nside÷2+1,nside)), reshape(vec[end÷2+1:end],(nside÷2+1,nside)))
end
length{F<:FlatS2Map}(::Type{F}) = (P=F.parameters[2]; 2Nside(P)^2) #todo: less hacky in 0.6.... 
length{F<:FlatS2Fourier}(::Type{F}) = (P=F.parameters[2]; 2Nside(P)*(Nside(P)÷2+1))



# pixel-by-pixel multiplication between a spin-0 map and a QU map, where Q&U are
# each individually multiplied
@swappable *{T,P}(a::FlatS0Map{T,P}, b::FlatS2QUMap{T,P}) = FlatS2QUMap{T,P}(a.Tx .* b.Qx, a.Tx .* b.Ux)

# needed for (df̃dϕ)ᵀ calculation, but need to think about how to really handle
# transposing given the several different spaces at play....
import Base: Ac_mul_B
Ac_mul_B{T,P}(a::FlatS2QUMap{T,P},b::FlatS2QUMap{T,P}) = FlatS0Map{T,P}(+(map(.*,data(conj.(a)),data(b))...))
