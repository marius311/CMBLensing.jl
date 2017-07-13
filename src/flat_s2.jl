
# this file defines a flat-sky pixelized spin-2 map (like a polarization Q&U map)
# and operators on this map

export
    QUMap, EBMap, QUFourier, EBFourier,
    FlatS2QUMap, FlatS2EBMap, FlatS2QUFourier, FlatS2EBFourier,
    FlatS2, FlatS2QU, FlatS2Map, FlatS2Fourier

abstract type QUMap <: Basis end
abstract type EBMap <: Basis end
abstract type QUFourier <: Basis end
abstract type EBFourier <: Basis end


struct FlatS2EBMap{T<:Real,P<:Flat} <: Field{P,S2,EBMap}
    Ex::Matrix
    Bx::Matrix{T}
    FlatS2EBMap{T,P}(Ex, Bx) where {T,P} = new(checkmap(P,Ex),checkmap(P,Bx))
end


struct FlatS2EBFourier{T<:Real,P<:Flat} <: Field{P,S2,EBFourier}
    El::Matrix{Complex{T}}
    Bl::Matrix{Complex{T}}
    FlatS2EBFourier{T,P}(El, Bl) where {T,P} = new(checkfourier(P,El),checkfourier(P,Bl))
end

struct FlatS2QUMap{T<:Real,P<:Flat} <: Field{P,S2,QUMap}
    Qx::Matrix{T}
    Ux::Matrix{T}
    FlatS2QUMap{T,P}(Qx,Ux) where {T,P} = new(checkmap(P,Qx),checkmap(P,Ux))
end

struct FlatS2QUFourier{T<:Real,P<:Flat} <: Field{P,S2,QUFourier}
    Ql::Matrix{Complex{T}}
    Ul::Matrix{Complex{T}}
    FlatS2QUFourier{T,P}(Ql,Ul) where {T,P} = new(checkfourier(P,Ql),checkfourier(P,Ul))
end

const FlatS2{T,P}=Union{FlatS2EBMap{T,P},FlatS2EBFourier{T,P},FlatS2QUMap{T,P},FlatS2QUFourier{T,P}}
const FlatS2QU{T,P}=Union{FlatS2QUMap{T,P},FlatS2QUFourier{T,P}}
const FlatS2EB{T,P}=Union{FlatS2EBMap{T,P},FlatS2EBFourier{T,P}}
const FlatS2Map{T,P}=Union{FlatS2QUMap{T,P},FlatS2EBMap{T,P}}
const FlatS2Fourier{T,P}=Union{FlatS2QUFourier{T,P},FlatS2EBFourier{T,P}}

# convenience constructors
for (F,T) in [(:FlatS2EBMap,:T),(:FlatS2QUMap,:T),(:FlatS2EBFourier,:(Complex{T})),(:FlatS2QUFourier,:(Complex{T}))]
    @eval ($F){T}(a::Matrix{$T},b::Matrix{$T},Θpix=Θpix₀) = ($F){T,Flat{Θpix,size(a,2)}}(a,b)
end
FlatS2QUMap(Q::FlatS0Map{T,P},U::FlatS0Map{T,P}) where {T,P} = FlatS2QUMap{T,P}(Q[:Tx],U[:Tx])


LenseBasis{F<:FlatS2}(::Type{F}) = QUMap

QUFourier{T,P}(f::FlatS2QUMap{T,P}) = FlatS2QUFourier{T,P}(ℱ{P}*f.Qx, ℱ{P}*f.Ux)
QUFourier{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier
function QUFourier{T,P}(f::FlatS2EBFourier{T,P})
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    Ql = @. - f.El * cos2ϕ + f.Bl * sin2ϕ
    Ul = @. - f.El * sin2ϕ - f.Bl * cos2ϕ
    FlatS2QUFourier{T,P}(Ql,Ul)
end

QUMap{T,P}(f::FlatS2QUFourier{T,P}) = FlatS2QUMap{T,P}(ℱ{P}\f.Ql, ℱ{P}\f.Ul)
QUMap{T,P}(f::FlatS2EBMap{T,P}) = f |> EBFourier |> QUFourier |> QUMap
QUMap{T,P}(f::FlatS2EBFourier{T,P}) = f |> QUFourier |> QUMap

EBFourier{T,P}(f::FlatS2EBMap{T,P}) = FlatS2EBFourier{T,P}(ℱ{P}*f.Ex, ℱ{P}*f.Bx)
EBFourier{T,P}(f::FlatS2QUMap{T,P}) = f |> QUFourier |> EBFourier
function EBFourier{T,P}(f::FlatS2QUFourier{T,P})
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    El = @. - f.Ql * cos2ϕ - f.Ul * sin2ϕ
    Bl = @.   f.Ql * sin2ϕ - f.Ul * cos2ϕ
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
    (FlatS2EBMap,     FlatS2EBFourier)  => FlatS2EBFourier,
    (FlatS2QUMap,     FlatS2EBMap)      => FlatS2QUMap,
    (FlatS2QUFourier, FlatS2EBFourier)  => FlatS2QUFourier,
    (FlatS2QUMap,     FlatS2EBFourier)  => FlatS2QUMap,
    (FlatS2QUFourier, FlatS2EBMap)      => FlatS2QUFourier
)

for ((F1,F2),Tout) in rules
    @eval promote_rule{T,P}(::Type{$F1{T,P}},::Type{$F2{T,P}})=$Tout{T,P}
end

function white_noise(::Type{F}) where {Θ,Nside,T,P<:Flat{Θ,Nside},F<:FlatS2{T,P}}
    FlatS2QUMap{T,P}((randn(Nside,Nside) / FFTgrid(T,P).Δx for i=1:2)...)
end

function Cℓ_to_cov{T,P}(::Type{T}, ::Type{P}, ::Type{S2}, ℓ, CℓEE, CℓBB)
    g = FFTgrid(T,P)
    n = g.nside÷2+1
    FullDiagOp(FlatS2EBFourier{T,P}(Cℓ_2D(ℓ, CℓEE, g.r)[1:n,:], Cℓ_2D(ℓ, CℓBB, g.r)[1:n,:]))
end

function get_Cℓ(f::FlatS2{T,P}; which=(:EE,:BB), kwargs...) where {T,P}
    Cℓs = [get_Cℓ((FlatS0Fourier{T,P}(f[Symbol(x,:l)]) for x=xs)...; kwargs...) for xs in string.(which)]
    (Cℓs[1][1], hcat(last.(Cℓs)...))
end


zero(::Type{F}) where {T,P,F<:FlatS2{T,P}} = FlatS2QUMap{T,P}(@repeated(zeros(Nside(P),Nside(P)),2)...)


# dot products
dot(a::F,b::F) where {T,P,F<:FlatS2Map{T,P}} = (a[:] ⋅ b[:]) * FFTgrid(T,P).Δx^2
@generated function dot(a::F,b::F) where {T,P,F<:FlatS2Fourier{T,P}}
    F0 = FlatS0Fourier{T,P}
    fn = fieldnames(a)
    :($F0(a.$(fn[1])) ⋅ $F0(b.$(fn[1])) + $F0(a.$(fn[2])) ⋅ $F0(b.$(fn[2])))
end

# vector conversions
length{T,P}(::Type{<:FlatS2{T,P}}) = 2Nside(P)^2
@generated getindex(f::FlatS2Map,::Colon) = :(vcat($((:(f.$x[:]) for x in fieldnames(f))...)))
@generated getindex(f::FlatS2Fourier,::Colon) = :(vcat($((:(rfft2vec(f.$x)) for x in fieldnames(f))...)))
function fromvec(::Type{F}, vec::AbstractVector) where {F<:FlatS2Map}
    nside = round(Int,√(length(vec)÷2))
    F(reshape(vec[1:end÷2],(nside,nside)), reshape(vec[end÷2+1:end],(nside,nside)))
end
fromvec(::Type{F}, vec::AbstractVector) where {F<:FlatS2Fourier} = F(vec2rfft(vec[1:end÷2]), vec2rfft(vec[end÷2+1:end]))


Ac_mul_B{T,P}(a::FlatS2QUMap{T,P},b::FlatS2QUMap{T,P}) = FlatS0Map{T,P}(@. a.Qx*b.Qx+a.Ux*b.Ux)

# norms (for e.g. ODE integration error tolerance)
pixstd{T,P}(f::FlatS2Map{T,P}) = mean(@. pixstd(FlatS0Map{T,P}(getfield(f,[1,2]))))
pixstd{T,P}(f::FlatS2Fourier{T,P}) = mean(@. pixstd(FlatS0Fourier{T,P}(getfield(f,[1,2]))))

ud_grade(f::FlatS2{T,P},θnew) where {T,P} = FlatS2QUMap((ud_grade(FlatS0Map{T,P}(f[x]),θnew) for x=[:Qx,:Ux])...)

getindex(f::FlatS2{T,P},::Type{Val{:E}}) where {T,P} = FlatS0Map{T,P}(f[:Ex])
getindex(f::FlatS2{T,P},::Type{Val{:B}}) where {T,P} = FlatS0Map{T,P}(f[:Bx])
