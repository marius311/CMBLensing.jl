
# this file defines a flat-sky pixelized spin-2 map (like a polarization Q&U map)
# and operators on this map

export
    QUMap, EBMap, QUFourier, EBFourier,
    FlatS2QUMap, FlatS2EBMap, FlatS2QUFourier, FlatS2EBFourier,
    FlatS2QU, FlatS2EB, FlatS2Map, FlatS2Fourier, FlatS2


struct FlatS2EBMap{T<:Real,P<:Flat} <: Field{EBMap,S2,P}
    Ex::Matrix{T}
    Bx::Matrix{T}
    FlatS2EBMap{T,P}(Ex, Bx) where {T,P} = new(checkmap(P,Ex),checkmap(P,Bx))
end

struct FlatS2EBFourier{T<:Real,P<:Flat} <: Field{EBFourier,S2,P}
    El::Matrix{Complex{T}}
    Bl::Matrix{Complex{T}}
    FlatS2EBFourier{T,P}(El, Bl) where {T,P} = new(checkfourier(P,El),checkfourier(P,Bl))
end

struct FlatS2QUMap{T<:Real,P<:Flat} <: Field{QUMap,S2,P}
    Qx::Matrix{T}
    Ux::Matrix{T}
    FlatS2QUMap{T,P}(Qx,Ux) where {T,P} = new(checkmap(P,Qx),checkmap(P,Ux))
end

struct FlatS2QUFourier{T<:Real,P<:Flat} <: Field{QUFourier,S2,P}
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
    @eval ($F)(a::Matrix{$T},b::Matrix{$T},Θpix=Θpix₀,∂mode=fourier∂) where {T} = ($F){T,Flat{Θpix,size(a,2),∂mode}}(a,b)
end
FlatS2QUMap(Q::FlatS0Map{T,P},U::FlatS0Map{T,P}) where {T,P} = FlatS2QUMap{T,P}(Q[:Tx],U[:Tx])


LenseBasis(::Type{<:FlatS2}) = QUMap

QUFourier(f::FlatS2QUMap{T,P})     where {T,P} = FlatS2QUFourier{T,P}(ℱ{P}*f.Qx, ℱ{P}*f.Ux)
QUFourier(f::FlatS2EBMap{T,P})     where {T,P} = f |> EBFourier |> QUFourier
QUFourier(f::FlatS2EBFourier{T,P}) where {T,P} = begin
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    Ql = @. - f.El * cos2ϕ + f.Bl * sin2ϕ
    Ul = @. - f.El * sin2ϕ - f.Bl * cos2ϕ
    FlatS2QUFourier{T,P}(Ql,Ul)
end

QUMap(f::FlatS2QUFourier{T,P}) where {T,P} = FlatS2QUMap{T,P}(ℱ{P}\f.Ql, ℱ{P}\f.Ul)
QUMap(f::FlatS2EBMap{T,P})     where {T,P} = f |> EBFourier |> QUFourier |> QUMap
QUMap(f::FlatS2EBFourier{T,P}) where {T,P} = f |> QUFourier |> QUMap

EBFourier(f::FlatS2EBMap{T,P})     where {T,P} = FlatS2EBFourier{T,P}(ℱ{P}*f.Ex, ℱ{P}*f.Bx)
EBFourier(f::FlatS2QUMap{T,P})     where {T,P} = f |> QUFourier |> EBFourier
EBFourier(f::FlatS2QUFourier{T,P}) where {T,P} = begin
    sin2ϕ, cos2ϕ = FFTgrid(T,P).sincos2ϕ
    El = @. - f.Ql * cos2ϕ - f.Ul * sin2ϕ
    Bl = @.   f.Ql * sin2ϕ - f.Ul * cos2ϕ
    FlatS2EBFourier{T,P}(El,Bl)
end

EBMap(f::FlatS2EBFourier{T,P}) where {T,P} = FlatS2EBMap{T,P}(ℱ{P}\f.El, ℱ{P}\f.Bl)
EBMap(f::FlatS2QUMap{T,P})     where {T,P} = f |> QUFourier |> EBFourier |> EBMap
EBMap(f::FlatS2QUFourier{T,P}) where {T,P} = f |> EBFourier |> EBMap

function QUFourier(f′::FlatS2QUFourier{T,P}, f::FlatS2QUMap{T,P}) where {T,P}
    mul!(f′.Ql, FFTgrid(T,P).FFT, f.Qx)
    mul!(f′.Ul, FFTgrid(T,P).FFT, f.Ux)
    f′
end

function QUMap(f′::FlatS2QUMap{T,P}, f::FlatS2QUFourier{T,P}) where {T,P}
    ldiv!(f′.Qx, FFTgrid(T,P).FFT, f.Ql)
    ldiv!(f′.Ux, FFTgrid(T,P).FFT, f.Ul)
    f′
end



function white_noise(::Type{F}) where {Θ,Nside,T,P<:Flat{Θ,Nside},F<:FlatS2{T,P}}
    FlatS2QUMap{T,P}((randn(Nside,Nside) / FFTgrid(T,P).Δx for i=1:2)...)
end

function Cℓ_to_cov(::Type{T}, ::Type{P}, ::Type{S2}, ℓ, CℓEE, CℓBB; mask_nyquist=true) where {T,θ,N,P<:Flat{θ,N}}
    _Mnyq = mask_nyquist ? Mnyq : identity
    FullDiagOp(FlatS2EBFourier{T,P}((_Mnyq(T,P,Cℓ_2D(P,ℓ, Cℓ)[1:(N÷2+1),:]) for Cℓ in (CℓEE,CℓBB))...))
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
length(::Type{<:FlatS2{T,P}}) where {T,P} = 2Nside(P)^2
@generated getindex(f::FlatS2Map,::Colon) = :(vcat($((:(f.$x[:]) for x in fieldnames(f))...)))
@generated getindex(f::FlatS2Fourier,::Colon) = :(vcat($((:(rfft2vec(f.$x)) for x in fieldnames(f))...)))
function fromvec(::Type{F}, vec::AbstractVector) where {F<:FlatS2Map}
    nside = round(Int,√(length(vec)÷2))
    F(reshape(vec[1:end÷2],(nside,nside)), reshape(vec[end÷2+1:end],(nside,nside)))
end
fromvec(::Type{F}, vec::AbstractVector) where {F<:FlatS2Fourier} = F(vec2rfft(vec[1:end÷2]), vec2rfft(vec[end÷2+1:end]))


Ac_mul_B(a::FlatS2QUMap{T,P},b::FlatS2QUMap{T,P}) where {T,P} = FlatS0Map{T,P}(@. a.Qx*b.Qx+a.Ux*b.Ux)

ud_grade(f::FlatS2{T,P}, args...; kwargs...) where {T,P} = FlatS2QUMap((Map(ud_grade(f[x],args...;kwargs...)) for x=[:Q,:U])...)

getindex(f::FlatS2{T,P},::Type{Val{:E}}) where {T,P} = FlatS0Map{T,P}(f[:Ex])
getindex(f::FlatS2{T,P},::Type{Val{:B}}) where {T,P} = FlatS0Map{T,P}(f[:Bx])
getindex(f::FlatS2{T,P},::Type{Val{:Q}}) where {T,P} = FlatS0Map{T,P}(f[:Qx])
getindex(f::FlatS2{T,P},::Type{Val{:U}}) where {T,P} = FlatS0Map{T,P}(f[:Ux])
getindex(op::FullDiagOp{FlatS2EBFourier{T,P}},::Type{Val{:E}}) where {T,P} = FullDiagOp(FlatS0Fourier{T,P}(op.f.El))
getindex(op::FullDiagOp{FlatS2EBFourier{T,P}},::Type{Val{:B}}) where {T,P} = FullDiagOp(FlatS0Fourier{T,P}(op.f.Bl))
