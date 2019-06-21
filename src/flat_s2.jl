
# this file defines a flat-sky pixelized spin-2 map (like a polarization Q&U map)
# and operators on this map

export
    QUMap, EBMap, QUFourier, EBFourier,
    FlatS2QUMap, FlatS2EBMap, FlatS2QUFourier, FlatS2EBFourier,
    FlatS2QU, FlatS2EB, FlatS2Map, FlatS2Fourier, FlatS2

## constructors
const FlatQUMap{P,T,M}     = FieldTuple{QUMap,     NamedTuple{(:Q,:U),NTuple{2,FlatMap{P,T,M}}}}
const FlatQUFourier{P,T,M} = FieldTuple{QUFourier, NamedTuple{(:Q,:U),NTuple{2,FlatFourier{P,T,M}}}}
const FlatEBMap{P,T,M}     = FieldTuple{EBMap,     NamedTuple{(:E,:B),NTuple{2,FlatMap{P,T,M}}}}
const FlatEBFourier{P,T,M} = FieldTuple{EBFourier, NamedTuple{(:E,:B),NTuple{2,FlatFourier{P,T,M}}}}

FlatQUMap(Qx, Ux; θpix=θpix₀, ∂mode=fourier∂) = FlatQUMap{Flat{size(Qx,2),θpix,∂mode}}(Qx, Ux)
FlatQUMap{P}(Qx::M, Ux::M) where {P,T,M<:AbstractMatrix{T}} = FlatQUMap{P,T,M}((Q=FlatMap{P,T,M}(Qx), U=FlatMap{P,T,M}(Ux)))




const FlatS2{P,T,M}=Union{FlatEBMap{P,T,M},FlatEBFourier{P,T,M},FlatQUMap{P,T,M},FlatQUFourier{P,T,M}}
const FlatQU{P,T,M}=Union{FlatQUMap{P,T,M},FlatQUFourier{P,T,M}}
const FlatEB{P,T,M}=Union{FlatEBMap{P,T,M},FlatEBFourier{P,T,M}}
const FlatS2Map{P,T,M}=Union{FlatQUMap{P,T,M},FlatEBMap{P,T,M}}
const FlatS2Fourier{P,T,M}=Union{FlatQUFourier{P,T,M},FlatEBFourier{P,T,M}}

# # convenience constructors
# for (F,T) in [(:FlatS2EBMap,:T),(:FlatS2QUMap,:T),(:FlatS2EBFourier,:(Complex{T})),(:FlatS2QUFourier,:(Complex{T}))]
#     @eval ($F)(a::Matrix{$T},b::Matrix{$T},θpix=θpix₀,∂mode=fourier∂) where {T} = ($F){T,Flat{θpix,size(a,2),∂mode}}(a,b)
# end
# FlatS2QUMap(Q::FlatS0Map{T,P},U::FlatS0Map{T,P}) where {T,P} = FlatS2QUMap{T,P}(Q.Tx, U.Tx)
# 

### properties
function propertynames(f::FlatS2)
    (:fs, propertynames(f.fs)..., 
     (Symbol(string(k,(f isa FlatMap ? "x" : "l"))) for (k,f) in pairs(f.fs) if f isa FlatMap)...)
end
getproperty(f::FlatQUMap,     ::Val{:Qx}) = getfield(f,:fs).Q.Ix
getproperty(f::FlatQUMap,     ::Val{:Ux}) = getfield(f,:fs).U.Ix
getproperty(f::FlatQUFourier, ::Val{:Ql}) = getfield(f,:fs).Q.Il
getproperty(f::FlatQUFourier, ::Val{:Ul}) = getfield(f,:fs).U.Il
getproperty(f::FlatEBMap,     ::Val{:Ex}) = getfield(f,:fs).E.Ix
getproperty(f::FlatEBMap,     ::Val{:Bx}) = getfield(f,:fs).B.Ix
getproperty(f::FlatEBFourier, ::Val{:El}) = getfield(f,:fs).E.Il
getproperty(f::FlatEBFourier, ::Val{:Bl}) = getfield(f,:fs).B.Il




### conversions

QUFourier(f::FlatQUMap) = FlatQUFourier(Fourier(f))
QUFourier(f::FlatEBMap) = f |> EBFourier |> QUFourier
QUFourier(f::FlatEBFourier{P,T}) where {P,T} = begin
    sin2ϕ, cos2ϕ = FFTgrid(P,T).sincos2ϕ
    Ql = @. - f.El * cos2ϕ + f.Bl * sin2ϕ
    Ul = @. - f.El * sin2ϕ - f.Bl * cos2ϕ
    FlatQUFourier(Q=FlatFourier{P}(Ql), U=FlatFourier{P}(Ul))
end

QUMap(f::FlatQUFourier)  = FlatQUMap(Map(f))
QUMap(f::FlatEBMap)      = f |> EBFourier |> QUFourier |> QUMap
QUMap(f::FlatEBFourier)  = f |> QUFourier |> QUMap

EBFourier(f::FlatEBMap) = FlatEBFourier(Fourier(f))
EBFourier(f::FlatQUMap) = f |> QUFourier |> EBFourier
EBFourier(f::FlatQUFourier{P,T}) where {P,T} = begin
    sin2ϕ, cos2ϕ = FFTgrid(P,T).sincos2ϕ
    El = @. - f.Ql * cos2ϕ - f.Ul * sin2ϕ
    Bl = @.   f.Ql * sin2ϕ - f.Ul * cos2ϕ
    FlatEBFourier(E=FlatFourier{P}(El), B=FlatFourier{P}(Bl))
end

EBMap(f::FlatEBFourier) = FlatEBMap(Map(f))
EBMap(f::FlatQUMap)     = f |> QUFourier |> EBFourier |> EBMap
EBMap(f::FlatQUFourier) = f |> EBFourier |> EBMap

QUFourier(f′::FlatQUFourier, f::FlatQUMap) = (map(Fourier,f′.fs,f.fs); f′)
QUMap(f′::FlatQUMap, f::FlatQUFourier) = (map(Map,f′.fs,f.fs); f′)

# 
# 
# 
# function white_noise(::Type{F}) where {Θ,Nside,T,P<:Flat{Θ,Nside},F<:FlatS2{T,P}}
#     FlatS2QUMap{T,P}((randn(Nside,Nside) / FFTgrid(T,P).Δx for i=1:2)...)
# end
# 
# function Cℓ_to_cov(::Type{T}, ::Type{P}, ::Type{S2}, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs; mask_nyquist=true) where {T,θ,N,P<:Flat{θ,N}}
#     _Mnyq = mask_nyquist ? Mnyq : identity
#     FullDiagOp(FlatS2EBFourier{T,P}((_Mnyq(T,P,Cℓ_2D(P, Cℓ.ℓ, Cℓ.Cℓ)[1:(N÷2+1),:]) for Cℓ in (CℓEE,CℓBB))...))
# end
# 
# function get_Cℓ(f::FlatS2{T,P}; which=(:EE,:BB), kwargs...) where {T,P}
#     [get_Cℓ((FlatS0Fourier{T,P}(f[Symbol(x,:l)]) for x=xs)...; kwargs...) for xs in string.(which)]
# end
# 
# 
# zero(::Type{F}) where {T,P,F<:FlatS2{T,P}} = FlatS2QUMap{T,P}(@repeated(zeros(Nside(P),Nside(P)),2)...)
# 
# 
# # dot products
# dot(a::F,b::F) where {T,P,F<:FlatS2Map{T,P}} = (a[:] ⋅ b[:]) * FFTgrid(T,P).Δx^2
# @generated function dot(a::F,b::F) where {T,P,F<:FlatS2Fourier{T,P}}
#     F0 = FlatS0Fourier{T,P}
#     fn = fieldnames(a)
#     :($F0(a.$(fn[1])) ⋅ $F0(b.$(fn[1])) + $F0(a.$(fn[2])) ⋅ $F0(b.$(fn[2])))
# end
# 
# # vector conversions
# length(::Type{<:FlatS2{T,P}}) where {T,P} = 2Nside(P)^2
# @generated getindex(f::FlatS2Map,::Colon) = :(vcat($((:(f.$x[:]) for x in fieldnames(f))...)))
# @generated getindex(f::FlatS2Fourier,::Colon) = :(vcat($((:(rfft2vec(f.$x)) for x in fieldnames(f))...)))
# function fromvec(::Type{F}, vec::AbstractVector) where {F<:FlatS2Map}
#     nside = round(Int,√(length(vec)÷2))
#     F(reshape(vec[1:end÷2],(nside,nside)), reshape(vec[end÷2+1:end],(nside,nside)))
# end
# fromvec(::Type{F}, vec::AbstractVector) where {F<:FlatS2Fourier} = F(vec2rfft(vec[1:end÷2]), vec2rfft(vec[end÷2+1:end]))
# 
# 
# Ac_mul_B(a::FlatS2QUMap{T,P},b::FlatS2QUMap{T,P}) where {T,P} = FlatS0Map{T,P}(@. a.Qx*b.Qx+a.Ux*b.Ux)
# 
# ud_grade(f::FlatS2{T,P}, args...; kwargs...) where {T,P} = FlatS2QUMap((Map(ud_grade(f[x],args...;kwargs...)) for x=[:Q,:U])...)
# 
# getproperty(f::FlatS2{T,P},::Val{:E}) where {T,P} = FlatS0Map{T,P}(f.Ex)
# getproperty(f::FlatS2{T,P},::Val{:B}) where {T,P} = FlatS0Map{T,P}(f.Bx)
# getproperty(f::FlatS2{T,P},::Val{:Q}) where {T,P} = FlatS0Map{T,P}(f.Qx)
# getproperty(f::FlatS2{T,P},::Val{:U}) where {T,P} = FlatS0Map{T,P}(f.Ux)
# 
# getindex(op::FullDiagOp, s::Symbol) = getindex(op, Val(s))
# getindex(op::FullDiagOp{FlatS2EBFourier{T,P}},::Val{:E}) where {T,P} = FullDiagOp(FlatS0Fourier{T,P}(op.f.El))
# getindex(op::FullDiagOp{FlatS2EBFourier{T,P}},::Val{:B}) where {T,P} = FullDiagOp(FlatS0Fourier{T,P}(op.f.Bl))
