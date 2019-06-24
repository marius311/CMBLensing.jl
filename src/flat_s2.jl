
# this file defines a flat-sky pixelized spin-2 map (like a polarization Q&U map)
# and operators on this map


### FlatS2 types
# spin-2 fields are just special FieldTuple's
const FlatQUMap{P,T,M}     = FieldTuple{QUMap,     NamedTuple{(:Q,:U),NTuple{2,FlatMap{P,T,M}}}, T}
const FlatQUFourier{P,T,M} = FieldTuple{QUFourier, NamedTuple{(:Q,:U),NTuple{2,FlatFourier{P,T,M}}}, Complex{T}}
const FlatEBMap{P,T,M}     = FieldTuple{EBMap,     NamedTuple{(:E,:B),NTuple{2,FlatMap{P,T,M}}}, T}
const FlatEBFourier{P,T,M} = FieldTuple{EBFourier, NamedTuple{(:E,:B),NTuple{2,FlatFourier{P,T,M}}}, Complex{T}}
# some handy Unions
const FlatS2{P,T,M}=Union{FlatEBMap{P,T,M},FlatEBFourier{P,T,M},FlatQUMap{P,T,M},FlatQUFourier{P,T,M}}
const FlatQU{P,T,M}=Union{FlatQUMap{P,T,M},FlatQUFourier{P,T,M}}
const FlatEB{P,T,M}=Union{FlatEBMap{P,T,M},FlatEBFourier{P,T,M}}
const FlatS2Map{P,T,M}=Union{FlatQUMap{P,T,M},FlatEBMap{P,T,M}}
const FlatS2Fourier{P,T,M}=Union{FlatQUFourier{P,T,M},FlatEBFourier{P,T,M}}

### convenience constructors
FlatQUMap(Qx, Ux; θpix=θpix₀, ∂mode=fourier∂) = FlatQUMap{Flat{size(Qx,2),θpix,∂mode}}(Qx, Ux)
FlatQUMap{P}(Qx::M, Ux::M) where {P,T,M<:AbstractMatrix{T}} = FlatQUMap{P,T,M}((Q=FlatMap{P,T,M}(Qx), U=FlatMap{P,T,M}(Ux)))

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

function getindex(f::FlatS2, k::Symbol)
    k in [:Ex,:Bx,:El,:Bl,:Qx,:Ux,:Ql,:Ul] || throw(ArgumentError("Invalid FlatS2 index: $k"))
    getproperty([EBMap,EBFourier,QUMap,QUFourier][in.(k, [(:Ex,:Bx),(:El,:Bl),(:Qx,:Ux),(:Ql,:Ul)])][1](f),k)
end


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


### simulation and power spectra

function white_noise(::Type{<:FlatS2{P,T}}) where {P,T}
    FlatEBMap(;(x => white_noise(FlatMap{P,T}) for x in [:E,:B])...)
end

function Cℓ_to_cov(::Type{P}, ::Type{T}, ::Type{S2}, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs; mask_nyquist=true) where {P,T}
    Diagonal(FlatEBFourier(;(x => Cℓ_to_cov(P,T,S0,Cℓ).diag for (x,Cℓ) in ((:E,CℓEE),(:B,CℓBB)))...))
end




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
