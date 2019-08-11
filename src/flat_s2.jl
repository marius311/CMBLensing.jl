
### FlatS2 types
# (spin-2 fields are just special FieldTuples)
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
     (Symbol(string(k,(f isa FlatMap ? "x" : "l"))) for (k,f) in pairs(f.fs))...)
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
    B = @match k begin
        (:E || :B || :Ex || :Bx) => EBMap
        (:E || :B || :El || :Bl) => EBFourier
        (:Q || :U || :Qx || :Ux) => QUMap
        (:Q || :U || :Ql || :Ul) => QUFourier
        _ => throw(ArgumentError("Invalid FlatS2 index: $k"))
    end
    getproperty(B(f),k)
end
getindex(D::DiagOp{<:FlatS2}, k::Symbol) =
    k in (:E,:B) ? Diagonal(getproperty(D.diag,k)) : throw(ArgumentError("Invalid Diagonal{:<FlatS2} index: $k"))

### basis conversion

QUFourier(f::FlatQUMap) = FlatQUFourier(Fourier(f))
QUFourier(f::FlatEBMap) = f |> EBFourier |> QUFourier
QUFourier(f::FlatEBFourier{P,T}) where {P,T} = begin
    @unpack sin2ϕ, cos2ϕ = FFTgrid(P,T)
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
    @unpack sin2ϕ, cos2ϕ = FFTgrid(P,T)
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
    FlatEBMap(E=white_noise(FlatMap{P,T}), B=white_noise(FlatMap{P,T}))
end
function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S2}, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs) where {P,T}
    Diagonal(FlatEBFourier(E=Cℓ_to_Cov(P,T,S0,CℓEE).diag, B=Cℓ_to_Cov(P,T,S0,CℓBB).diag))
end


function get_Cℓ(f::FlatS2; which=(:EE,:BB), kwargs...)
    Cℓ = [get_Cℓ(getproperty(f,Symbol(x1)),getproperty(f,Symbol(x2))) for (x1,x2) in split.(string.(ensure1d(which)),"")]
    which isa Symbol ? Cℓ[1] : Cℓ
end


function ud_grade(f::FlatS2{P}, args...; kwargs...) where {P} 
    f′ = FieldTuple(map(f->ud_grade(f, args...; kwargs...), f.fs))
    B′ = (f′[1] isa FlatMap) ? (f isa FlatQU ? QUMap : EBMap) : (f isa FlatQU ? QUFourier : EBFourier)
    FieldTuple{B′}(f′)
end
