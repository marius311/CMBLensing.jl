
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
for (F,F0,(X,Y),T) in [
        (:FlatQUMap,     :FlatMap,     (:Qx,:Ux), :T),
        (:FlatQUFourier, :FlatFourier, (:Ql,:Ul), :(Complex{T})),
        (:FlatEBMap,     :FlatMap,     (:Ex,:Bx), :T),
        (:FlatEBFourier, :FlatFourier, (:El,:Bl), :(Complex{T}))
    ]
    doc = """
        # main constructor:
        $F($X::AbstractMatrix, $Y::AbstractMatrix[, θpix={resolution in arcmin}, ∂mode={fourier∂ or map∂})
        
        # more low-level:
        $F{P}($X::AbstractMatrix, $Y::AbstractMatrix) # specify pixelization P explicilty
        $F{P,T}($X::AbstractMatrix, $Y::AbstractMatrix) # additionally, convert elements to type $T
        $F{P,T,M<:AbstractMatrix{$T}}($X::M, $Y::M) # specify everything explicilty
        
    Construct a `$F` object. The top form of the constructor is most convenient
    for interactive work, while the others may be more useful for low-level code.
    """
    @eval begin
        @doc $doc $F
        $F($X::AbstractMatrix, $Y::AbstractMatrix; kwargs...) = $F{Flat(Nside=size($X,2);kwargs...)}($X, $Y)
        $F{P}($X::AbstractMatrix, $Y::AbstractMatrix) where {P} = $F{P}($F0{P}($X), $F0{P}($Y))
        $F{P,T}($X::AbstractMatrix, $Y::AbstractMatrix) where {P,T} = $F{P,T}($F0{P,T}($X), $F0{P,T}($Y))
    end
end

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
        (:Ex || :Bx) => EBMap
        (:El || :Bl) => EBFourier
        (:Qx || :Ux) => QUMap
        (:Ql || :Ul) => QUFourier
        (:E  || :B)  => basis(f)==QUMap ? EBMap : basis(f)==QUFourier ? EBFourier : Basis
        (:Q  || :U)  => basis(f)==EBMap ? QUMap : basis(f)==EBFourier ? QUFourier : Basis
        _ => throw(ArgumentError("Invalid FlatS2 index: $k"))
    end
    getproperty(B(f),k)
end
getindex(D::DiagOp{<:FlatS2}, k::Symbol) =
    k in (:E,:B) ? Diagonal(getproperty(D.diag,k)) : throw(ArgumentError("Invalid Diagonal{:<FlatS2} index: $k"))

### basis conversion

QUFourier(f::FlatQUMap) = FlatQUFourier(map(Fourier,f.fs))
QUFourier(f::FlatEBMap) = f |> EBFourier |> QUFourier
QUFourier(f::FlatEBFourier{P,T}) where {P,T} = begin
    @unpack sin2ϕ, cos2ϕ = FFTgrid(P,T)
    Ql = @. - f.El * cos2ϕ + f.Bl * sin2ϕ
    Ul = @. - f.El * sin2ϕ - f.Bl * cos2ϕ
    FlatQUFourier(Q=FlatFourier{P}(Ql), U=FlatFourier{P}(Ul))
end

QUMap(f::FlatQUFourier)  = FlatQUMap(map(Map,f.fs))
QUMap(f::FlatEBMap)      = f |> EBFourier |> QUFourier |> QUMap
QUMap(f::FlatEBFourier)  = f |> QUFourier |> QUMap

EBFourier(f::FlatEBMap) = FlatEBFourier(map(Fourier,f.fs))
EBFourier(f::FlatQUMap) = f |> QUFourier |> EBFourier
EBFourier(f::FlatQUFourier{P,T}) where {P,T} = begin
    @unpack sin2ϕ, cos2ϕ = FFTgrid(P,T)
    El = @. - f.Ql * cos2ϕ - f.Ul * sin2ϕ
    Bl = @.   f.Ql * sin2ϕ - f.Ul * cos2ϕ
    FlatEBFourier(E=FlatFourier{P}(El), B=FlatFourier{P}(Bl))
end

EBMap(f::FlatEBFourier) = FlatEBMap(map(Map,f.fs))
EBMap(f::FlatQUMap)     = f |> QUFourier |> EBFourier |> EBMap
EBMap(f::FlatQUFourier) = f |> EBFourier |> EBMap

QUFourier(f′::FlatQUFourier, f::FlatQUMap) = (map(Fourier,f′.fs,f.fs); f′)
QUMap(f′::FlatQUMap, f::FlatQUFourier) = (map(Map,f′.fs,f.fs); f′)

Map(f::FlatQU) = QUMap(f)
Map(f::FlatEB) = EBMap(f)
Fourier(f::FlatQU) = QUFourier(f)
Fourier(f::FlatEB) = EBFourier(f)


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
