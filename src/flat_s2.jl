
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

spin(::Type{<:FlatS2}) = S2

### convenience constructors
for (F,F0,(X,Y),T) in [
        (:FlatQUMap,     :FlatMap,     (:Qx,:Ux), :T),
        (:FlatQUFourier, :FlatFourier, (:Ql,:Ul), :(Complex{T})),
        (:FlatEBMap,     :FlatMap,     (:Ex,:Bx), :T),
        (:FlatEBFourier, :FlatFourier, (:El,:Bl), :(Complex{T}))
    ]
    doc = """
        # main constructor:
        $F($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array[, θpix={resolution in arcmin}, ∂mode={fourier∂ or map∂})
        
        # more low-level:
        $F{P}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array) # specify pixelization P explicilty
        $F{P,T}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array) # additionally, convert elements to type $T
        $F{P,T,M<:AbstractRank2or3Array{$T}}($X::M, $Y::M) # specify everything explicilty
        
    Construct a `$F` object. The top form of the constructor is most convenient
    for interactive work, while the others may be more useful for low-level code.
    """
    @eval begin
        @doc $doc $F
        $F($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array; kwargs...) =
            $F{Flat(Nside=size($X,2),D=size($X,3);kwargs...)}($X, $Y)
        $F{P}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array) where {P} =
            $F{P}(($F0{P}($X), $F0{P}($Y)))
        $F{P,T}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array) where {P,T} =
            $F{P,T}($F0{P,T}($X), $F0{P,T}($Y))
        $F{P,T,M}($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array) where {P,T,M} =
            $F{P,T,M}($F0{P,T,M}($X), $F0{P,T,M}($Y))
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
getproperty(f::FlatS2,        ::Val{:P})  = f
function getindex(f::FlatS2, k::Symbol)
    B = @match k begin
        (:P)         => Basis
        (:E  || :B)  => f isa FlatQUMap ? EBMap : f isa FlatQUFourier ? EBFourier : Basis
        (:Q  || :U)  => f isa FlatEBMap ? QUMap : f isa FlatEBFourier ? QUFourier : Basis
        (:Ex || :Bx) => EBMap
        (:El || :Bl) => EBFourier
        (:Qx || :Ux) => QUMap
        (:Ql || :Ul) => QUFourier
        _ => throw(ArgumentError("Invalid FlatS2 index: $k"))
    end
    getproperty(B(f),k)
end

function Base.getindex(D::DiagOp{<:FlatEBFourier}, s::Symbol)
    @unpack El, Bl = diag(D);
    @unpack sin2ϕ, cos2ϕ, P = fieldinfo(diag(D))
    f = @match s begin
        :QQ          => FlatFourier{P}(@. Bl*sin2ϕ^2 + El*cos2ϕ^2)
        (:QU || :UQ) => FlatFourier{P}(@. (El - Bl) * sin2ϕ * cos2ϕ)
        :UU          => FlatFourier{P}(@. Bl*cos2ϕ^2 + El*sin2ϕ^2)
        _            => getproperty(D.diag, s)
    end
    Diagonal(f)
end

### basis conversion

QUFourier(f::FlatQUMap) = FlatQUFourier(map(Fourier,f.fs))
QUFourier(f::FlatEBMap) = f |> EBFourier |> QUFourier
QUFourier(f::FlatEBFourier{P,T}) where {P,T} = begin
    @unpack sin2ϕ, cos2ϕ = fieldinfo(f)
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
    @unpack sin2ϕ, cos2ϕ = fieldinfo(f)
    El = @. - f.Ql * cos2ϕ - f.Ul * sin2ϕ
    Bl = @.   f.Ql * sin2ϕ - f.Ul * cos2ϕ
    FlatEBFourier(E=FlatFourier{P}(El), B=FlatFourier{P}(Bl))
end

EBMap(f::FlatEBFourier) = FlatEBMap(map(Map,f.fs))
EBMap(f::FlatQUMap)     = f |> QUFourier |> EBFourier |> EBMap
EBMap(f::FlatQUFourier) = f |> EBFourier |> EBMap

QUFourier(f′::FlatQUFourier, f::FlatQUMap) = (map(Fourier,f′.fs,f.fs); f′)
QUMap(f′::FlatQUMap, f::FlatQUFourier) = (map(Map,f′.fs,f.fs); f′)

Map(f::FlatQUFourier) = QUMap(f)
Map(f::FlatEBFourier) = EBMap(f)
Map(f::FlatS2Map) = f
Fourier(f::FlatQUMap) = QUFourier(f)
Fourier(f::FlatEBMap) = EBFourier(f)
Fourier(f::FlatS2Fourier) = f


### simulation and power spectra
function white_noise(rng::AbstractRNG, ::Type{F2}) where {F2<:FlatS2}
    F = (((::Type{<:FieldTuple{B,NamedTuple{Names,NTuple{2,F}}}}) where {B,Names,F}) -> F)(F2)
    FlatEBMap(E=white_noise(rng,F), B=white_noise(rng,F))
end
function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S2}, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs; kwargs...) where {P,T,M}
    Diagonal(FlatEBFourier(E=diag(Cℓ_to_Cov(P,T,S0,CℓEE;kwargs...)), B=diag(Cℓ_to_Cov(P,T,S0,CℓBB;kwargs...))))
end
function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S2}, (CℓEE, ℓedges, θname)::Tuple, CℓBB::InterpolatedCℓs; units=fieldinfo(P).Ωpix) where {P,T}
    C₀ = Cℓ_to_Cov(P, T, S2, CℓEE, CℓBB, units=units)
    Cbins = Diagonal.(FlatEBFourier.(MidPasses(ℓedges) .* [diag(C₀).E], [zero(diag(C₀).B)]))
    BinRescaledOp(C₀,Cbins,θname)
end



function get_Cℓ(f1::FlatS2, f2::FlatS2=f1; which=(:EE,:BB), kwargs...)
    Cℓ = (;[Symbol(x1*x2) => get_Cℓ(getindex(f1,Symbol(x1)),getindex(f2,Symbol(x2)); kwargs...) for (x1,x2) in split.(string.(ensure1d(which)),"")]...)
    which isa Symbol ? Cℓ[1] : Cℓ
end


function ud_grade(f::FlatS2{P}, args...; kwargs...) where {P} 
    f′ = FieldTuple(map(f->ud_grade(f, args...; kwargs...), f.fs))
    B′ = (f′[1] isa FlatMap) ? (f isa FlatQU ? QUMap : EBMap) : (f isa FlatQU ? QUFourier : EBFourier)
    FieldTuple{B′}(f′)
end
