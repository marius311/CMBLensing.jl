
### FlatField types
# these are all just appropriately-parameterized BaseFields 
# note: the seemingly-redundant <:AbstractArray{T} in the argument
# (which is enforced in BaseField anyway) is there to help prevent
# method ambiguities

# spin-0
const FlatMap{        M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{Map,        M, T, A}
const FlatFourier{    M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{Fourier,    M, T, A}
# spin-2
const FlatQUMap{      M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{QUMap,      M, T, A}
const FlatQUFourier{  M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{QUFourier,  M, T, A}
const FlatEBMap{      M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{EBMap,      M, T, A}
const FlatEBFourier{  M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{EBFourier,  M, T, A}
# spin-(0,2)
const FlatIQUMap{     M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{IQUMap,     M, T, A}
const FlatIQUFourier{ M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{IQUFourier, M, T, A}
const FlatIEBMap{     M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{IEBMap,     M, T, A}
const FlatIEBFourier{ M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{IEBFourier, M, T, A}

## FlatField unions
# spin-0
const FlatS0{         B<:Union{Map,Fourier},                         M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
# spin-2
const FlatQU{         B<:Union{QUMap,QUFourier},                     M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatEB{         B<:Union{EBMap,EBFourier},                     M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatS2Map{      B<:Union{QUMap,EBMap},                         M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatS2Fourier{  B<:Union{QUFourier,QUFourier},                 M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatS2{         B<:Union{QUMap,QUFourier,EBMap,EBFourier},     M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
# spin-(0,2)
const FlatIQU{        B<:Union{IQUMap,IQUFourier},                   M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatIEB{        B<:Union{IEBMap,IEBFourier},                   M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatS02Map{     B<:Union{IQUMap,IEBMap},                       M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatS02Fourier{ B<:Union{IQUFourier,IQUFourier},               M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
const FlatS02{        B<:Union{IQUMap,IQUFourier,IEBMap,IEBFourier}, M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}
# any flat projection
const FlatField{      B,                                             M<:FlatProj, T, A<:AbstractArray{T} } = BaseField{B, M, T, A}


# A Flat TEB covariance of the form:
# 
#    [ΣTT ΣTE  ⋅
#     ΣTE ΣEE  ⋅
#      ⋅   ⋅  ΣBB]
# 
# We store the 2x2 block as a 2x2 SMatrix, ΣTE, so that we can easily call sqrt/inv on
# it, and the ΣBB block separately as ΣB. 
struct FlatIEBCov{T,F} <: ImplicitOp{T}
    ΣTE :: SMatrix{2,2,Diagonal{T,F},4}
    ΣB :: Diagonal{T,F}
end


### basis-like definitions
LenseBasis(::Type{<:FlatS0})    = Map
LenseBasis(::Type{<:FlatS2})    = QUMap
LenseBasis(::Type{<:FlatS02})   = IQUMap
DerivBasis(::Type{<:FlatS0})    = Fourier
DerivBasis(::Type{<:FlatS2})    = QUFourier
DerivBasis(::Type{<:FlatS02})   = IQUFourier
HarmonicBasis(::Type{<:FlatS0}) = Fourier
HarmonicBasis(::Type{<:FlatQU}) = QUFourier
HarmonicBasis(::Type{<:FlatEB}) = EBFourier


### constructors
# the default constructor uses ProjLambert, but the code in this file
# should be agnostic to the projection type

_reshape_batch(arr::AbstractArray{T,3}) where {T} = reshape(arr, size(arr,1), size(arr,2), 1, size(arr,3))
_reshape_batch(arr) = arr

## constructing from arrays
# spin-0
function FlatMap(Ix::A; kwargs...) where {T, A<:AbstractArray{T}}
    FlatMap(_reshape_batch(Ix), ProjLambert(;Ny=size(Ix,1), Nx=size(Ix,2), T, storage=basetype(A), kwargs...))
end
function FlatFourier(Il::A; Ny, kwargs...) where {T, A<:AbstractArray{T}}
    FlatFourier(_reshape_batch(Il), ProjLambert(;Ny, Nx=size(Il,2), T, storage=basetype(A), kwargs...))
end
# spin-2
function FlatField{B}(X::A, Y::A, metadata)  where {T, A<:AbstractArray{T}, B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁}}}
    FlatField{B}(cat(_reshape_batch(X), _reshape_batch(Y), dims=Val(3)), metadata)
end
function FlatField{B}(X::A, Y::A; kwargs...) where {T, A<:AbstractArray{T}, B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁}}}
    FlatField{B}(X, Y, ProjLambert(;Ny=size(X,1), Nx=size(X,2), T, storage=basetype(A), kwargs...))
end
# spin-(0,2)
function FlatField{B}(X::A, Y::A, Z::A, metadata) where {T, A<:AbstractArray{T}, B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁}}}
    FlatField{B}(cat(_reshape_batch(X), _reshape_batch(Y), _reshape_batch(Z), dims=Val(3)), metadata)
end
function FlatField{B}(X::A, Y::A, Z::A; kwargs...) where {T, A<:AbstractArray{T}, B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁}}}
    FlatField{B}(X, Y, Z, ProjLambert(;Ny=size(X,1), Nx=size(X,2), T, storage=basetype(A), kwargs...))
end
## constructing from other fields
function FlatField{B}(X::FlatField{B₀}, Y::FlatField{B₀}) where {B₀<:Union{Map,Fourier}, B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁},B₀}}
    FlatField{B}(cat(X.arr, Y.arr, dims=Val(3)), get_metadata_strict(X, Y))
end
function FlatField{B}(X::FlatField{B₀}, Y::FlatField{Basis2Prod{Pol,B₀}}) where {B₀<:Union{Map,Fourier}, Pol<:Union{𝐐𝐔,𝐄𝐁}, B<:Basis3Prod{𝐈,Pol,B₀}}
    FlatField{B}(cat(X.arr, Y.arr, dims=Val(3)), get_metadata_strict(X, Y))
end
function FlatField{B}(X::FlatField{B₀}, Y::FlatField{B₀}, Z::FlatField{B₀}) where {B₀<:Union{Map,Fourier}, B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁},B₀}}
    FlatField{B}(cat(X.arr, Y.arr, Z.arr, dims=Val(3)), get_metadata_strict(X, Y, Z))
end


# todo: doc strings

### array interface
# most is inherited from BaseField. the main thing we have specify
# here has to do with which dimension is the "batch" dimension
# (dimension 4), since that is not assumed in BaseField
similar(f::FlatField{B}, Nbatch::Int) where {B} = FlatField{B}(similar(f.arr, size(f.arr,1), size(f.arr,2), size(f.arr,3), Nbatch), f.metadata)
batch_axes(f::FlatField{B,M,T,A}) where {B,M,T,A<:AbstractArray{T,4}} = (f.Nbatch,)
batch_axes(f::FlatField{B,M,T,A}) where {B,M,T,A<:AbstractArray{T}} = ()
nonbatch_dims(f::FlatField) = ntuple(identity,min(3,ndims(f.arr)))
require_unbatched(f::FlatField) = (f.Nbatch==1) || error("This function not implemented for batched fields.")

### properties
# generic
getproperty(f::FlatField, ::Val{:Nbatch}) = size(getfield(f,:arr), 4)
getproperty(f::FlatField, ::Val{:Npol})   = size(getfield(f,:arr), 3)
getproperty(f::FlatField, ::Val{:T})      = eltype(f)
# sub-components
for (F, keys) in [
    (FlatMap,        ("Ix"=>:, "I"=>:)),
    (FlatFourier,    ("Il"=>:, "I"=>:)),
    (FlatQUMap,      ("Qx"=>1, "Ux"=>2, "Q" =>1, "U"=>2, "P"=>:)),
    (FlatQUFourier,  ("Ql"=>1, "Ul"=>2, "Q" =>1, "U"=>2, "P"=>:)),
    (FlatEBMap,      ("Ex"=>1, "Bx"=>2, "E" =>1, "B"=>2, "P"=>:)),
    (FlatEBFourier,  ("El"=>1, "Bl"=>2, "E" =>1, "B"=>2, "P"=>:)),
    (FlatIQUMap,     ("Ix"=>1, "Qx"=>2, "Ux"=>3, "I"=>1, "Q"=>2, "U"=>3, "P"=>2:3, "IP"=>:)),
    (FlatIQUFourier, ("Il"=>1, "Ql"=>2, "Ul"=>3, "I"=>1, "Q"=>2, "U"=>3, "P"=>2:3, "IP"=>:)),
    (FlatIEBMap,     ("Ix"=>1, "Ex"=>2, "Bx"=>3, "I"=>1, "E"=>2, "B"=>3, "P"=>2:3, "IP"=>:)),
    (FlatIEBFourier, ("Il"=>1, "El"=>2, "Bl"=>3, "I"=>1, "E"=>2, "B"=>3, "P"=>2:3, "IP"=>:)),
]
    for (k,I) in keys
        body = if k[end] in "xl"
            I==(:) ? :(getfield(f,:arr)) : :(view(getfield(f,:arr), :, :, $I, ntuple(_->:,max(0,N-3))...))
        else
            I==(:) ? :f : :($(FlatField{k=="P" ? Basis2Prod{basis(F).parameters[end-1:end]...} : basis(F).parameters[end]})(view(getfield(f,:arr), :, :, $I, ntuple(_->:,max(0,N-3))...), f.metadata))
        end
        @eval getproperty(f::$F{M,T,A}, ::Val{$(QuoteNode(Symbol(k)))}) where {M<:FlatProj,T,N,A<:AbstractArray{T,N}} = $body
    end
end


### indices
function getindex(f::FlatS0, k::Symbol; full_plane=false)
    maybe_unfold = full_plane ? x->unfold(x,fieldinfo(f).Ny) : identity
    @match k begin
        :I  => f
        :Ix => Map(f).Ix
        :Il => maybe_unfold(Fourier(f).Il)
        _   => throw(ArgumentError("Invalid FlatS0 index: $k"))
    end
end
function getindex(f::FlatS2{Basis2Prod{B₁,B₂}}, k::Symbol; full_plane=false) where {B₁,B₂}
    maybe_unfold = (full_plane && k in [:El,:Bl,:Ql,:Ul]) ? x->unfold(x,fieldinfo(f).Ny) : identity
    B = @match k begin
        (:P)         => identity
        (:E  || :B)  => Basis2Prod{𝐄𝐁,B₂}
        (:Q  || :U)  => Basis2Prod{𝐐𝐔,B₂}
        (:Ex || :Bx) => Basis2Prod{𝐄𝐁,Map}
        (:El || :Bl) => Basis2Prod{𝐄𝐁,Fourier}
        (:Qx || :Ux) => Basis2Prod{𝐐𝐔,Map}
        (:Ql || :Ul) => Basis2Prod{𝐐𝐔,Fourier}
        _ => throw(ArgumentError("Invalid FlatS2 index: $k"))
    end
    maybe_unfold(getproperty(B(f),k))
end
function getindex(f::FlatS02{Basis3Prod{B₁,B₂,B₃}}, k::Symbol; full_plane=false) where {B₁,B₂,B₃}
    maybe_unfold = (full_plane && k in [:Il,:El,:Bl,:Ql,:Ul]) ? x->unfold(x,fieldinfo(f).Ny) : identity
    B = @match k begin
        (:I  || :P)  => identity
        (:E  || :B)  => Basis3Prod{𝐈,𝐄𝐁,B₃}
        (:Q  || :U)  => Basis3Prod{𝐈,𝐐𝐔,B₃}
        (:Ix)        => Basis3Prod{𝐈,B₂,Map}
        (:Il)        => Basis3Prod{𝐈,B₂,Fourier}
        (:Ex || :Bx) => Basis3Prod{𝐈,𝐄𝐁,Map}
        (:El || :Bl) => Basis3Prod{𝐈,𝐄𝐁,Fourier}
        (:Qx || :Ux) => Basis3Prod{𝐈,𝐐𝐔,Map}
        (:Ql || :Ul) => Basis3Prod{𝐈,𝐐𝐔,Fourier}
        _ => throw(ArgumentError("Invalid FlatS02 index: $k"))
    end
    maybe_unfold(getproperty(B(f),k))
end
function getindex(D::DiagOp{<:FlatEBFourier}, k::Symbol)
    @unpack El, Bl, metadata = diag(D)
    @unpack sin2ϕ, cos2ϕ = fieldinfo(diag(D))
    f = @match k begin
        (:QQ)        => FlatFourier((@. Bl*sin2ϕ^2 + El*cos2ϕ^2),   metadata)
        (:QU || :UQ) => FlatFourier((@. (El - Bl) * sin2ϕ * cos2ϕ), metadata)
        (:UU)        => FlatFourier((@. Bl*cos2ϕ^2 + El*sin2ϕ^2),   metadata)
        _            => getproperty(D.diag, k)
    end
    Diagonal(f)
end
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




### basis conversion
## spin-0
Fourier(f::FlatMap) = FlatFourier(m_rfft(f.arr, (1,2)), f.metadata)
Fourier(f′::FlatFourier, f::FlatMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)
Map(f::FlatFourier) = FlatMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
Map(f′::FlatMap, f::FlatFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)

## spin-2
QUFourier(f::FlatQUMap) = FlatQUFourier(m_rfft(f.arr, (1,2)), f.metadata)
QUFourier(f::FlatEBMap) = f |> EBFourier |> QUFourier
QUFourier(f::FlatEBFourier) = begin
    @unpack El, Bl, sin2ϕ, cos2ϕ = fieldinfo(f)
    Ql = @. - El * cos2ϕ + Bl * sin2ϕ
    Ul = @. - El * sin2ϕ - Bl * cos2ϕ
    FlatQUFourier(Ql, Ul, f.metadata)
end

QUMap(f::FlatQUFourier) = FlatQUMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
QUMap(f::FlatEBMap)      = f |> EBFourier |> QUFourier |> QUMap
QUMap(f::FlatEBFourier)  = f |> QUFourier |> QUMap

EBFourier(f::FlatEBMap) = FlatEBFourier(m_rfft(f.arr, (1,2)), f.metadata)
EBFourier(f::FlatQUMap) = f |> QUFourier |> EBFourier
EBFourier(f::FlatQUFourier) = begin
    @unpack Ql, Ul, sin2ϕ, cos2ϕ = fieldinfo(f)
    El = @. - Ql * cos2ϕ - Ul * sin2ϕ
    Bl = @.   Ql * sin2ϕ - Ul * cos2ϕ
    FlatEBFourier(El, Bl, f.metadata)
end

EBMap(f::FlatEBFourier) = FlatEBMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
EBMap(f::FlatQUMap)     = f |> QUFourier |> EBFourier |> EBMap
EBMap(f::FlatQUFourier) = f |> EBFourier |> EBMap

# in-place
QUMap(f′::FlatQUMap, f::FlatQUFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)
QUFourier(f′::FlatQUFourier, f::FlatQUMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)

## spin-(0,2)
IQUFourier(f::FlatIQUMap) = FlatIQUFourier(m_rfft(f.arr, (1,2)), f.metadata)
IQUFourier(f::FlatIEBMap) = f |> IEBFourier |> IQUFourier
IQUFourier(f::FlatIEBFourier) = FlatIQUFourier(f.I, QUFourier(f.P))

IQUMap(f::FlatIQUFourier) = FlatIQUMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
IQUMap(f::FlatIEBMap)      = f |> IEBFourier |> IQUFourier |> IQUMap
IQUMap(f::FlatIEBFourier)  = f |> IQUFourier |> IQUMap

IEBFourier(f::FlatIEBMap) = FlatIEBFourier(m_rfft(f.arr, (1,2)), f.metadata)
IEBFourier(f::FlatIQUMap) = f |> IQUFourier |> IEBFourier
IEBFourier(f::FlatIQUFourier) = FlatIEBFourier(f.I, EBFourier(f.P))

IEBMap(f::FlatIEBFourier) = FlatIEBMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
IEBMap(f::FlatIQUMap)     = f |> IQUFourier |> IEBFourier |> IEBMap
IEBMap(f::FlatIQUFourier) = f |> IEBFourier |> IEBMap

# in-place
IQUMap(f′::FlatIQUMap, f::FlatIQUFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)
IQUFourier(f′::FlatIQUFourier, f::FlatIQUMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)

# spin-0 bases applied to spin-2 and spin-(0,2)
Fourier(f::FlatField{B}) where {B<:BasisProd} = Fourier(B)(f)
Map(f::FlatField{B}) where {B<:BasisProd} = Map(B)(f)



### pretty printing
typealias_def(::Type{F}) where {B,M,T,A,F<:FlatField{B,M,T,A}} = "Flat$(typealias(B)){$(typealias(A)),$(typealias(M))}"
function Base.summary(io::IO, f::FlatField)
    @unpack Nx,Ny,Nbatch,θpix = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel $(θpix)′-resolution ")
    Base.showarg(io, f, true)
end



### dot products
# do in Map space (the LenseBasis, Ł) for simplicity
function dot(a::FlatField, b::FlatField)
    z = Ł(a) .* Ł(b)
    batch(sum_dropdims(z.arr, dims=nonbatch_dims(z)))
end

### logdets

function logdet(L::Diagonal{<:Union{Real,Complex},<:FlatField{B}}) where {B<:Union{Fourier,Basis2Prod{<:Any,Fourier},Basis3Prod{<:Any,<:Any,Fourier}}}
    # half the Fourier plane needs to be counted twice since the real
    # FFT only stores half of it
    @unpack Ny, arr = L.diag
    λ = adapt(typeof(arr), rfft_degeneracy_fac(Ny))
    # note: since our maps are required to be real, the logdet of any
    # operator which preserves this property is also guaranteed to be
    # real, hence the `real` and `abs` below are valid
    batch(real.(sum_dropdims(nan2zero.(log.(abs.(arr)) .* λ), dims=nonbatch_dims(L.diag))))
end

function logdet(L::Diagonal{<:Real,<:FlatField{B}}) where {B<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}}}
    batch(
        sum_dropdims(log.(abs.(L.diag.arr)), dims=nonbatch_dims(L.diag)) 
        .+ dropdims(log.(prod(sign.(L.diag.arr), dims=nonbatch_dims(L.diag))), dims=nonbatch_dims(L.diag))
    )
end


### traces

function tr(L::Diagonal{<:Union{Real,Complex},<:FlatField{B}}) where {B<:Union{Fourier,Basis2Prod{<:Any,Fourier},Basis3Prod{<:Any,<:Any,Fourier}}}
    @unpack Ny, Nx, arr = L.diag
    λ = adapt(typeof(arr), rfft_degeneracy_fac(Ny))
    # the `real` is ok bc the imaginary parts of the half-plane which
    # is stored would cancel with those from the other half-plane
    batch(real.(sum_dropdims(arr .* λ, dims=nonbatch_dims(L.diag))))
end

function tr(L::Diagonal{<:Real,<:FlatField{B}}) where {B<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}}}
    batch(sum_dropdims(L.diag.arr, dims=nonbatch_dims(L.diag)))
end


### simulation
_white_noise(ξ::FlatField, rng::AbstractRNG) = 
    (randn!(similar(ξ.arr, real(eltype(ξ)), ξ.Ny, size(ξ.arr)[2:end]...)), ξ.metadata)
white_noise(ξ::FlatS0,  rng::AbstractRNG) = FlatMap(_white_noise(ξ,rng)...)
white_noise(ξ::FlatS2,  rng::AbstractRNG) = FlatEBMap(_white_noise(ξ,rng)...)
white_noise(ξ::FlatS02, rng::AbstractRNG) = FlatIEBMap(_white_noise(ξ,rng)...)


### creating covariance operators
# fixed covariances
Cℓ_to_Cov(pol::Symbol, args...; kwargs...) = Cℓ_to_Cov(Val(pol), args...; kwargs...)
function Cℓ_to_Cov(::Val{:I}, proj::ProjLambert, Cℓ::InterpolatedCℓs; units=proj.Ωpix)
    Diagonal(FlatFourier(Cℓ_to_2D(Cℓ,proj), proj) / units)
end
function Cℓ_to_Cov(::Val{:P}, proj::ProjLambert, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs; units=proj.Ωpix)
    Diagonal(FlatEBFourier(Cℓ_to_2D(CℓEE,proj), Cℓ_to_2D(CℓBB,proj), proj) / units)
end
function Cℓ_to_Cov(::Val{:IP}, proj::ProjLambert, CℓTT, CℓEE, CℓBB, CℓTE; kwargs...)
    ΣTT, ΣEE, ΣBB, ΣTE = [Cℓ_to_Cov(:I,proj,Cℓ; kwargs...) for Cℓ in (CℓTT,CℓEE,CℓBB,CℓTE)]
    FlatIEBCov(@SMatrix([ΣTT ΣTE; ΣTE ΣEE]), ΣBB)
end
# ParamDependentOp covariances scaled by amplitudes in different ℓ-bins
function Cℓ_to_Cov(::Val{:I}, proj::ProjLambert{T}, (Cℓ, ℓedges, θname)::Tuple; kwargs...) where {T}
    # we need an @eval here since we want to dynamically select a
    # keyword argument name, θname. the @eval happens into Main rather
    # than CMBLensing as a workaround for
    # https://discourse.julialang.org/t/closure-not-shipping-to-remote-workers-except-from-main/38831
    C₀ = diag(Cℓ_to_Cov(:I, proj, Cℓ; kwargs...))
    @eval Main let ℓedges=$((T.(ℓedges))...,), C₀=$C₀
        ParamDependentOp(function (;$θname=ones($T,length(ℓedges)-1),_...)
            _A = $preprocess.(Ref((nothing,C₀.metadata)), $θname)
            Diagonal(FlatFourier($bandpower_rescale(ℓedges, C₀.ℓmag, C₀.arr, _A...), C₀.metadata))
        end)
    end
end
function Cℓ_to_Cov(::Val{:P}, proj::ProjLambert{T}, (CℓEE, ℓedges, θname)::Tuple, CℓBB::InterpolatedCℓs; kwargs...) where {T}
    C₀ = diag(Cℓ_to_Cov(:P, proj, CℓEE, CℓBB; kwargs...))
    @eval Main let ℓedges=$((T.(ℓedges))...,), C₀=$C₀
        ParamDependentOp(function (;$θname=ones($T,length(ℓedges)-1),_...)
            _E = $preprocess.(Ref((nothing,C₀.metadata)),      $θname)
            _B = $preprocess.(Ref((nothing,C₀.metadata)), one.($θname))
            Diagonal(FlatEBFourier($bandpower_rescale(ℓedges, C₀.ℓmag, C₀.El, _E...), C₀.Bl .* _B[1], C₀.metadata))
        end)
    end
end
# cant reliably get Zygote's gradients to work through these
# broadcasts, which on GPU use ForwardDiff, so write the adjoint by
# hand for now. likely more performant, in any case. 
function bandpower_rescale(ℓedges, ℓ, Cℓ, A...)
    length(A)==length(ℓedges)-1 || error("Expected $(length(ℓedges)-1) bandpower parameters, not $(length(A)).")
    broadcast(ℓ, Cℓ, A...) do ℓ, Cℓ, A...
        for i=1:length(ℓedges)-1
            (ℓedges[i] < ℓ < ℓedges[i+1]) && return A[i] * Cℓ
        end
        return Cℓ
    end
end
@adjoint function bandpower_rescale(ℓedges, ℓ, Cℓ, A...)
    function back(Δ)
        Ā = map(1:length(A)) do i
            sum_dropdims(
                broadcast(Δ, ℓ, Cℓ) do Δ, ℓ, Cℓ
                    (ℓedges[i] < ℓ < ℓedges[i+1]) ? Cℓ*Δ : zero(Cℓ)
                end,
                dims = ntuple(identity, Val(ndims(Cℓ)))
            )
        end
        (nothing, nothing, nothing, Ā...)
    end
    bandpower_rescale(ℓedges, ℓ, Cℓ, A...), back
end





### spin adjoints
function *(a::SpinAdjoint{F}, b::F) where {B<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}},F<:FlatField{B}}
    FlatMap(dropdims(sum(a.f.arr .* b.arr, dims=3), dims=((a.f.Nbatch>1 || b.Nbatch>1) ? 3 : ())), get_metadata_strict(a, b))
end
function mul!(dst::FlatMap, a::SpinAdjoint{F}, b::F) where {F<:FlatField{<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}}}}
    copyto!(dst.arr, sum(a.f.arr .* b.arr, dims=3))
    dst
end


### FlatIEBCov
# applying
*(L::FlatIEBCov, f::FlatS02) =       L * IEBFourier(f)
\(L::FlatIEBCov, f::FlatS02) = pinv(L) * IEBFourier(f)
function *(L::FlatIEBCov, f::FlatIEBFourier)
    (i,e),b = (L.ΣTE * [f.I, f.E]), L.ΣB * f.B
    FlatIEBFourier(i,e,b)
end
# manipulating
size(L::FlatIEBCov) = 3 .* size(L.ΣB)
adjoint(L::FlatIEBCov) = L
sqrt(L::FlatIEBCov) = FlatIEBCov(sqrt(L.ΣTE), sqrt(L.ΣB))
pinv(L::FlatIEBCov) = FlatIEBCov(pinv(L.ΣTE), pinv(L.ΣB))
global_rng_for(::Type{FlatIEBCov{T,F}}) where {T,F} = global_rng_for(F)
diag(L::FlatIEBCov) = FlatIEBFourier(L.ΣTE[1,1].diag, L.ΣTE[2,2].diag, L.ΣB.diag)
similar(L::FlatIEBCov) = FlatIEBCov(similar.(L.ΣTE), similar(L.ΣB))
get_storage(L::FlatIEBCov) = get_storage(L.ΣB)
simulate(rng::AbstractRNG, L::FlatIEBCov; Nbatch=nothing) = 
    sqrt(L) * white_noise(similar(diag(L), (isnothing(Nbatch) || Nbatch==1 ? () : (Nbatch,))...), rng)
# arithmetic
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


### batching

batch_length(f::FlatField) = f.Nbatch

"""
    batch(fs::FlatField...)
    batch(fs::Vector{<:FlatField})
    batch(fs::TUple{<:FlatField})

Concatenate one of more FlatFields along the "batch" dimension
(dimension 4 of the underlying array). For the inverse operation, see
[`unbatch`](@ref). 
"""
batch(fs::FlatField{B}...) where {B} = 
    FlatField{B}(cat(getfield.(fs,:arr)..., dims=Val(4)), only(unique(getfield.(fs,:metadata))))

"""
    unbatch(f::FlatField)

Return an Array of FlatFields corresponding to each batch index. For
the inverse operation, see [`batch`](@ref).
"""
unbatch(f::FlatField{B}) where {B} = [f[!,i] for i=1:batch_length(f)]

batch_index(f::FlatField{B}, I) where {B<:Union{Map,Fourier}} = FlatField{B}(f.arr[:,:,1,I], f.metadata)
batch_index(f::FlatField{B}, I) where {B} = FlatField{B}(f.arr[:,:,:,I], f.metadata)


###

make_mask(f::FlatField; kwargs...) = make_mask((f.Ny,f.Nx), f.θpix; kwargs...)



### power spectra

function get_Cℓ(f₁::FlatS0, f₂::FlatS0=f₁; Δℓ=50, ℓedges=0:Δℓ:16000, Cℓfid=ℓ->1, err_estimate=false)
    @unpack Nx, Ny, Δx, ℓmag = fieldinfo(f₁)
    ℓmag = unfold(ℓmag, Ny)
    α = Nx*Ny/Δx^2

    # faster to excise unused parts:
    ℓmask = (ℓmag .> minimum(ℓedges)) .&  (ℓmag .< maximum(ℓedges))
    L = ℓmag[ℓmask]
    CLobs = 1/α .* real.(dot.(
        adapt(Array{Float64},f₁)[:Il, full_plane=true][ℓmask], 
        adapt(Array{Float64},f₂)[:Il, full_plane=true][ℓmask]
    ))
    w = @. nan2zero((2*Cℓfid(L)^2/(2L+1))^-1)
    
    sum_in_ℓbins(x) = fit(Histogram, L, Weights(x), ℓedges).weights

    local A, Cℓ, ℓ, N, Cℓ²
    Threads.@sync begin
        Threads.@spawn A  = sum_in_ℓbins(w)
        Threads.@spawn Cℓ = sum_in_ℓbins(w .* CLobs)
        Threads.@spawn ℓ  = sum_in_ℓbins(w .* L)
        if err_estimate
            Threads.@spawn N   = sum_in_ℓbins(one.(w)) / 2
            Threads.@spawn Cℓ² = sum_in_ℓbins(w .* CLobs.^2)
        end
    end

    if err_estimate
        σℓ  = sqrt.((Cℓ² ./ A .- Cℓ.^2) ./ N)
        InterpolatedCℓs(ℓ./A,  Cℓ./A .± σℓ)
    else
        InterpolatedCℓs(ℓ./A,  Cℓ./A)
    end
end

function get_Cℓ(f1::FlatS2, f2::FlatS2=f1; which=(:EE,:BB), kwargs...)
    Cℓ = (;[Symbol(x1*x2) => get_Cℓ(getindex(f1,Symbol(x1)),getindex(f2,Symbol(x2)); kwargs...) for (x1,x2) in split.(string.(ensure1d(which)),"")]...)
    which isa Symbol ? Cℓ[1] : Cℓ
end

function get_Cℓ(f1::FlatS02, f2::FlatS02=f1; which=(:II,:EE,:BB,:IE,:IB,:EB), kwargs...)
    Cℓ = (;[Symbol(x1*x2) => get_Cℓ(getindex(f1,Symbol(x1)),getindex(f2,Symbol(x2)); kwargs...) for (x1,x2) in split.(string.(ensure1d(which)),"")]...)
    which isa Symbol ? Cℓ[1] : Cℓ
end




"""
    ud_grade(f::Field, θnew, mode=:map, deconv_pixwin=true, anti_aliasing=true)

Up- or down-grades field `f` to new resolution `θnew` (only in integer steps).
Two modes are available specified by the `mode` argument: 

* `:map`     — Up/downgrade by replicating/averaging pixels in map-space
* `:fourier` — Up/downgrade by extending/truncating the Fourier grid

For `:map` mode, two additional options are possible. If `deconv_pixwin` is
true, deconvolves the pixel window function from the downgraded map so the
spectrum of the new and old maps are the same. If `anti_aliasing` is true,
filters out frequencies above Nyquist prior to down-sampling. 

"""
function ud_grade(
    f :: FlatField{B},
    θnew;
    mode = :map,
    deconv_pixwin = (mode==:map),
    anti_aliasing = (mode==:map)
) where {B}

    N,θ,T = (f.Ny, f.Nx), f.θpix, f.T
    θnew==θ && return f
    (mode in [:map,:fourier]) || throw(ArgumentError("Available modes: [:map,:fourier]"))

    fac = θnew > θ ? θnew÷θ : θ÷θnew
    (round(Int, fac) ≈ fac) || throw(ArgumentError("Can only ud_grade in integer steps"))
    fac = round(Int, fac)
    Nnew = @. round(Int, N * θ ÷ θnew)
    proj = ProjLambert(;Ny=Nnew[1], Nx=Nnew[2], θpix=θnew, T=real(T), f.storage)
    @unpack Δx,ℓx,ℓy,Nx,Ny,nyquist = proj

    PWF = deconv_pixwin ? Diagonal(FlatFourier((@. T((pixwin(θnew,ℓy)*pixwin(θnew,ℓx)')/(pixwin(θ,ℓy)*pixwin(θ,ℓx)'))), proj)) : I

    if θnew > θ
        # downgrade
        AA = anti_aliasing ? Diagonal(FlatFourier(ifelse.((abs.(f.ℓy) .>= nyquist) .| (abs.(f.ℓx') .>= nyquist), 0, 1), f.metadata)) : I
        if mode == :map
            PWF \ FlatField{Map(B)}(dropdims(mean(reshape(Map(AA*f).arr, fac, Ny, fac, Nx, size.(Ref(f.arr),nonbatch_dims(f)[3:end])...), dims=(1,3)), dims=(1,3)), proj)
        else
            error("Not implemented")
            # FlatFourier{Pnew}((AA*f)[:Il][1:(Nnew÷2+1), [1:(isodd(Nnew) ? Nnew÷2+1 : Nnew÷2); (end-Nnew÷2+1):end]])
        end
    else
        error("Not implemented")
        # # upgrade
        # @assert fieldinfo(f).Nside isa Int "Upgrading resolution only implemented for maps where `Nside isa Int`"
        # if mode==:map
        #     fnew = FlatMap{Pnew}(permutedims(hvcat(N,(x->fill(x,(fac,fac))).(f[:Ix])...)))
        #     deconv_pixwin ? FlatFourier{Pnew}(fnew[:Il] .* PWF' .* PWF[1:Nnew÷2+1]) : fnew
        # else
        #     fnew = FlatFourier{Pnew}(zeros(Nnew÷2+1,Nnew))
        #     setindex!.(Ref(fnew.Il), f[:Il], 1:(N÷2+1), [findfirst(fieldinfo(fnew).k .≈ fieldinfo(f).k[i]) for i=1:N]')
        #     deconv_pixwin ? fnew * fac^2 : fnew
        # end

    end
end