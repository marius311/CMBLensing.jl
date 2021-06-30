
# We first define a "Flat" projection where the metric in the
# resulting cartesian space is assumed to be "flat" / "euclidean" and
# convolution operators are diagonalized by the 2D FFT. This is also
# known as the "flat-sky approximation". However, we don't actually
# define any functionality for it, this is mostly a placeholder for
# future extensions, and also provides backwards compatibility with
# previous use of FlatMap, etc.. in CMBLensing.jl. The only such
# projection we have implemented is ProjLambert, and everything in
# this file is actually defined on LambertFields, not FlatFields.
# However, with some thought, most of what is in here could be
# generalized to other fields.
abstract type FlatProj <: CartesianProj end
make_field_aliases("Flat", FlatProj)
default_proj(::Type{F}) where {F<:BaseField{<:Any,<:FlatProj}} = ProjLambert


# A Lambert azimuthal equal-area projection
# 
# The `rotator` field is an argument passed to `healpy.Rotator`
# specifying a rotation which rotates the north pole to the
# center of the desired field. 
# 
struct ProjLambert{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}} <: FlatProj
    # these must be the same to broadcast together
    Ny        :: Int
    Nx        :: Int
    θpix      :: Float64
    rotator   :: NTuple{3,Float64}
    # these can be different and still broadcast (including different types)
    storage
    Δx        :: T
    Ωpix      :: T
    nyquist   :: T
    Δℓx       :: T
    Δℓy       :: T
    ℓy        :: V
    ℓx        :: V
    ℓmag      :: M
    sin2ϕ     :: M
    cos2ϕ     :: M
end

real_type(T) = promote_type(real(T), Float32)
@init @require Unitful="1986cc42-f94f-5a68-af5c-568840ba703d" real_type(::Type{<:Unitful.Quantity{T}}) where {T} = real_type(T)

ProjLambert(;Ny, Nx, θpix=1, rotator=(0,90,0), T=Float32, storage=Array) = 
    ProjLambert(Ny, Nx, Float64(θpix), Float64.(rotator), real_type(T), storage)

@memoize function ProjLambert(Ny, Nx, θpix, rotator, ::Type{T}, storage) where {T}

    Δx           = T(deg2rad(θpix/60))
    Δℓx          = T(2π/(Nx*Δx))
    Δℓy          = T(2π/(Ny*Δx))
    nyquist      = T(2π/(2Δx))
    Ωpix         = T(Δx^2)
    ℓy           = adapt(storage, (ifftshift(-Ny÷2:(Ny-1)÷2) .* Δℓy)[1:Ny÷2+1])
    ℓx           = adapt(storage, (ifftshift(-Nx÷2:(Nx-1)÷2) .* Δℓx))
    ℓmag         = @. sqrt(ℓx'^2 + ℓy^2)
    ϕ            = @. angle(ℓx' + im*ℓy)
    sin2ϕ, cos2ϕ = @. sin(2ϕ), cos(2ϕ)
    if iseven(Ny)
        sin2ϕ[end, end:-1:(Nx÷2+2)] .= sin2ϕ[end, 2:Nx÷2]
    end

    ProjLambert(Ny,Nx,θpix,rotator,storage,Δx,Ωpix,nyquist,Δℓx,Δℓy,ℓy,ℓx,ℓmag,sin2ϕ,cos2ϕ)
    
end

# make LambertMap, LambertFourier, etc... type aliases
make_field_aliases("Lambert", ProjLambert)

# for printing
typealias_def(::Type{F}) where {B,M<:ProjLambert,T,A,F<:LambertField{B,M,T,A}} = "Lambert$(typealias(B)){$(typealias(A))}"
function Base.summary(io::IO, f::LambertField)
    @unpack Nx,Ny,Nbatch,θpix = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel $(θpix)′-resolution ")
    Base.showarg(io, f, true)
end

### promotion

# used in broadcasting to decide the resulting metadata when
# broadcasting over two fields
function promote_metadata_strict(metadata₁::ProjLambert{T₁}, metadata₂::ProjLambert{T₂} ) where {T₁,T₂}

    if (
        metadata₁.θpix === metadata₂.θpix &&
        metadata₁.Ny   === metadata₂.Ny   &&
        metadata₁.Nx   === metadata₂.Nx      
    )
        
        # always returning the "wider" metadata even if T₁==T₂ helps
        # inference and is optimized away anyway
        promote_type(T₁,T₂) == T₁ ? metadata₁ : metadata₂
        
    else
        error("""Can't broadcast two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:θpix,:Ny,:Nx)))
        2: $(select(fields(metadata₂),(:θpix,:Ny,:Nx)))
        """)
    end

end


# used in non-broadcasted algebra to decide the resulting metadata
# when performing some operation across two fields. this is free to do
# more generic promotion than promote_metadata_strict (although this
# is currently not used, but in the future could include promoting
# resolution, etc...). the result should be a common metadata which we
# can convert both fields to then do a succesful broadcast
promote_metadata_generic(metadata₁::ProjLambert, metadata₂::ProjLambert) = 
    promote_metadata_strict(metadata₁, metadata₂)


### preprocessing
# defines how ImplicitFields and BatchedReals behave when broadcasted
# with ProjLambert fields. these can return arrays, but can also
# return `Broadcasted` objects which are spliced into the final
# broadcast, thus avoiding allocating any temporary arrays.

function preprocess((_,proj)::Tuple{<:Any,<:ProjLambert{T,V}}, r::Real) where {T,V}
    r isa BatchedReal ? adapt(V, reshape(r.vals, 1, 1, 1, :)) : r
end
# need custom adjoint here bc Δ can come back batched from the
# backward pass even though r was not batched on the forward pass
@adjoint function preprocess(m::Tuple{<:Any,<:ProjLambert{T,V}}, r::Real) where {T,V}
    preprocess(m, r), Δ -> (nothing, Δ isa AbstractArray ? batch(real.(Δ[:])) : Δ)
end


function preprocess((_,proj)::Tuple{BaseFieldStyle{S,B},<:ProjLambert}, ∇d::∇diag) where {S,B}

    (B <: Union{Fourier,QUFourier,IQUFourier}) ||
        error("Can't broadcast ∇[$(∇d.coord)] as a $(typealias(B)), its not diagonal in this basis.")

    # turn both into 2D matrices so this function is type-stable
    # (reshape doesnt actually make a copy here, so this doesn't
    # impact performance)
    if ∇d.coord == 1
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓx, 1, :))
    else
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓy, :, 1))
    end
end

function preprocess((_,proj)::Tuple{BaseFieldStyle{S,B},<:ProjLambert}, ::∇²diag) where {S,B}
    
    (B <: Union{Fourier,<:Basis2Prod{<:Any,Fourier},<:Basis3Prod{<:Any,<:Any,Fourier}}) ||
        error("Can't broadcast a BandPass as a $(typealias(B)), its not diagonal in this basis.")

    broadcasted(+, broadcasted(^, proj.ℓx', 2), broadcasted(^, proj.ℓy, 2))
end

function preprocess((_,proj)::Tuple{<:Any,<:ProjLambert}, bp::BandPass)
    Cℓ_to_2D(bp.Wℓ, proj)
end

function Cℓ_to_2D(Cℓ, proj::ProjLambert{T}) where {T}
    Complex{T}.(nan2zero.(Cℓ.(proj.ℓmag)))
end


### adapting

# dont adapt the fields in proj, instead re-call into the memoized
# ProjLambert so we always get back the singleton ProjLambert object
# for the given set of parameters (helps reduce memory usage and
# speed-up subsequent broadcasts which would otherwise not hit the
# "===" branch of the "promote_*" methods)
function adapt_structure(storage, proj::ProjLambert{T}) where {T}
    @unpack Ny, Nx, θpix = proj
    T′ = eltype(storage)
    ProjLambert(;Ny, Nx, θpix, T=(T′==Any ? T : real(T′)), storage)
end
adapt_structure(::Nothing, proj::ProjLambert{T}) where {T} = proj



@doc doc"""
    pixwin(θpix, ℓ)

Returns the pixel window function for square flat-sky pixels of width `θpix` (in
arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
the power spectrum will be pixwin^2. 
"""
pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)



### serialization
# makes it so the arrays in ProjLambert objects aren't actually
# serialized, instead just (Ny, Nx, θpix, rotator, T) are stored, and
# deserializing just recreates the ProjLambert object, possibly from
# the memoized cache if it already exists.

function _serialization_key(proj::ProjLambert{T}) where {T}
    @unpack Ny, Nx, θpix, rotator, storage = proj
    (;Ny, Nx, θpix, rotator, T, storage)
end

# Julia serialization
function Serialization.serialize(s::AbstractSerializer, proj::ProjLambert)
    @unpack Ny, Nx, θpix, rotator = proj
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, ProjLambert)
    Serialization.serialize(s, _serialization_key(proj))
end
function Serialization.deserialize(s::AbstractSerializer, ::Type{ProjLambert})
    ProjLambert(; Serialization.deserialize(s)...)
end

# JLD2 serialization
# (always deserialize as Array)
function JLD2.writeas(::Type{<:ProjLambert})
    Tuple{Val{ProjLambert},NamedTuple{(:Ny,:Nx,:θpix,:rotator,:T),Tuple{Int,Int,Float64,NTuple{3,Float64},DataType}}}
end
function JLD2.wconvert(::Type{<:Tuple{Val{ProjLambert},NamedTuple}}, proj::ProjLambert)
    (Val(ProjLambert), delete(_serialization_key(proj), :storage))
end
function JLD2.rconvert(::Type{<:ProjLambert}, (_,s)::Tuple{Val{ProjLambert},NamedTuple})
    ProjLambert(; storage=Array, s...)
end



### indices
function getindex(f::LambertS0, k::Symbol; full_plane=false)
    maybe_unfold = full_plane ? x->unfold(x,fieldinfo(f).Ny) : identity
    @match k begin
        :I  => f
        :Ix => Map(f).Ix
        :Il => maybe_unfold(Fourier(f).Il)
        _   => throw(ArgumentError("Invalid LambertS0 index: $k"))
    end
end
function getindex(f::LambertS2{Basis2Prod{B₁,B₂}}, k::Symbol; full_plane=false) where {B₁,B₂}
    maybe_unfold = (full_plane && k in [:El,:Bl,:Ql,:Ul]) ? x->unfold(x,fieldinfo(f).Ny) : identity
    B = @match k begin
        (:P)         => identity
        (:E  || :B)  => Basis2Prod{𝐄𝐁,B₂}
        (:Q  || :U)  => Basis2Prod{𝐐𝐔,B₂}
        (:Ex || :Bx) => Basis2Prod{𝐄𝐁,Map}
        (:El || :Bl) => Basis2Prod{𝐄𝐁,Fourier}
        (:Qx || :Ux) => Basis2Prod{𝐐𝐔,Map}
        (:Ql || :Ul) => Basis2Prod{𝐐𝐔,Fourier}
        _ => throw(ArgumentError("Invalid LambertS2 index: $k"))
    end
    maybe_unfold(getproperty(B(f),k))
end
function getindex(f::LambertS02{Basis3Prod{B₁,B₂,B₃}}, k::Symbol; full_plane=false) where {B₁,B₂,B₃}
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
        _ => throw(ArgumentError("Invalid LambertS02 index: $k"))
    end
    maybe_unfold(getproperty(B(f),k))
end
function getindex(D::DiagOp{<:LambertEBFourier}, k::Symbol)
    @unpack El, Bl, metadata = diag(D)
    @unpack sin2ϕ, cos2ϕ = fieldinfo(diag(D))
    f = @match k begin
        (:QQ)        => LambertFourier((@. Bl*sin2ϕ^2 + El*cos2ϕ^2),   metadata)
        (:QU || :UQ) => LambertFourier((@. (El - Bl) * sin2ϕ * cos2ϕ), metadata)
        (:UU)        => LambertFourier((@. Bl*cos2ϕ^2 + El*sin2ϕ^2),   metadata)
        _            => getproperty(D.diag, k)
    end
    Diagonal(f)
end
function getindex(L::BlockDiagIEB{<:Any,<:LambertField}, k::Symbol)
    @match k begin
        :IP => L
        :I => L.ΣTE[1,1]
        :E => L.ΣTE[2,2]
        :B => L.ΣB
        :P => Diagonal(LambertEBFourier(L[:E].diag, L[:B].diag))
        (:QQ || :UU || :QU || :UQ) => getindex(L[:P], k)
        _ => throw(ArgumentError("Invalid BlockDiagIEB index: $k"))
    end
end


### basis conversion
## spin-0
Fourier(f::LambertMap) = LambertFourier(m_rfft(f.arr, (1,2)), f.metadata)
Fourier(f′::LambertFourier, f::LambertMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)
Map(f::LambertFourier) = LambertMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
Map(f′::LambertMap, f::LambertFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)

## spin-2
QUFourier(f::LambertQUMap) = LambertQUFourier(m_rfft(f.arr, (1,2)), f.metadata)
QUFourier(f::LambertEBMap) = f |> EBFourier |> QUFourier
QUFourier(f::LambertEBFourier) = begin
    @unpack El, Bl, sin2ϕ, cos2ϕ = fieldinfo(f)
    Ql = @. - El * cos2ϕ + Bl * sin2ϕ
    Ul = @. - El * sin2ϕ - Bl * cos2ϕ
    LambertQUFourier(Ql, Ul, f.metadata)
end

QUMap(f::LambertQUFourier) = LambertQUMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
QUMap(f::LambertEBMap)      = f |> EBFourier |> QUFourier |> QUMap
QUMap(f::LambertEBFourier)  = f |> QUFourier |> QUMap

EBFourier(f::LambertEBMap) = LambertEBFourier(m_rfft(f.arr, (1,2)), f.metadata)
EBFourier(f::LambertQUMap) = f |> QUFourier |> EBFourier
EBFourier(f::LambertQUFourier) = begin
    @unpack Ql, Ul, sin2ϕ, cos2ϕ = fieldinfo(f)
    El = @. - Ql * cos2ϕ - Ul * sin2ϕ
    Bl = @.   Ql * sin2ϕ - Ul * cos2ϕ
    LambertEBFourier(El, Bl, f.metadata)
end

EBMap(f::LambertEBFourier) = LambertEBMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
EBMap(f::LambertQUMap)     = f |> QUFourier |> EBFourier |> EBMap
EBMap(f::LambertQUFourier) = f |> EBFourier |> EBMap

# in-place
QUMap(f′::LambertQUMap, f::LambertQUFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)
QUFourier(f′::LambertQUFourier, f::LambertQUMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)

## spin-(0,2)
IQUFourier(f::LambertIQUMap) = LambertIQUFourier(m_rfft(f.arr, (1,2)), f.metadata)
IQUFourier(f::LambertIEBMap) = f |> IEBFourier |> IQUFourier
IQUFourier(f::LambertIEBFourier) = LambertIQUFourier(f.I, QUFourier(f.P))

IQUMap(f::LambertIQUFourier) = LambertIQUMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
IQUMap(f::LambertIEBMap)      = f |> IEBFourier |> IQUFourier |> IQUMap
IQUMap(f::LambertIEBFourier)  = f |> IQUFourier |> IQUMap

IEBFourier(f::LambertIEBMap) = LambertIEBFourier(m_rfft(f.arr, (1,2)), f.metadata)
IEBFourier(f::LambertIQUMap) = f |> IQUFourier |> IEBFourier
IEBFourier(f::LambertIQUFourier) = LambertIEBFourier(f.I, EBFourier(f.P))

IEBMap(f::LambertIEBFourier) = LambertIEBMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
IEBMap(f::LambertIQUMap)     = f |> IQUFourier |> IEBFourier |> IEBMap
IEBMap(f::LambertIQUFourier) = f |> IEBFourier |> IEBMap

# in-place
IQUMap(f′::LambertIQUMap, f::LambertIQUFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)
IQUFourier(f′::LambertIQUFourier, f::LambertIQUMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)

# spin-0 bases applied to spin-2 and spin-(0,2)
Fourier(f::LambertField{B}) where {B<:BasisProd} = Fourier(B)(f)
Map(f::LambertField{B}) where {B<:BasisProd} = Map(B)(f)



### dot products
# do in Map space (the LenseBasis, Ł) for simplicity
function dot(a::LambertField, b::LambertField)
    z = Ł(a) .* Ł(b)
    batch(sum_dropdims(z.arr, dims=nonbatch_dims(z)))
end

### logdets

function logdet(L::Diagonal{<:Union{Real,Complex},<:LambertField{B}}) where {B<:Union{Fourier,Basis2Prod{<:Any,Fourier},Basis3Prod{<:Any,<:Any,Fourier}}}
    # half the Fourier plane needs to be counted twice since the real
    # FFT only stores half of it
    @unpack Ny, arr = L.diag
    λ = adapt(typeof(arr), rfft_degeneracy_fac(Ny))
    # note: since our maps are required to be real, the logdet of any
    # operator which preserves this property is also guaranteed to be
    # real, hence the `real` and `abs` below are valid
    batch(real.(sum_dropdims(nan2zero.(log.(abs.(arr)) .* λ), dims=nonbatch_dims(L.diag))))
end

function logdet(L::Diagonal{<:Real,<:LambertField{B}}) where {B<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}}}
    batch(
        sum_dropdims(log.(abs.(L.diag.arr)), dims=nonbatch_dims(L.diag)) 
        .+ dropdims(log.(prod(sign.(L.diag.arr), dims=nonbatch_dims(L.diag))), dims=nonbatch_dims(L.diag))
    )
end


### traces

function tr(L::Diagonal{<:Union{Real,Complex},<:LambertField{B}}) where {B<:Union{Fourier,Basis2Prod{<:Any,Fourier},Basis3Prod{<:Any,<:Any,Fourier}}}
    @unpack Ny, Nx, arr = L.diag
    λ = adapt(typeof(arr), rfft_degeneracy_fac(Ny))
    # the `real` is ok bc the imaginary parts of the half-plane which
    # is stored would cancel with those from the other half-plane
    batch(real.(sum_dropdims(arr .* λ, dims=nonbatch_dims(L.diag))))
end

function tr(L::Diagonal{<:Real,<:LambertField{B}}) where {B<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}}}
    batch(sum_dropdims(L.diag.arr, dims=nonbatch_dims(L.diag)))
end




### simulation
_white_noise(ξ::LambertField, rng::AbstractRNG) = 
    (randn!(rng, similar(ξ.arr, real(eltype(ξ)), ξ.Ny, size(ξ.arr)[2:end]...)), ξ.metadata)
white_noise(ξ::LambertS0,  rng::AbstractRNG) = LambertMap(_white_noise(ξ,rng)...)
white_noise(ξ::LambertS2,  rng::AbstractRNG) = LambertEBMap(_white_noise(ξ,rng)...)
white_noise(ξ::LambertS02, rng::AbstractRNG) = LambertIEBMap(_white_noise(ξ,rng)...)


### creating covariance operators
# fixed covariances
Cℓ_to_Cov(pol::Symbol, args...; kwargs...) = Cℓ_to_Cov(Val(pol), args...; kwargs...)
function Cℓ_to_Cov(::Val{:I}, proj::ProjLambert, Cℓ::InterpolatedCℓs; units=proj.Ωpix)
    Diagonal(LambertFourier(Cℓ_to_2D(Cℓ,proj), proj) / units)
end
function Cℓ_to_Cov(::Val{:P}, proj::ProjLambert, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs; units=proj.Ωpix)
    Diagonal(LambertEBFourier(Cℓ_to_2D(CℓEE,proj), Cℓ_to_2D(CℓBB,proj), proj) / units)
end
function Cℓ_to_Cov(::Val{:IP}, proj::ProjLambert, CℓTT, CℓEE, CℓBB, CℓTE; kwargs...)
    ΣTT, ΣEE, ΣBB, ΣTE = [Cℓ_to_Cov(:I,proj,Cℓ; kwargs...) for Cℓ in (CℓTT,CℓEE,CℓBB,CℓTE)]
    BlockDiagIEB(@SMatrix([ΣTT ΣTE; ΣTE ΣEE]), ΣBB)
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
            _A = $preprocess.(Ref((nothing,C₀.metadata)), $T.($ensure1d($θname)))
            Diagonal(LambertFourier($bandpower_rescale!(ℓedges, C₀.ℓmag, copy(C₀.arr), _A...), C₀.metadata))
        end)
    end
end
function Cℓ_to_Cov(::Val{:P}, proj::ProjLambert{T}, (CℓEE, ℓedges, θname)::Tuple, CℓBB::InterpolatedCℓs; kwargs...) where {T}
    C₀ = diag(Cℓ_to_Cov(:P, proj, CℓEE, CℓBB; kwargs...))
    @eval Main let ℓedges=$((T.(ℓedges))...,), C₀=$C₀
        ParamDependentOp(function (;$θname=ones($T,length(ℓedges)-1),_...)
            _E = $preprocess.(Ref((nothing,C₀.metadata)),      $T.($ensure1d($θname)))
            _B = $preprocess.(Ref((nothing,C₀.metadata)), one.($T.($ensure1d($θname))))
            Diagonal(LambertEBFourier($bandpower_rescale!(ℓedges, C₀.ℓmag, copy(C₀.El), _E...), C₀.Bl .* _B[1], C₀.metadata))
        end)
    end
end
# this is written weird because the stuff inside the broadcast! needs
# to work as a GPU kernel
function bandpower_rescale!(ℓedges, ℓ, Cℓ, A...)
    length(A)==length(ℓedges)-1 || error("Expected $(length(ℓedges)-1) bandpower parameters, got $(length(A)).")
    eltype(A[1]) <: Real || error("Bandpower parameters must be real numbers.")
    if length(A)>30
        # if more than 30 bandpowers, we need to chunk the rescaling
        # because of a maximum argument limit of CUDA kernels
        for p in partition(1:length(A), 30)
            bandpower_rescale!(ℓedges[p.start:(p.stop+1)], ℓ, Cℓ, A[p]...)
        end
    else
        broadcast!(Cℓ, ℓ, Cℓ, A...) do ℓ, Cℓ, A...
            for i=1:length(ℓedges)-1
                (ℓedges[i] < ℓ < ℓedges[i+1]) && return A[i] * Cℓ
            end
            return Cℓ
        end
    end
    Cℓ
end
# cant reliably get Zygote's gradients to work through these
# broadcasts, which on GPU use ForwardDiff, so write the adjoint by
# hand for now. likely more performant, in any case. 
@adjoint function bandpower_rescale!(ℓedges, ℓ, Cℓ, A...)
    function back(Δ)
        Ā = map(1:length(A)) do i
            sum(
                real,
                broadcast(Δ, ℓ, Cℓ) do Δ, ℓ, Cℓ
                    (ℓedges[i] < ℓ < ℓedges[i+1]) ? Cℓ*Δ : zero(Cℓ)
                end,
                dims = ndims(Δ)==4 ? (1,2) : (:)
            )
        end
        (nothing, nothing, nothing, Ā...)
    end
    bandpower_rescale!(ℓedges, ℓ, Cℓ, A...), back
end
function cov_to_Cℓ(C::DiagOp{<:LambertS0}; kwargs...)
    @unpack Nx, Ny, Δx = diag(C)
    α = Nx*Ny/Δx^2
    get_Cℓ(sqrt.(diag(C)); kwargs...)*sqrt(α)
end




### spin adjoints
function *(a::SpinAdjoint{F}, b::F) where {B<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}},F<:LambertField{B}}
    r = sum(a.f.arr .* b.arr, dims=3)
    LambertMap(dropdims(r, dims=(ndims(r)==3 ? 3 : ())), get_metadata_strict(a, b))
end
function mul!(dst::LambertMap, a::SpinAdjoint{F}, b::F) where {F<:LambertField{<:Union{Map,Basis2Prod{<:Any,Map},Basis3Prod{<:Any,<:Any,Map}}}}
    dst.arr .= reshape(sum(a.f.arr .* b.arr, dims=3), size(dst.arr))
    dst
end



### batching

batch_length(f::LambertField) = f.Nbatch

"""
    batch(fs::LambertField...)
    batch(fs::Vector{<:LambertField})

Concatenate one of more LambertFields along the "batch" dimension
(dimension 4 of the underlying array). For the inverse operation, see
[`unbatch`](@ref). 
"""
batch(fs::LambertField{B}...) where {B} = batch(collect(fs))
batch(fs::AbstractVector{<:LambertField{B}}) where {B} =
    LambertField{B}(cat(getfield.(fs,:arr)..., dims=Val(4)), only(unique(getfield.(fs,:metadata))))

"""
    unbatch(f::LambertField)

Return an Array of LambertFields corresponding to each batch index. For
the inverse operation, see [`batch`](@ref).
"""
unbatch(f::LambertField{B}) where {B} = [f[!,i] for i=1:batch_length(f)]

batch_index(f::LambertField{B}, I) where {B<:Union{Map,Fourier}} = LambertField{B}(f.arr[:,:,1,I], f.metadata)
batch_index(f::LambertField{B}, I) where {B} = LambertField{B}(f.arr[:,:,:,I], f.metadata)


###

make_mask(f::LambertField; kwargs...) = make_mask((f.Ny,f.Nx), f.θpix; kwargs...)



### power spectra

function get_Cℓ(f₁::LambertS0, f₂::LambertS0=f₁; Δℓ=50, ℓedges=0:Δℓ:16000, Cℓfid=ℓ->1, err_estimate=false)
    @unpack Nx, Ny, Δx, ℓmag = fieldinfo(f₁)
    ℓmag = unfold(ℓmag, Ny)
    α = Nx*Ny/Δx^2

    # faster to excise unused parts:
    ℓmask = (ℓmag .> minimum(ℓedges)) .&  (ℓmag .< maximum(ℓedges))
    L = ℓmag[ℓmask]
    CLobs = 1/α .* real.(dot.(
        adapt(Array,f₁)[:Il, full_plane=true][ℓmask], 
        adapt(Array,f₂)[:Il, full_plane=true][ℓmask]
    ))
    w = @. nan2zero((2*Cℓfid(L)^2/(2L+1))^-1)
    
    sum_in_ℓbins(x) = Float64.(fit(Histogram, L, Weights(x), ℓedges).weights)

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

function get_Cℓ(f1::LambertS2, f2::LambertS2=f1; which=(:EE,:BB), kwargs...)
    Cℓ = (;[Symbol(x1*x2) => get_Cℓ(getindex(f1,Symbol(x1)),getindex(f2,Symbol(x2)); kwargs...) for (x1,x2) in split.(string.(ensure1d(which)),"")]...)
    which isa Symbol ? Cℓ[1] : Cℓ
end

function get_Cℓ(f1::LambertS02, f2::LambertS02=f1; which=(:II,:EE,:BB,:IE,:IB,:EB), kwargs...)
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
    f :: LambertField{B},
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
    Ny_new, Nx_new = @. round(Int, N * θ ÷ θnew)
    proj = ProjLambert(;Ny=Ny_new, Nx=Nx_new, θpix=θnew, T=real(T), f.storage)
    @unpack Δx,ℓx,ℓy,Nx,Ny,nyquist = proj

    PWF = deconv_pixwin ? Diagonal(LambertFourier((@. T((pixwin(θnew,ℓy)*pixwin(θnew,ℓx)')/(pixwin(θ,ℓy)*pixwin(θ,ℓx)'))), proj)) : I

    if θnew > θ
        # downgrade
        if anti_aliasing
            f = Diagonal(LambertFourier(ifelse.((abs.(f.ℓy) .>= nyquist) .| (abs.(f.ℓx') .>= nyquist), 0, 1), f.metadata)) * f
        end
        if mode == :map
            fnew = LambertField{Map(B)}(dropdims(mean(reshape(Map(f).arr, fac, Ny, fac, Nx, size.(Ref(f.arr),nonbatch_dims(f)[3:end])...), dims=(1,3)), dims=(1,3)), proj)
        else
            fnew = LambertField{Fourier(B)}(Fourier(f).arr[1:(Ny_new÷2+1), [1:(isodd(Nx_new) ? Nx_new÷2+1 : Nx_new÷2); (end-Nx_new÷2+1):end], repeated(:, length(nonbatch_dims(f))-2)...], proj)
        end
        if deconv_pixwin
            fnew = Diagonal(LambertFourier((@. T((pixwin(θnew,ℓy)*pixwin(θnew,ℓx)')/(pixwin(θ,ℓy)*pixwin(θ,ℓx)'))), proj)) \ fnew
        end
    else
        error("Not implemented")
        # # upgrade
        # @assert fieldinfo(f).Nside isa Int "Upgrading resolution only implemented for maps where `Nside isa Int`"
        # if mode==:map
        #     fnew = LambertMap{Pnew}(permutedims(hvcat(N,(x->fill(x,(fac,fac))).(f[:Ix])...)))
        #     deconv_pixwin ? LambertFourier{Pnew}(fnew[:Il] .* PWF' .* PWF[1:Nnew÷2+1]) : fnew
        # else
        #     fnew = LambertFourier{Pnew}(zeros(Nnew÷2+1,Nnew))
        #     setindex!.(Ref(fnew.Il), f[:Il], 1:(N÷2+1), [findfirst(fieldinfo(fnew).k .≈ fieldinfo(f).k[i]) for i=1:N]')
        #     deconv_pixwin ? fnew * fac^2 : fnew
        # end

    end
    return fnew
end