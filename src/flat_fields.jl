
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

_reshape_batch(arr::AbstractArray{T,3}) where {T} = reshape(arr, size(arr,1), size(arr,2), 1, size(arr,3))
_reshape_batch(arr) = arr

function FlatMap(Ix::A; kwargs...) where {T, A<:AbstractArray{T}}
    FlatMap(
        _reshape_batch(Ix),
        ProjLambert(;Ny=size(Ix,1), Nx=size(Ix,2), T, storage=basetype(A), kwargs...)
    )
end
function FlatFourier(Il::A; Ny, kwargs...) where {T, A<:AbstractArray{T}}
    FlatFourier(
        _reshape_batch(Ix),
        ProjLambert(;Ny, Nx=size(Il,2), T, storage=basetype(A), kwargs...)
    )
end


for (F,Xs,Ny) in [
    (:FlatQUMap,      (:Qx,:Ux),      false),
    (:FlatQUFourier,  (:Ql,:Ul),      true),
    (:FlatEBMap,      (:Ex,:Bx),      false),
    (:FlatEBFourier,  (:El,:Bl),      true),
    (:FlatIQUMap,     (:Ix, :Qx,:Ux), false),
    (:FlatIQUFourier, (:Il, :Ql,:Ul), true),
    (:FlatIEBMap,     (:Ix, :Ex,:Bx), false),
    (:FlatIEBFourier, (:Il, :El,:Bl), true),
]
    @eval begin
        function $F($((:($X::A) for X in Xs)...), metadata) where {T,A<:AbstractArray{T}}
            $F(
                cat($((:(_reshape_batch($X)) for X in Xs)...), dims=Val(3)),
                metadata
            )
        end
        function $F($((:($X::A) for X in Xs)...); $((Ny ? (:Ny,) : ())...), kwargs...) where {T,A<:AbstractArray{T}}
            $F(
                $(Xs...),
                ProjLambert(;Ny=$(Ny ? :Ny : :(size($(Xs[1]),1))), Nx=size($(Xs[1]),2), T, storage=basetype(A), kwargs...)
            )
        end
    end
end

# todo: enumerate rest and add doc strings


### properties
# generic
getproperty(f::FlatField, ::Val{:Nbatch}) = size(getfield(f,:arr),4)
getproperty(f::FlatField, ::Val{:T})      = eltype(f)
# sub-components
for (F,ks) in [
    (FlatMap,        ("Ix",)),
    (FlatFourier,    ("Il",)),
    (FlatQUMap,      ("Qx", "Ux")),
    (FlatQUFourier,  ("Ql", "Ul")),
    (FlatEBMap,      ("Ex", "Bx")),
    (FlatEBFourier,  ("El", "Bl")),
    (FlatIQUMap,     ("Ix", "Qx", "Ux")),
    (FlatIQUFourier, ("Il", "Ql", "Ul")),
    (FlatIEBMap,     ("Ix", "Ex", "Bx")),
    (FlatIEBFourier, ("Il", "El", "Bl"))
]
    for (i,k) in enumerate(ks)
        F₀ = endswith(k,"x") ? FlatMap : FlatFourier
        @eval getproperty(f::$F, ::Val{$(QuoteNode(Symbol(k)))}) = view(getfield(f,:arr), :, :, $i, ..)
        @eval getproperty(f::$F, ::Val{$(QuoteNode(Symbol(k[1])))}) = $F₀(view(getfield(f,:arr), :, :, $i, ..), f.metadata)
    end
end
# not sure I really like these...
# getproperty(f::FlatS2{B}, ::Val{:P}) where {B} = FlatS2{B}(view(getfield(f,:arr)),                f.metadata)
# getproperty(f::FlatS02Map,     ::Val{:I})  =       FlatMap(view(getfield(f,:arr), :, :, 1,   ..), f.metadata)
# getproperty(f::FlatS02Fourier, ::Val{:I})  =   FlatFourier(view(getfield(f,:arr), :, :, 1,   ..), f.metadata)
# getproperty(f::FlatIQUMap,     ::Val{:P})  =     FlatQUMap(view(getfield(f,:arr), :, :, 2:3, ..), f.metadata)
# getproperty(f::FlatIQUFourier, ::Val{:P})  = FlatQUFourier(view(getfield(f,:arr), :, :, 2:3, ..), f.metadata)
# getproperty(f::FlatIEBMap,     ::Val{:P})  =     FlatEBMap(view(getfield(f,:arr), :, :, 2:3, ..), f.metadata)
# getproperty(f::FlatIEBFourier, ::Val{:P})  = FlatEBFourier(view(getfield(f,:arr), :, :, 2:3, ..), f.metadata)


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
function getindex(f::FlatS2, k::Symbol; full_plane=false)
    maybe_unfold = (full_plane && k in [:El,:Bl,:Ql,:Ul]) ? x->unfold(x,fieldinfo(f).Ny) : identity
    B = @match k begin
        (:P)         => identity
        (:E  || :B)  => @match f begin
            _ :: FlatQUMap     => EBMap
            _ :: FlatQUFourier => EBFourier
            _                  => identity
        end
        (:Q  || :U)  => @match f begin
            _ :: FlatEBMap     => QUMap
            _ :: FlatEBFourier => QUFourier
            _                  => identity
        end
        (:Ex || :Bx) => EBMap
        (:El || :Bl) => EBFourier
        (:Qx || :Ux) => QUMap
        (:Ql || :Ul) => QUFourier
        _ => throw(ArgumentError("Invalid FlatS2 index: $k"))
    end
    maybe_unfold(getproperty(B(f),k))
end
function getindex(f::FlatS02, k::Symbol; kwargs...)
    @match k begin
        (:IP) => f
        (:I || :P) => getfield(f.fs,k)
        (:Q || :U || :E || :B) => getindex(f.P,k; kwargs...)
        (:Ix || :Il) => getindex(f.I,k; kwargs...)
        (:Qx || :Ux || :Ql || :Ul || :Ex || :Bx || :El || :Bl) => getindex(f.P,k; kwargs...)
        _ => throw(ArgumentError("Invalid FlatS02 index: $k"))
    end
end
function getindex(D::DiagOp{<:FlatEBFourier}, k::Symbol)
    @unpack El, Bl = diag(D)
    @unpack sin2ϕ, cos2ϕ = fieldinfo(diag(D))
    f = @match k begin
        (:QQ)        => FlatFourier((@. Bl*sin2ϕ^2 + El*cos2ϕ^2),   f.metadata)
        (:QU || :UQ) => FlatFourier((@. (El - Bl) * sin2ϕ * cos2ϕ), f.metadata)
        (:UU)        => FlatFourier((@. Bl*cos2ϕ^2 + El*sin2ϕ^2),   f.metadata)
        _            => getproperty(D.diag, k)
    end
    Diagonal(f)
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

QUMap(f′::FlatQUMap, f::FlatQUFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)
QUFourier(f′::FlatQUFourier, f::FlatQUMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)

Map(f::FlatQUFourier) = QUMap(f)
Map(f::FlatEBFourier) = EBMap(f)
Map(f::FlatS2Map) = f
Fourier(f::FlatQUMap) = QUFourier(f)
Fourier(f::FlatEBMap) = EBFourier(f)
Fourier(f::FlatS2Fourier) = f

## spin-(0,2)
# ...




### pretty printing
typealias_def(::Type{F}) where {B,M,T,A,F<:FlatField{B,M,T,A}} = "Flat$(typealias(B)){$(typealias(A)),$(typealias(M))}"
function Base.summary(io::IO, f::FlatField)
    @unpack Nx,Ny,Nbatch,θpix = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-map $(θpix)′-pixels ")
    Base.showarg(io, f, true)
end



### dot products

nonbatch_dims(f::FlatField) = ntuple(identity,min(3,ndims(f.arr)))

# do in Map space (the LenseBasis, Ł) for simplicity, and use sum_kbn to reduce roundoff error
function dot(a::FlatField, b::FlatField)
    z = Ł(a) .* Ł(b)
    sum_kbn(z.arr, dims=nonbatch_dims(z))
end


### logdets
logdet(L::Diagonal{<:Complex,<:FlatFourier}) = real(sum_kbn(nan2zero.(log.(L.diag[:Il,full_plane=true])),dims=(1,2)))
logdet(L::Diagonal{<:Real,   <:FlatMap})     = real(sum_kbn(nan2zero.(log.(complex.(L.diag.Ix))),dims=(1,2)))
# ### traces
# tr(L::Diagonal{<:Complex,<:FlatFourier}) = batch(real(sum_kbn(L.diag[:Il,full_plane=true],dims=(1,2))))
# tr(L::Diagonal{<:Real,   <:FlatMap})     = batch(real(sum_kbn(complex.(L.diag.Ix),dims=(1,2))))


# ### isapprox
# ≈(a::F, b::F) where {P,T,F<:FlatS0{P,T}} = all(.≈(a[:], b[:], atol=sqrt(eps(T)), rtol=sqrt(eps(T))))


### simulation
_white_noise(rng::AbstractRNG, f::FlatField) = (randn!(similar(f.arr, real(eltype(f)), f.Ny, size(f.arr)[2:end]...)), f.metadata)
white_noise(rng::AbstractRNG, f::FlatS0)  = FlatMap(_white_noise(rng,f)...)
white_noise(rng::AbstractRNG, f::FlatS2)  = FlatEBMap(_white_noise(rng,f)...)
white_noise(rng::AbstractRNG, f::FlatS02) = FlatIEBMap(_white_noise(rng,f)...)


### covariance operators
Cℓ_to_Cov(pol::Symbol, args...; kwargs...) = Cℓ_to_Cov(Val(pol), args...; kwargs...)
function Cℓ_to_Cov(::Val{:I}, proj::ProjLambert, Cℓ::InterpolatedCℓs; units=proj.Ωpix)
    Diagonal(FlatFourier(Cℓ_to_2D(Cℓ,proj), proj) / units)
end
function Cℓ_to_Cov(::Val{:P}, proj::ProjLambert, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs; units=proj.Ωpix)
    Diagonal(FlatEBFourier(cat(Cℓ_to_2D(CℓEE,proj),Cℓ_to_2D(CℓBB,proj), dims=3), proj) / units)
end


# function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S0}, (Cℓ, ℓedges, θname)::Tuple; units=fieldinfo(P).Ωpix) where {P,T}
#     C₀ = Cℓ_to_Cov(P, T, S0, Cℓ, units=units)
#     Cbins = Diagonal.(MidPasses(ℓedges) .* [diag(C₀)])
#     BinRescaledOp(C₀,Cbins,θname)
# end
# function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S2}, (CℓEE, ℓedges, θname)::Tuple, CℓBB::InterpolatedCℓs; units=fieldinfo(P).Ωpix) where {P,T}
#     C₀ = Cℓ_to_Cov(P, T, S2, CℓEE, CℓBB, units=units)
#     Cbins = Diagonal.(FlatEBFourier.(MidPasses(ℓedges) .* [diag(C₀).E], [zero(diag(C₀).B)]))
#     BinRescaledOp(C₀,Cbins,θname)
# end




### power spectra



# function get_Cℓ(f1::FlatS2, f2::FlatS2=f1; which=(:EE,:BB), kwargs...)
#     Cℓ = (;[Symbol(x1*x2) => get_Cℓ(getindex(f1,Symbol(x1)),getindex(f2,Symbol(x2)); kwargs...) for (x1,x2) in split.(string.(ensure1d(which)),"")]...)
#     which isa Symbol ? Cℓ[1] : Cℓ
# end


# function ud_grade(f::FlatS2{P}, args...; kwargs...) where {P} 
#     f′ = FieldTuple(map(f->ud_grade(f, args...; kwargs...), f.fs))
#     B′ = (f′[1] isa FlatMap) ? (f isa FlatQU ? QUMap : EBMap) : (f isa FlatQU ? QUFourier : EBFourier)
#     FieldTuple{B′}(f′)
# end

# function cov_to_Cℓ(L::DiagOp{<:FlatS0{P}}; units=fieldinfo(P).Ωpix) where {P}
#     ii = sortperm(fieldinfo(L.diag).kmag[:])
#     InterpolatedCℓs(fieldinfo(L.diag).kmag[ii], real.(unfold(L.diag.Il, fieldinfo(L.diag).Ny))[ii] * units, concrete=false)
# end

# function get_Cℓ(f::FlatS0{P}, f2::FlatS0{P}=f; Δℓ=50, ℓedges=0:Δℓ:16000, Cℓfid=ℓ->1, err_estimate=false) where {P}
#     @unpack Nx, Ny ,Δx,kmag = fieldinfo(f)
#     α = Nx*Ny/Δx^2

#     # faster to excise unused parts:
#     kmask = (kmag .> minimum(ℓedges)) .&  (kmag .< maximum(ℓedges)) 
#     L = Float64.(kmag[kmask])
#     CLobs = real.(dot.(unfold(Float64(f)[:Il],Ny)[kmask], unfold(Float64(f2)[:Il],Ny)[kmask])) ./ α
#     w = @. nan2zero((2*Cℓfid(L)^2/(2L+1))^-1)
    
#     sum_in_ℓbins(x) = fit(Histogram, L, Weights(x), ℓedges).weights

#     local A, Cℓ, ℓ, N, Cℓ²
#     Threads.@sync begin
#         Threads.@spawn A  = sum_in_ℓbins(w)
#         Threads.@spawn Cℓ = sum_in_ℓbins(w .* CLobs)
#         Threads.@spawn ℓ  = sum_in_ℓbins(w .* L)
#         if err_estimate
#             Threads.@spawn N   = sum_in_ℓbins(one.(w)) / 2
#             Threads.@spawn Cℓ² = sum_in_ℓbins(w .* CLobs.^2)
#         end
#     end

#     if err_estimate
#         σℓ  = sqrt.((Cℓ² ./ A .- Cℓ.^2) ./ N)
#         InterpolatedCℓs(ℓ./A,  Cℓ./A .± σℓ)
#     else
#         InterpolatedCℓs(ℓ./A,  Cℓ./A)
#     end
# end




# """
#     ud_grade(f::Field, θnew, mode=:map, deconv_pixwin=true, anti_aliasing=true)

# Up- or down-grades field `f` to new resolution `θnew` (only in integer steps).
# Two modes are available specified by the `mode` argument: 

# * `:map`     — Up/downgrade by replicating/averaging pixels in map-space
# * `:fourier` — Up/downgrade by extending/truncating the Fourier grid

# For `:map` mode, two additional options are possible. If `deconv_pixwin` is
# true, deconvolves the pixel window function from the downgraded map so the
# spectrum of the new and old maps are the same. If `anti_aliasing` is true,
# filters out frequencies above Nyquist prior to down-sampling. 

# """
# function ud_grade(f::FlatS0{P,T,M}, θnew; mode=:map, deconv_pixwin=(mode==:map), anti_aliasing=(mode==:map)) where {T,M,θ,N,∂mode,P<:Flat{N,θ,∂mode}}
#     θnew==θ && return f
#     (mode in [:map,:fourier]) || throw(ArgumentError("Available modes: [:map,:fourier]"))

#     fac = θnew > θ ? θnew÷θ : θ÷θnew
#     (round(Int, fac) ≈ fac) || throw(ArgumentError("Can only ud_grade in integer steps"))
#     fac = round(Int, fac)
#     Nnew = @. round(Int, N * θ ÷ θnew)
#     Pnew = Flat(Nside=Nnew, θpix=θnew, ∂mode=∂mode)
#     @unpack Δx,kx,ky,Nx,Ny,nyquist = fieldinfo(Pnew,T,M)

#     if deconv_pixwin
#         PWF = @. T((pixwin(θnew,ky[1:end÷2+1])*pixwin(θnew,kx)')/(pixwin(θ,ky[1:end÷2+1])*pixwin(θ,kx)'))
#     else
#         PWF = 1
#     end

#     if θnew>θ
#         # downgrade
#         if anti_aliasing
#             AA = Diagonal(FlatFourier{P}(
#                 ifelse.((abs.(fieldinfo(P,T,M).ky[1:end÷2+1]) .> nyquist) .| (abs.(fieldinfo(P,T,M).kx') .> nyquist), 0, 1)
#             ))
#         else
#             AA = 1
#         end
#         if mode==:map
#             fnew = FlatMap{Pnew}(mapslices(mean,reshape((AA*f)[:Ix],(fac,Ny,fac,Nx)),dims=(1,3))[1,:,1,:])
#             deconv_pixwin ? FlatFourier{Pnew}(fnew[:Il] ./ PWF) : fnew
#         else
#             @assert fieldinfo(f).Nside isa Int "Downgrading resolution with `mode=:fourier` only implemented for maps where `Nside isa Int`"
#             FlatFourier{Pnew}((AA*f)[:Il][1:(Nnew÷2+1), [1:(isodd(Nnew) ? Nnew÷2+1 : Nnew÷2); (end-Nnew÷2+1):end]])
#         end
#     else
#         # upgrade
#         @assert fieldinfo(f).Nside isa Int "Upgrading resolution only implemented for maps where `Nside isa Int`"
#         if mode==:map
#             fnew = FlatMap{Pnew}(permutedims(hvcat(N,(x->fill(x,(fac,fac))).(f[:Ix])...)))
#             deconv_pixwin ? FlatFourier{Pnew}(fnew[:Il] .* PWF' .* PWF[1:Nnew÷2+1]) : fnew
#         else
#             fnew = FlatFourier{Pnew}(zeros(Nnew÷2+1,Nnew))
#             setindex!.(Ref(fnew.Il), f[:Il], 1:(N÷2+1), [findfirst(fieldinfo(fnew).k .≈ fieldinfo(f).k[i]) for i=1:N]')
#             deconv_pixwin ? fnew * fac^2 : fnew
#         end
#     end
# end
