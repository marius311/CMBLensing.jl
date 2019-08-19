
### FlatMap and FlatFourier types
struct FlatMap{P<:Flat,T<:Real,M<:AbstractMatrix{T}} <: Field{Map,S0,P,T}
    Ix :: M
end
struct FlatFourier{P<:Flat,T<:Real,M<:AbstractMatrix{Complex{T}}} <: Field{Fourier,S0,P,Complex{T}}
    Il :: M
end
const FlatS0{P,T,M} = Union{FlatMap{P,T,M},FlatFourier{P,T,M}}

### convenience constructors
Flat(;Nside, θpix=θpix₀, ∂mode=fourier∂) = Flat{Nside,θpix,∂mode}
FlatMap(Ix; kwargs...) = FlatMap{Flat(Nside=size(Ix,2);kwargs...)}(Ix)
FlatMap{P}(Ix::M) where {P,T,M<:AbstractMatrix{T}} = FlatMap{P,T,M}(Ix)
FlatFourier(Il; kwargs...) = FlatFourier{Flat(Nside=size(Il,2);kwargs...)}(Il)
FlatFourier{P}(Il::M) where {P,T,M<:AbstractMatrix{Complex{T}}} = FlatFourier{P,T,M}(Il)
FlatFourier{P}(Il::AbstractMatrix{<:Real}) where {P} = FlatFourier{P}(complex(Il))

### array interface 
size(f::FlatS0) = (length(firstfield(f)),)
lastindex(f::FlatS0, i::Int) = lastindex(f.Ix, i)
size_2d(::Type{<:FlatMap{<:Flat{N}}}) where {N} = (N,N)
size_2d(::Type{<:FlatFourier{<:Flat{N}}}) where {N} = (N÷2+1,N)
@propagate_inbounds @inline getindex(f::FlatS0, I...) = getindex(firstfield(f), I...)
@propagate_inbounds @inline setindex!(f::FlatS0, X, I...) = (setindex!(firstfield(f), X, I...); f)
similar(f::F) where {F<:FlatS0} = similar(F,eltype(f))
similar(::Type{F}) where {T,F<:FlatS0{<:Any,T}} = similar(F,T)
similar(f::F,::Type{T}) where {T,F<:FlatS0} = similar(F,T)
similar(::Type{F},::Type{T}) where {N,P<:Flat{N},T,M,F<:FlatS0{P,<:Any,M}} = basetype(F){P}(basetype(M){T}(undef,size_2d(F)...))

### broadcasting
BroadcastStyle(::Type{F}) where {F<:FlatS0} = ArrayStyle{F}()
BroadcastStyle(::ArrayStyle{F1}, ::ArrayStyle{F2}) where {P,F1<:FlatMap{P},F2<:FlatMap{P}} = ArrayStyle{FlatMap{P,Real,Matrix{Real}}}()
BroadcastStyle(::ArrayStyle{F1}, ::ArrayStyle{F2}) where {P,F1<:FlatFourier{P},F2<:FlatFourier{P}} = ArrayStyle{FlatFourier{P,Real,Matrix{Complex{Real}}}}()
BroadcastStyle(::ArrayStyle{FT}, ::ArrayStyle{<:FlatS0}) where {FT<:FieldTuple} = ArrayStyle{FT}()
similar(bc::Broadcasted{ArrayStyle{F}}, ::Type{T}) where {T, F<:FlatS0} = similar(F,T)
@inline preprocess(dest::F, bc::Broadcasted) where {F<:FlatS0} = Broadcasted{DefaultArrayStyle{2}}(bc.f, preprocess_args(dest, bc.args), map(OneTo,size_2d(F)))
preprocess(dest::F, arg) where {F<:FlatS0} = broadcastable(F, arg)
broadcastable(::Type{<:FlatS0{P}}, f::FlatS0{P}) where {P} = firstfield(f)
broadcastable(::Any, x) = x

### basis conversion
Fourier(f::FlatMap{P,T}) where {P,T} = FlatFourier{P}(FFTgrid(P,T).FFT * f.Ix)
Map(f::FlatFourier{P,T}) where {P,T} = FlatMap{P}(FFTgrid(P,T).FFT \ f.Il)

### inplace conversion
Fourier(f′::FlatFourier{P,T}, f::FlatMap{P,T}) where {P,T} =  (mul!(f′.Il, FFTgrid(P,T).FFT, f.Ix); f′)
Map(f′::FlatMap{P,T}, f::FlatFourier{P,T}) where {P,T}     = (ldiv!(f′.Ix, FFTgrid(P,T).FFT, f.Il); f′)

### properties
getproperty(f::FlatS0, ::Val{s}) where {s} = getproperty(f,s)
function getindex(f::FlatS0, k::Symbol)
    k in [:Ix,:Il] || throw(ArgumentError("Invalid FlatS0 index: $k"))
    getproperty([Map,Fourier][k .== [:Ix,:Il]][1](f),k)
end

### simulation and power spectra
function white_noise(::Type{F}) where {N,T,P<:Flat{N},F<:FlatS0{P,T}}
    FlatMap{P}(randn(T,N,N) / FFTgrid(P,T).Δx)
end
function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S0}, Cℓ::InterpolatedCℓs) where {P,T}
    Diagonal(FlatFourier{P}(Cℓ_to_2D(P,T,Cℓ)))
end


function cov_to_Cℓ(L::DiagOp{<:FlatS0})
    ii = sortperm(FFTgrid(L.diag).r[:])
    InterpolatedCℓs(FFTgrid(L.diag).r[ii], real.(unfold(L.diag.Il))[ii], concrete=false)
end

function get_Cℓ(f::FlatS0{P}, f2::FlatS0{P}=f; Δℓ=50, ℓedges=0:Δℓ:16000, Cℓfid=ℓ->1) where {P}
    g = FFTgrid(f)
    α = g.Δx^2/(4π^2)*g.Nside^2

    L = g.r[:]
    CLobs = real.(dot.(unfold(f[:Il]),unfold(f2[:Il])))[:]
    w = @. nan2zero((2*Cℓfid(L)^2/(2L+1))^-1)

    power       = fit(Histogram, L, Weights(w .* CLobs), ℓedges).weights
    bandcenters = fit(Histogram, L, Weights(w .* L),     ℓedges).weights
    counts      = fit(Histogram, L, Weights(w),          ℓedges).weights

    InterpolatedCℓs(bandcenters ./ counts,  power ./ counts ./ α)
end

"""
    ud_grade(f::Field, θnew, mode=:map, deconv_pixwin=true, anti_aliasing=true)

Up- or down-grades field `f` to new resolution `θnew` (only in integer steps).
Two modes are available specified by the `mode` argument: 

    *`:map`     : Up/downgrade by replicating/averaging pixels in map-space
    *`:fourier` : Up/downgrade by extending/truncating the Fourier grid

For `:map` mode, two additional options are possible. If `deconv_pixwin` is
true, deconvolves the pixel window function from the downgraded map so the
spectrum of the new and old maps are the same. If `anti_aliasing` is true,
filters out frequencies above Nyquist prior to down-sampling. 

"""
function ud_grade(f::FlatS0{P,T}, θnew; mode=:map, deconv_pixwin=(mode==:map), anti_aliasing=(mode==:map)) where {T,θ,N,∂mode,P<:Flat{N,θ,∂mode}}
    θnew==θ && return f
    (isinteger(θnew//θ) || isinteger(θ//θnew)) || throw(ArgumentError("Can only ud_grade in integer steps"))
    (mode in [:map,:fourier]) || throw(ArgumentError("Available modes: [:map,:fourier]"))

    fac = θnew > θ ? θnew÷θ : θ÷θnew
    Nnew = N * θ ÷ θnew
    Pnew = Flat{Nnew,θnew,∂mode}

    if deconv_pixwin
        @unpack Δx,k = FFTgrid(Pnew,T)
        Wk =  @. T(pixwin(θnew, k) / pixwin(θ, k))
    end

    if θnew>θ
        # downgrade
        if anti_aliasing
            kmask = ifelse.(abs.(FFTgrid(P,T).k) .> FFTgrid(Pnew,T).nyq, 0, 1)
            AA = Diagonal(FlatFourier{P}(kmask[1:N÷2+1] .* kmask'))
        else
            AA = 1
        end
        if mode==:map
            fnew = FlatMap{Pnew}(mapslices(mean,reshape((AA*f)[:Ix],(fac,Nnew,fac,Nnew)),dims=(1,3))[1,:,1,:])
            deconv_pixwin ? FlatFourier{Pnew}(fnew[:Il] ./ Wk' ./ Wk[1:Nnew÷2+1]) : fnew
        else
            FlatFourier{Pnew}((AA*f)[:Il][1:(Nnew÷2+1), [1:(isodd(Nnew) ? Nnew÷2+1 : Nnew÷2); (end-Nnew÷2+1):end]])
        end
    else
        error("Upgrading resolution not implemented yet.") # still needs updating from old code
        # upgrade
        # if mode==:map
        #     fnew = FlatMap{Pnew}(hvcat(N,(x->fill(x,(fac,fac))).(f[:Ix])...)')
        #     deconv_pixwin ? FlatFourier{Pnew}(fnew[:Il] .* Wk' .* Wk[1:Nnew÷2+1]) : fnew
        # else
        #     fnew = FlatFourier{P}(zeros(Nnew÷2+1,Nnew))
        #     broadcast_setindex!(fnew.Il, f[:Il], 1:(N÷2+1), [findfirst(FFTgrid(fnew).k .≈ FFTgrid(f).k[i]) for i=1:N]');
        #     fnew
        # end
    end
end
