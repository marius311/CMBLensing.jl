
# this file defines a flat-sky pixelized spin-0 map (like T or ϕ)
# and operators on this map

export FlatS0Fourier, FlatS0Map, FlatS0

struct FlatS0Map{T<:Real,P<:Flat} <: Field{Map,S0,P}
    Tx::Matrix{T}
    FlatS0Map{T,P}(Tx::AbstractMatrix) where {T,P} = new{T,P}(checkmap(P,Tx))
end

struct FlatS0Fourier{T<:Real,P<:Flat} <: Field{Fourier,S0,P}
    Tl::Matrix{Complex{T}}
    FlatS0Fourier{T,P}(Tl::AbstractMatrix) where {T,P} = new{T,P}(checkfourier(P,Tl))
end

const FlatS0{T,P}=Union{FlatS0Map{T,P},FlatS0Fourier{T,P}}

# convenience constructors
FlatS0Map(Tx::Matrix{T},Θpix=Θpix₀,∂mode=fourier∂) where {T} = FlatS0Map{T,Flat{Θpix,size(Tx,2),∂mode}}(Tx)
FlatS0Fourier(Tl::Matrix{Complex{T}},Θpix=Θpix₀,∂mode=fourier∂) where {T} = FlatS0Fourier{T,Flat{Θpix,size(Tl,2),∂mode}}(Tl)

# convenience conversion funtions:
Fourier(f::FlatS0Map{T,P}) where {T,P} = FlatS0Fourier{T,P}(ℱ{P}*f.Tx)
Map(f::FlatS0Fourier{T,P}) where {T,P} = FlatS0Map{T,P}(ℱ{P}\f.Tl)

# inplace conversions
Fourier(f′::FlatS0Fourier{T,P}, f::FlatS0Map{T,P}) where {T,P} = (mul!(f′.Tl,  FFTgrid(T,P).FFT, f.Tx); f′)
Map(f′::FlatS0Map{T,P}, f::FlatS0Fourier{T,P}) where {T,P}     = (ldiv!(f′.Tx, FFTgrid(T,P).FFT, f.Tl); f′)


LenseBasis(::Type{<:FlatS0}) = Map

function white_noise(::Type{F}) where {Θ,Nside,T,P<:Flat{Θ,Nside},F<:FlatS0{T,P}}
    FlatS0Map{T,P}(randn(Nside,Nside) / FFTgrid(T,P).Δx)
end

""" Convert power spectrum Cℓ to a flat sky diagonal covariance """
function Cℓ_to_cov(::Type{T}, ::Type{P}, ::Type{S0}, ℓ, CℓTT) where {T,P}
    g = FFTgrid(T,P)
    FullDiagOp(FlatS0Fourier{T,P}(Cℓ_2D(ℓ, CℓTT, g.r)[1:g.nside÷2+1,:]))
end

function get_Cℓ(f::FlatS0{T,P}, f2::FlatS0{T,P}=f; Δℓ=50, ℓedges=0:Δℓ:16000) where {T,P}
    g = FFTgrid(T,P)
    α = g.Δx^2/(4π^2)*g.nside^2
    power = fit(Histogram,g.r[:],Weights(real.(dot.(unfold(f.Tl),unfold(f2.Tl)))[:]),ℓedges,closed=:right)
    counts = fit(Histogram,g.r[:],ℓedges,closed=:right)
    h = Histogram(ℓedges, (@. power.weights / counts.weights / α), :right)
    ((h.edges[1][1:end-1]+h.edges[1][2:end])/2, h.weights)
end


zero(::Type{<:FlatS0{T,P}}) where {T,P} = FlatS0Map{T,P}(zeros(Nside(P),Nside(P)))


# dot products
dot(a::FlatS0Map{T,P}, b::FlatS0Map{T,P}) where {T,P} = dot(a.Tx,b.Tx) * FFTgrid(T,P).Δx^2
dot(a::FlatS0Fourier{T,P}, b::FlatS0Fourier{T,P}) where {T,P} = begin
    @unpack nside,Δℓ = FFTgrid(T,P)
    if isodd(nside)
        @views real(2 * (a.Tl[2:end,:][:] ⋅ b.Tl[2:end,:][:]) + (a.Tl[1,:][:] ⋅ b.Tl[1,:][:])) * Δℓ^2
    else
        @views real(2 * (a.Tl[2:end-1,:][:] ⋅ b.Tl[2:end-1,:][:]) + (a.Tl[[1,end],:][:] ⋅ b.Tl[[1,end],:][:])) * Δℓ^2
    end
end



# vector conversion
length(::Type{<:FlatS0{T,P}}) where {T,P} = Nside(P)^2
getindex(f::FlatS0Map,::Colon) = f.Tx[:]
getindex(f::FlatS0Fourier,::Colon) = rfft2vec(f.Tl)
fromvec(::Type{FlatS0Map{T,P}},     vec::AbstractVector) where {T,P} = FlatS0Map{T,P}(reshape(vec,(Nside(P),Nside(P))))
fromvec(::Type{FlatS0Fourier{T,P}}, vec::AbstractVector) where {T,P} = FlatS0Fourier{T,P}(vec2rfft(vec))



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
function ud_grade(f::FlatS0{T,P}, θnew; mode=:map, deconv_pixwin=(mode==:map), anti_aliasing=(mode==:map)) where {T,θ,N,∂mode,P<:Flat{θ,N,∂mode}}
    θnew==θ && return f
    (isinteger(θnew//θ) || isinteger(θ//θnew)) || throw(ArgumentError("Can only ud_grade in integer steps"))
    (mode in [:map,:fourier]) || throw(ArgumentError("Available modes: [:map,:fourier]"))
    
    fac = θnew > θ ? θnew÷θ : θ÷θnew
    Nnew = N * θ ÷ θnew
    Pnew = Flat{θnew,Nnew,∂mode}
    
    if deconv_pixwin
        @unpack Δx,k = FFTgrid(T,Pnew)
        Wk =  @. T(pixwin(θnew, k) / pixwin(θ, k))
    end
    
    if θnew>θ
        # downgrade
        if anti_aliasing
            kmask = ifelse.(abs.(FFTgrid(T,P).k) .> FFTgrid(T,Pnew).nyq, 0, 1)
            AA = FullDiagOp(FlatS0Fourier{T,P}(kmask[1:N÷2+1] .* kmask'))
        else
            AA = 1
        end
        if mode==:map
            fnew = FlatS0Map{T,Pnew}(mapslices(mean,reshape((AA*f)[:Tx],(fac,Nnew,fac,Nnew)),(1,3))[1,:,1,:])
            deconv_pixwin ? FlatS0Fourier{T,Pnew}(fnew[:Tl] ./ Wk' ./ Wk[1:Nnew÷2+1]) : fnew
        else
            FlatS0Fourier{T,Pnew}((AA*f)[:Tl][1:(Nnew÷2+1), [1:Nnew÷2; (end-Nnew÷2+1):end]])
        end
    else
        # upgrade
        if mode==:map
            fnew = FlatS0Map{T,Pnew}(hvcat(N,(x->fill(x,(fac,fac))).(f.Tx)...)')
            deconv_pixwin ? FlatS0Fourier{T,Pnew}(fnew[:Tl] .* Wk' .* Wk[1:Nnew÷2+1]) : fnew
        else
            fnew = Fourier(zero(FlatS0Map{T,Pnew}))
            broadcast_setindex!(fnew.Tl, f[:Tl], 1:(N÷2+1), [findfirst(FFTgrid(fnew).k .≈ FFTgrid(f).k[i]) for i=1:N]');
            fnew
        end
    end
end
