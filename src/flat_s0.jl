
# this file defines a flat-sky pixelized spin-0 map (like T or ϕ)
# and operators on this map

export FlatFourier, FlatMap, FlatS0


## FlatMap and FlatFourier types
struct FlatMap{P<:Flat,T<:Real,M<:AbstractMatrix{T}} <: Field{Map,S0,P,T}
    Ix :: M
end
struct FlatFourier{P<:Flat,T<:Real,M<:AbstractMatrix{Complex{T}}} <: Field{Fourier,S0,P,Complex{T}}
    Il :: M
end
const FlatS0{P,T,M} = Union{FlatMap{P,T,M},FlatFourier{P,T,M}}

## convenience constructors
FlatMap(Ix; θpix=θpix₀, ∂mode=fourier∂) = FlatMap{Flat{size(Ix,2),θpix,∂mode}}(Ix)
FlatMap{P}(Ix::M) where {P,T,M<:AbstractMatrix{T}} = FlatMap{P,T,M}(Ix)
FlatFourier(Il; θpix=θpix₀, ∂mode=fourier∂) = FlatFourier{Flat{size(Il,2),θpix,∂mode}}(Il)
FlatFourier{P}(Il::M) where {P,T,M<:AbstractMatrix{Complex{T}}} = FlatFourier{P,T,M}(Il)

## pretty printing
Base.print_array(io::IO, f::FlatS0) = Base.print_array(io, broadcast_data(f)[:])
Base.summary(io::IO, f::F) where {N,θpix,∂mode,F<:FlatS0{Flat{N,θpix,∂mode}}} = 
print(io, "$(length(f))-element $(F.name.name){$N×$N map, $(θpix)′ pixels, $(∂mode.name.name)}")

## array interface 
size(f::FlatS0) = (length(broadcast_data(f)),)
@propagate_inbounds @inline getindex(f::FlatS0, I...) = getindex(broadcast_data(f), I...)
@propagate_inbounds @inline setindex!(f::FlatS0, X, I...) = setindex!(broadcast_data(f), X, I...)

## broadcasting
BroadcastStyle(::Type{F}) where {F<:FlatMap} = ArrayStyle{F}()
similar(bc::Broadcasted{ArrayStyle{F}}, ::Type{T}) where {T, N, P<:Flat{N}, F<:FlatMap{P}} = FlatMap{P}(similar(Array{T}, N, N))
function Broadcast.preprocess(dest::F, bc::Broadcasted{Nothing}) where {F<:Field}
    bc′ = Broadcast.flatten(bc)
    Broadcasted{Nothing}(bc′.f, map(arg->broadcast_data(F,arg), bc′.args), axes(dest))
end
broadcast_data(::Any, f) = f
broadcast_data(f::FlatS0) = first(fieldvalues(f))




# 
# # convenience conversion funtions:
# Fourier(f::FlatMap{T,P}) where {T,P} = FlatFourier{T,P}(ℱ{P}*f.Ix)
# Map(f::FlatFourier{T,P}) where {T,P} = FlatMap{T,P}(ℱ{P}\f.Tl)
# 
# # inplace conversions
# Fourier(f′::FlatFourier{T,P}, f::FlatMap{T,P}) where {T,P} = (mul!(f′.Tl,  FFTgrid(T,P).FFT, f.Ix); f′)
# Map(f′::FlatMap{T,P}, f::FlatFourier{T,P}) where {T,P}     = (ldiv!(f′.Ix, FFTgrid(T,P).FFT, f.Tl); f′)
# 
# 
# LenseBasis(::Type{<:FlatS0}) = Map
# 
# function white_noise(::Type{F}) where {Θ,Nside,T,P<:Flat{Θ,Nside},F<:FlatS0{T,P}}
#     FlatMap{T,P}(randn(Nside,Nside) / FFTgrid(T,P).Δx)
# end
# 
# """ Convert power spectrum Cℓ to a flat sky diagonal covariance """
# function Cℓ_to_cov(::Type{T}, ::Type{P}, ::Type{S0}, CℓTT::InterpolatedCℓs) where {T,P}
#     g = FFTgrid(T,P)
#     FullDiagOp(FlatFourier{T,P}(Cℓ_2D(CℓTT.ℓ, CℓTT.Cℓ, g.r)[1:g.nside÷2+1,:]))
# end
# 
# function cov_to_Cℓ(L::FullDiagOp)
#     ii = sortperm(FFTgrid(L.f).r[:])
#     InterpolatedCℓs(FFTgrid(L.f).r[ii], real.(unfold(L.f.Tl))[ii], concrete=false)
# end
# 
# function get_Cℓ(f::FlatS0{T,P}, f2::FlatS0{T,P}=f; Δℓ=50, ℓedges=0:Δℓ:16000, Cℓfid=ℓ->1) where {T,P}
#     g = FFTgrid(T,P)
#     α = g.Δx^2/(4π^2)*g.nside^2
# 
#     L = g.r[:]
#     CLobs = real.(dot.(unfold(f.Tl),unfold(f2.Tl)))[:]
#     w = @. nan2zero((2*Cℓfid(L)^2/(2L+1))^-1)
# 
#     power       = fit(Histogram, L, Weights(w .* CLobs), ℓedges).weights
#     bandcenters = fit(Histogram, L, Weights(w .* L),     ℓedges).weights
#     counts      = fit(Histogram, L, Weights(w),          ℓedges).weights
# 
#     InterpolatedCℓs(bandcenters ./ counts,  power ./ counts ./ α)
# end
# 
# zero(::Type{<:FlatS0{T,P}}) where {T,P} = FlatMap{T,P}(zeros(Nside(P),Nside(P)))
# one(::Type{<:FlatMap{T,P}}) where {T,P} = FlatMap{T,P}(ones(Nside(P),Nside(P)))
# one(::Type{<:FlatFourier{T,P}}) where {T,P} = FlatFourier{T,P}(ones(Complex{T},Nside(P)÷2+1,Nside(P)))
# 
# # dot products
# dot(a::FlatMap{T,P}, b::FlatMap{T,P}) where {T,P} = dot(a.Ix,b.Ix) * FFTgrid(T,P).Δx^2
# dot(a::FlatFourier{T,P}, b::FlatFourier{T,P}) where {T,P} = begin
#     @unpack nside,Δℓ = FFTgrid(T,P)
#     if isodd(nside)
#         @views real(2 * (a.Tl[2:end,:][:] ⋅ b.Tl[2:end,:][:]) + (a.Tl[1,:][:] ⋅ b.Tl[1,:][:])) * Δℓ^2
#     else
#         @views real(2 * (a.Tl[2:end-1,:][:] ⋅ b.Tl[2:end-1,:][:]) + (a.Tl[[1,end],:][:] ⋅ b.Tl[[1,end],:][:])) * Δℓ^2
#     end
# end
# 
# 
# 
# # vector conversion
# length(::Type{<:FlatS0{T,P}}) where {T,P} = Nside(P)^2
# getindex(f::FlatMap,::Colon) = f.Ix[:]
# getindex(f::FlatFourier,::Colon) = rfft2vec(f.Tl)
# fromvec(::Type{FlatMap{T,P}},     vec::AbstractVector) where {T,P} = FlatMap{T,P}(reshape(vec,(Nside(P),Nside(P))))
# fromvec(::Type{FlatFourier{T,P}}, vec::AbstractVector) where {T,P} = FlatFourier{T,P}(vec2rfft(vec))
# 
# 
# 
# """
#     ud_grade(f::Field, θnew, mode=:map, deconv_pixwin=true, anti_aliasing=true)
# 
# Up- or down-grades field `f` to new resolution `θnew` (only in integer steps).
# Two modes are available specified by the `mode` argument: 
# 
#     *`:map`     : Up/downgrade by replicating/averaging pixels in map-space
#     *`:fourier` : Up/downgrade by extending/truncating the Fourier grid
# 
# For `:map` mode, two additional options are possible. If `deconv_pixwin` is
# true, deconvolves the pixel window function from the downgraded map so the
# spectrum of the new and old maps are the same. If `anti_aliasing` is true,
# filters out frequencies above Nyquist prior to down-sampling. 
# 
# """
# function ud_grade(f::FlatS0{T,P}, θnew; mode=:map, deconv_pixwin=(mode==:map), anti_aliasing=(mode==:map)) where {T,θ,N,∂mode,P<:Flat{θ,N,∂mode}}
#     θnew==θ && return f
#     (isinteger(θnew//θ) || isinteger(θ//θnew)) || throw(ArgumentError("Can only ud_grade in integer steps"))
#     (mode in [:map,:fourier]) || throw(ArgumentError("Available modes: [:map,:fourier]"))
# 
#     fac = θnew > θ ? θnew÷θ : θ÷θnew
#     Nnew = N * θ ÷ θnew
#     Pnew = Flat{θnew,Nnew,∂mode}
# 
#     if deconv_pixwin
#         @unpack Δx,k = FFTgrid(T,Pnew)
#         Wk =  @. T(pixwin(θnew, k) / pixwin(θ, k))
#     end
# 
#     if θnew>θ
#         # downgrade
#         if anti_aliasing
#             kmask = ifelse.(abs.(FFTgrid(T,P).k) .> FFTgrid(T,Pnew).nyq, 0, 1)
#             AA = FullDiagOp(FlatFourier{T,P}(kmask[1:N÷2+1] .* kmask'))
#         else
#             AA = 1
#         end
#         if mode==:map
#             fnew = FlatMap{T,Pnew}(mapslices(mean,reshape((AA*f)[:Ix],(fac,Nnew,fac,Nnew)),dims=(1,3))[1,:,1,:])
#             deconv_pixwin ? FlatFourier{T,Pnew}(fnew[:Tl] ./ Wk' ./ Wk[1:Nnew÷2+1]) : fnew
#         else
#             FlatFourier{T,Pnew}((AA*f)[:Tl][1:(Nnew÷2+1), [1:(isodd(Nnew) ? Nnew÷2+1 : Nnew÷2); (end-Nnew÷2+1):end]])
#         end
#     else
#         # upgrade
#         if mode==:map
#             fnew = FlatMap{T,Pnew}(hvcat(N,(x->fill(x,(fac,fac))).(f.Ix)...)')
#             deconv_pixwin ? FlatFourier{T,Pnew}(fnew[:Tl] .* Wk' .* Wk[1:Nnew÷2+1]) : fnew
#         else
#             fnew = Fourier(zero(FlatMap{T,Pnew}))
#             broadcast_setindex!(fnew.Tl, f[:Tl], 1:(N÷2+1), [findfirst(FFTgrid(fnew).k .≈ FFTgrid(f).k[i]) for i=1:N]');
#             fnew
#         end
#     end
# end
