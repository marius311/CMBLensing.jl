
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
FlatFourier(Il; kwargs...) = FlatFourier{Flat(Nside=size(Ix,2);kwargs...)}(Il)
FlatFourier{P}(Il::M) where {P,T,M<:AbstractMatrix{Complex{T}}} = FlatFourier{P,T,M}(Il)

### pretty printing
show_datatype(io::IO, ::Type{F}) where {N,θ,∂mode,T,M,F<:FlatS0{Flat{N,θ,∂mode},T,M}} =
    print(io, "$(F.name.name){$(N)×$(N) map, $(θ)′ pixels, $(∂mode.name.name), $(M.name.name){$(M.parameters[1])}}")

### array interface 
size(f::FlatS0) = (length(firstfield(f)),)
@propagate_inbounds @inline getindex(f::FlatS0, I...) = getindex(firstfield(f), I...)
@propagate_inbounds @inline setindex!(f::FlatS0, X, I...) = (setindex!(firstfield(f), X, I...); f)
similar(f::F) where {F<:FlatS0} = similar(F,eltype(f))
similar(::Type{F}) where {T,F<:FlatS0{<:Any,T}} = similar(F,T)
similar(f::F,::Type{T}) where {T,F<:FlatS0} = similar(F,T)
similar(::Type{F},::Type{T}) where {N,P<:Flat{N},T,M,F<:FlatMap{P,<:Any,M}} = FlatMap{P}(basetype(M){T}(undef,N,N))
similar(::Type{F},::Type{T}) where {N,P<:Flat{N},T,M,F<:FlatFourier{P,<:Any,M}} = FlatFourier{P}(basetype(M){T}(undef,N÷2+1,N))

### broadcasting
BroadcastStyle(::Type{F}) where {F<:FlatS0} = ArrayStyle{F}()
BroadcastStyle(::ArrayStyle{F1}, ::ArrayStyle{F2}) where {P,F1<:FlatMap{P},F2<:FlatMap{P}} = ArrayStyle{FlatMap{P,Real,Matrix{Real}}}()
BroadcastStyle(::ArrayStyle{F1}, ::ArrayStyle{F2}) where {P,F1<:FlatFourier{P},F2<:FlatFourier{P}} = ArrayStyle{FlatFourier{P,Real,Matrix{Complex{Real}}}}()
BroadcastStyle(::ArrayStyle{FT}, ::ArrayStyle{<:FlatS0}) where {FT<:FieldTuple} = ArrayStyle{FT}()
similar(bc::Broadcasted{ArrayStyle{F}}, ::Type{T}) where {T, F<:FlatS0} = similar(F,T)
@inline preprocess(dest::F, bc::Broadcasted) where {F<:FlatS0} = Broadcasted{DefaultArrayStyle{2}}(bc.f, preprocess_args(dest, bc.args))
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
    CLobs = real.(dot.(unfold(f.Il),unfold(f2.Il)))[:]
    w = @. nan2zero((2*Cℓfid(L)^2/(2L+1))^-1)

    power       = fit(Histogram, L, Weights(w .* CLobs), ℓedges).weights
    bandcenters = fit(Histogram, L, Weights(w .* L),     ℓedges).weights
    counts      = fit(Histogram, L, Weights(w),          ℓedges).weights

    InterpolatedCℓs(bandcenters ./ counts,  power ./ counts ./ α)
end

# zero(::Type{<:FlatS0{T,P}}) where {T,P} = FlatMap{T,P}(zeros(Nside(P),Nside(P)))
# one(::Type{<:FlatMap{T,P}}) where {T,P} = FlatMap{T,P}(ones(Nside(P),Nside(P)))
# one(::Type{<:FlatFourier{T,P}}) where {T,P} = FlatFourier{T,P}(ones(Complex{T},Nside(P)÷2+1,Nside(P)))
# 
# # dot products
# dot(a::FlatMap{T,P}, b::FlatMap{T,P}) where {T,P} = dot(a.Ix,b.Ix) * FFTgrid(P,T).Δx^2
# dot(a::FlatFourier{T,P}, b::FlatFourier{T,P}) where {T,P} = begin
#     @unpack nside,Δℓ = FFTgrid(P,T)
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
#             kmask = ifelse.(abs.(FFTgrid(P,T).k) .> FFTgrid(T,Pnew).nyq, 0, 1)
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
