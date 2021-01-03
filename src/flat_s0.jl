
## FlatS0 types
const FlatMap{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{Map, M, T, A}
const FlatFourier{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{Fourier, M, T, A}
## FlatS0 unions
const FlatS0{M,T,A} = Union{FlatMap{M,T,A}, FlatFourier{M,T,A}}

## constructors
function FlatMap(Ix::A; θpix=θpix₀) where {T, A<:AbstractArray{T}}
    FlatMap(
        drop_tail_singleton_dims(reshape(Ix, size(Ix,1), size(Ix,2), 1, size(Ix,3))),
        ProjLambert(T, basetype(A), θpix, size(Ix,1), size(Ix,2))
    )
end

# FlatFourier(Il::A; Ny, θpix=θpix₀) where {T, A<:AbstractArray{T}} = 
#     FlatFourier(complex(Il[:,:,:,:]), ProjLambert(real(T), basetype(A), θpix, Ny, size(Ix,2)...))

## properties
getproperty(f::FlatMap,     ::Val{:Ix}) = getfield(f,:arr)
getproperty(f::FlatFourier, ::Val{:Il}) = getfield(f,:arr)

## basis conversion
# out-of-place
Fourier(f::FlatMap) = FlatFourier(m_rfft(f.Ix, (1,2)), f.metadata)
Map(f::FlatFourier) = FlatMap(m_irfft(f.Il, f.Ny, (1,2)), f.metadata)
# in-place
Fourier(f′::FlatFourier, f::FlatMap) =  (m_rfft!(f′.Il, f.Ix, (1,2)); f′)
Map(f′::FlatMap, f::FlatFourier)     = (m_irfft!(f′.Ix, f.Il, (1,2)); f′)


# ### convenience constructors
# for (F, X, T) in [
#     (:FlatMap,     :Ix, :T),
#     (:FlatFourier, :Il, :(Complex{T})),
# ]
#     doc = """
#         # main constructor:
#         $F(
#             $X::AbstractArray; $(F==:FlatFourier ? "\n        Nside, # required, size of the map in pixels" : "")
#             θpix,  # optional, resolution in arcmin (default: 1)
#             ∂mode, # optional, fourier∂ or map∂ (default: fourier∂)
#         )
        
#         # more low-level:
#         $F{P}($X::AbstractArray) # specify pixelization P explicilty
#         $F{P,T}($X::AbstractArray) # additionally, convert elements to type $T
#         $F{P,T,M<:AbstractArray{$T}}($X::M) # specify everything explicilty
        
#     Construct a `$F` object. The top form of the constructor is most convenient
#     for interactive work, while the others may be more useful for low-level code.
#     """
#     @eval begin
#         @doc $doc $F
#         $F{P}($X::M) where {P,T,M<:AbstractRank2or3Array{$T}} = $F{P,T,M}($X)
#         $F{P,T}($X::AbstractRank2or3Array) where {P,T} = $F{P}($T.($X))
#     end
#     T!=:T && @eval $F{P}($X::M) where {P,T,M<:AbstractRank2or3Array{T}} = $F{P,T}($X)
# end
# FlatMap(Ix::AbstractRank2or3Array; kwargs...) = 
#     FlatMap{Flat(Nside=(size(Ix,1)==size(Ix,2) ? size(Ix,1) : size(Ix)[1:2]), D=size(Ix,3); kwargs...)}(Ix)
# FlatFourier(Il::AbstractRank2or3Array; Nside, kwargs...) = 
#     FlatFourier{Flat(Nside=Nside, D=size(Il,3); kwargs...)}(Il)


### array interface 
# adapt_structure(to, f::F) where {P,F<:FlatS0{P}} = basetype(F){P}(adapt(to,firstfield(f)))
# adapt_structure(::Type{T}, f::F) where {T<:Union{Float32,Float64},P,F<:FlatS0{P}} = T(f)


# ### inplace conversion
# Fourier(f′::FlatFourier, f::FlatMap) =  (mul!(f′.Il, fieldinfo(f).FFT, f.Ix); f′)
# Map(f′::FlatMap, f::FlatFourier)     = (ldiv!(f′.Ix, fieldinfo(f).FFT, maybecopy(f.Il)); f′)
# # need this for FFTW (but not MKL) see https://github.com/JuliaMath/FFTW.jl/issues/158
# @static FFTW.fftw_vendor==:fftw ? maybecopy(x) = copy(x) : maybecopy(x) = x

# ### dot products
# do in Map space for simplicity, and use sum_kbn to reduce roundoff error
# dot(a::FlatField, b::FlatField) where {N,θ} = batch(sum_kbn(Map(a).Ix .* Map(b).Ix, dims=(1,2)))

# ### isapprox
# ≈(a::F, b::F) where {P,T,F<:FlatS0{P,T}} = all(.≈(a[:], b[:], atol=sqrt(eps(T)), rtol=sqrt(eps(T))))

# ### simulation and power spectra
# function white_noise(rng::AbstractRNG, ::Type{F}) where {N,P<:Flat{N},T,M,F<:FlatS0{P,T,M}}
#     FlatMap{P}(randn!(rng, basetype(M){T}(undef, content_size(FlatMap{P}))))
# end
# function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S0}, Cℓ::InterpolatedCℓs; units=fieldinfo(P).Ωpix) where {P,T}
#     Diagonal(FlatFourier{P}(Cℓ_to_2D(P,T,Cℓ)) / units)
# end
# function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S0}, (Cℓ, ℓedges, θname)::Tuple; units=fieldinfo(P).Ωpix) where {P,T}
#     C₀ = Cℓ_to_Cov(P, T, S0, Cℓ, units=units)
#     Cbins = Diagonal.(MidPasses(ℓedges) .* [diag(C₀)])
#     BinRescaledOp(C₀,Cbins,θname)
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
#     @unpack Δx,kx,ky,Nx,Ny,nyq = fieldinfo(Pnew,T,M)

#     if deconv_pixwin
#         PWF = @. T((pixwin(θnew,ky[1:end÷2+1])*pixwin(θnew,kx)')/(pixwin(θ,ky[1:end÷2+1])*pixwin(θ,kx)'))
#     else
#         PWF = 1
#     end

#     if θnew>θ
#         # downgrade
#         if anti_aliasing
#             AA = Diagonal(FlatFourier{P}(
#                 ifelse.((abs.(fieldinfo(P,T,M).ky[1:end÷2+1]) .> nyq) .| (abs.(fieldinfo(P,T,M).kx') .> nyq), 0, 1)
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
