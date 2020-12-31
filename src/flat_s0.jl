

abstract type FlatProj end

const FlatMap{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{Map, M, T, A}
const FlatFourier{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{Fourier, M, T, A}

const FlatS0{M,T,A} = Union{FlatMap{M,T,A}, FlatFourier{M,T,A}}

LenseBasis(::Type{<:FlatS0}) = Map
DerivBasis(::Type{<:FlatS0}) = Fourier


const FlatField{G, M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{G, M, T, A}


# default angular resolution used by a number of convenience constructors
θpix₀ = 1


@doc """
The number of threads used by FFTW for CPU FFTs (default is the environment
variable `FFTW_NUM_THREADS`, or if that is not specified its
`Sys.CPU_THREADS÷2`). This must be set before creating any `FlatField` objects.
"""
FFTW_NUM_THREADS = nothing
@init global FFTW_NUM_THREADS = parse(Int,get(ENV,"FFTW_NUM_THREADS","$(Sys.CPU_THREADS÷2)"))


@doc """
Time-limit for FFT planning on CPU (default: 5 seconds). This must be set before
creating any `FlatField` objects.
"""
FFTW_TIMELIMIT = 5


@kwdef struct ProjLambert{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, FFT} <: FlatProj
    θpix
    Δx       :: T
    Ny       :: Int
    Nx       :: Int
    # Nbatch   :: Int
    Ωpix     :: T
    nyq      :: T
    Δℓx      :: T
    Δℓy      :: T
    ℓy       :: V
    ℓx       :: V
    ℓmag     :: M
    sin2ϕ    :: M
    cos2ϕ    :: M
    fft_plan :: FFT
    units = 1
    θϕ_center = nothing
end


@memoize function ProjLambert(T, ArrType, θpix, Ny, Nx, Nbatch)

    FFTW.set_num_threads(FFTW_NUM_THREADS)

    Δx           = T(deg2rad(θpix/60))
    fft_plan     = plan_rfft(ArrType{T}(undef,Ny,Nx,(Nbatch==1 ? () : (Nbatch,))...), (1,2); (ArrType <: Array ? (timelimit=FFTW_TIMELIMIT,) : ())...)
    Δℓx          = T(2π/(Nx*Δx))
    Δℓy          = T(2π/(Ny*Δx))
    nyq          = T(2π/(2Δx))
    Ωpix         = T(Δx^2)
    ℓy           = (ifftshift(-Ny÷2:(Ny-1)÷2) .* Δℓy)[1:Ny÷2+1]
    ℓx           = (ifftshift(-Nx÷2:(Nx-1)÷2) .* Δℓx)
    ℓmag         = @. sqrt(ℓx'^2 + ℓy^2)
    ϕ            = @. angle(ℓx' + im*ℓy)
    sin2ϕ, cos2ϕ = @. sin(2ϕ), cos(2ϕ)
    if iseven(Ny)
        sin2ϕ[end, end:-1:(Nx÷2+2)] .= sin2ϕ[end, 2:Nx÷2]
    end
    
    ProjLambert(;θpix, Nx, Ny, Δx, Δℓx, Δℓy, nyq, Ωpix, ℓx, ℓy, ℓmag, sin2ϕ, cos2ϕ, fft_plan)

end


getproperty(f::FlatMap, ::Val{:Ix}) = getfield(f,:arr)
getproperty(f::FlatFourier, ::Val{:Il}) = getfield(f,:arr)

FlatMap(Ix::A; θpix=1) where {T, A<:Array{T}} = FlatMap(Ix, ProjLambert(T, basetype(A), θpix, size(Ix)[1:2]..., size(Ix,3)))
FlatFourier(Il::A; θpix=1) where {T, A<:Array{T}} = FlatMap(Ix, ProjLambert(real(T), basetype(A), θpix, size(Ix)[1:2]..., size(Ix,3)))


# out-of-place basis conversions
Fourier(f::FlatMap) = FlatFourier(f.fft_plan * f.Ix, f.metadata)
Map(f::FlatFourier) = FlatMap(f.fft_plan \ f.Il, f.metadata)
# in-place conversion
Fourier(f′::FlatFourier, f::FlatMap) =  (mul!(f′.Il, f.fft_plan, f.Ix); f′)
Map(f′::FlatMap, f::FlatFourier)     = (ldiv!(f′.Ix, f.fft_plan, maybecopy(f.Il)); f′)
# need this for FFTW (but not MKL) see https://github.com/JuliaMath/FFTW.jl/issues/158
@static FFTW.fftw_vendor==:fftw ? maybecopy(x) = copy(x) : maybecopy(x) = x





@show_datatype Base.show_datatype(io::IO, t::Type{F}) where {G,M,T,A,F<:FlatField{G,M,T,A}} =
    print(io, "$(pretty_type_name(F)){$A,$(M.name.name)}")
for F in (:FlatMap, :FlatFourier)
    @eval pretty_type_name(::Type{<:$F}) = $(string(F))
end
function Base.summary(io::IO, f::FlatField)
    @unpack Nx,Ny,θpix = f
    Nbatch = size(f.arr, 3)
    print(io, "$(length(f))-element [$Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)") map, $(θpix)′ pixels] ")
    Base.showarg(io, f, true)
end


function promote_b_metadata(
    (b, metadata) ::Tuple{B,<:ProjLambert{T}}, 
    (b′,metadata′)::Tuple{B,<:ProjLambert{T′}}
) where {B,T,T′}

    if metadata===metadata′
        return (b, metadata)
    elseif @show(metadata.θpix==metadata′.θpix && metadata.Ny==metadata′.Ny && metadata.Nx==metadata′.Nx)
        Tres = promote_type(T,T′)
        if Tres==T
            return (b, metadata)
        elseif Tres==T′
            return (b, metadata′)
        end
    end
    return Unknown()

end

function preprocess((g,metadata)::Tuple{<:Any,<:ProjLambert}, ∇d::∇diag)
    if ∇d.coord == 1
        broadcasted(*, ∇d.prefactor * im, metadata.ℓx')
    elseif ∇d.coord == 2
        broadcasted(*, ∇d.prefactor * im, metadata.ℓy)
    else
        error()
    end
end






# const FlatS0{P,T,M} = Union{FlatMap{P,T,M},FlatFourier{P,T,M}}

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
