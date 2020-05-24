
# derivatives can either be applied in fourier space by multipliying by im*k or
# in map space by finite differences. this tracks which option to use for a
# given field
abstract type ∂modes end
struct fourier∂ <: ∂modes end
struct map∂ <: ∂modes end
promote_rule(::Type{map∂}, ::Type{fourier∂}) = fourier∂

# Flat{Nside,θpix,∂mode} is a flat sky pixelization with `Nside` pixels per side
# and pixels of width `θpix` arcmins, where derivatives are done according to ∂mode
abstract type Flat{Nside,θpix,∂mode<:∂modes,D} <: Pix end

# abstract type for the kind of data that can be stored in FlatFields
const AbstractRank2or3Array{T} = Union{AbstractArray{T,2},AbstractArray{T,3}}

# for convenience
Flat(;Nside, θpix=θpix₀, ∂mode=fourier∂, D=1) = Flat{Nside,θpix,∂mode,D}
Nside(::Type{P}) where {N,P<:Flat{N}} = N

# default angular resolution used by a number of convenience constructors
θpix₀ = 1


@doc """
The number of FFTW threads to use. This must be set via e.g.:

    CMBLensing.FFTW_NUM_THREADS[] = 4

*before* creating any `FlatField` objects; subsequent changes to this variable
will be ignored. The default value is the environment variable `FFTW_NUM_THREADS`,
or if that is not specified its `Sys.CPU_THREADS÷2`.
"""
const FFTW_NUM_THREADS = Ref{Int}()
@init FFTW_NUM_THREADS[] = parse(Int,get(ENV,"FFTW_NUM_THREADS","$(Sys.CPU_THREADS÷2)"))


@generated function FlatInfo(::Type{T}, ::Type{Arr}, ::Val{θpix}, ::Val{Nside}, ::Val{D}) where {T<:Real, Arr<:AbstractArray, θpix, Nside, D}

    FFTW.set_num_threads(FFTW_NUM_THREADS[])

    Δx   = T(deg2rad(θpix/60))
    FFT  = plan_rfft(Arr{T}(undef,Nside,Nside,(D==1 ? () : (D,))...), (1,2))
    Δℓ   = T(2π/(Nside*Δx))
    nyq  = T(2π/(2Δx))
    Ωpix = T(Δx^2)
    x,k  = (ifftshift(-Nside÷2:(Nside-1)÷2),) .* (Δx,Δℓ)
    kmag = @. sqrt(k'^2 + k^2)
    ϕ    = @. angle(k' + im*k)[1:Nside÷2+1,:]
    sin2ϕ, cos2ϕ = @. sin(2ϕ), cos(2ϕ)
    if iseven(Nside)
        sin2ϕ[end, end:-1:(Nside÷2+2)] .= sin2ϕ[end, 2:Nside÷2]
    end
    
    @namedtuple(T, θpix, Nside, Δx, Δℓ, nyq, Ωpix, x, k, kmag, sin2ϕ=Arr(sin2ϕ), cos2ϕ=Arr(cos2ϕ), FFT)

end

function Cℓ_to_2D(::Type{P}, ::Type{T}, Cℓ) where {T,N,P<:Flat{N}}
    Complex{T}.(nan2zero.(Cℓ.(fieldinfo(P,T).kmag[1:N÷2+1,:])))
end


@doc doc"""
    pixwin(θpix, ℓ)

Returns the pixel window function for square flat-sky pixels of width `θpix` (in
arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
the power spectrum will be pixwin^2. 
"""
pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)
