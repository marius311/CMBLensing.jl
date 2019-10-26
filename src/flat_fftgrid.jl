
# derivatives can either be applied in fourier space by multipliying by im*k or
# in map space by finite differences. this tracks which option to use for a
# given field
abstract type ∂modes end
struct fourier∂ <: ∂modes end
struct map∂ <: ∂modes end
promote_rule(::Type{map∂}, ::Type{fourier∂}) = fourier∂

# Flat{Nside,θpix,∂mode} is a flat sky pixelization with `Nside` pixels per side
# and pixels of width `θpix` arcmins, where derivatives are done according to ∂mode
abstract type Flat{Nside,θpix,∂mode<:∂modes} <: Pix end

# for convenience
Flat(;Nside, θpix=θpix₀, ∂mode=fourier∂) = Flat{Nside,θpix,∂mode}
Nside(::Type{P}) where {N,P<:Flat{N}} = N

# default angular resolution used by a number of convenience constructors
θpix₀ = 1

# stores FFT plan and other info needed for manipulating a Flat map
struct FFTgrid{T,F}
    θpix :: T
    Nside :: Int64
    Δx :: T
    Δℓ :: T
    nyq :: T
    x :: Vector{T}
    k :: Vector{T}
    r :: Matrix{T}
    sin2ϕ :: Matrix{T}
    cos2ϕ :: Matrix{T}
    FFT :: F
end

@doc """
The number of FFTW threads to use. This must be set via e.g.:

    CMBLensing.FFTW_NUM_THREADS[] = 4

*before* creating any `FlatField` objects; subsequent changes to this variable
will be ignored. The default value is `Sys.CPU_THREADS`.
"""
const FFTW_NUM_THREADS = Ref{Int}()
@init FFTW_NUM_THREADS[] = Sys.CPU_THREADS
 
# use @generated function to memoize FFTgrid for given (T,θ,Nside) combinations
FFTgrid(::Type{<:Flat{Nside,θpix}}, ::Type{T}) where {T, θpix, Nside} = FFTgrid(T, Val(θpix), Val(Nside))
@generated function FFTgrid(::Type{T}, ::Val{θpix}, ::Val{Nside}) where {T<:Real, θpix, Nside}
    Δx  = deg2rad(θpix/60)
    FFTW.set_num_threads(FFTW_NUM_THREADS[])
    FFT = plan_rfft(Matrix{T}(undef,Nside,Nside); flags=FFTW.ESTIMATE, timelimit=5)
    Δℓ  = 2π/(Nside*Δx)
    nyq = 2π/(2Δx)
    x,k = (ifftshift(-Nside÷2:(Nside-1)÷2),) .* [Δx,Δℓ]'
    r   = @. sqrt(k'^2 + k^2)
    ϕ   = @. angle(k' + im*k)[1:Nside÷2+1,:]
    sin2ϕ, cos2ϕ = @. sin(2ϕ), cos(2ϕ)
    if iseven(Nside)
        sin2ϕ[end, end:-1:(Nside÷2+2)] .= sin2ϕ[end, 2:Nside÷2]
    end
    FFTgrid{T,typeof(FFT)}(θpix, Nside, Δx, Δℓ, nyq, x, k, r, sin2ϕ, cos2ϕ, FFT)
end

function Cℓ_to_2D(::Type{P}, ::Type{T}, Cℓ) where {T,N,P<:Flat{N}}
    Complex{T}.(nan2zero.(Cℓ.(FFTgrid(P,T).r[1:N÷2+1,:])))
end


@doc doc"""
    pixwin(θpix, ℓ)

Returns the pixel window function for square flat-sky pixels of width `θpix` (in
arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
the power spectrum will be pixwin^2. 
"""
pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)
