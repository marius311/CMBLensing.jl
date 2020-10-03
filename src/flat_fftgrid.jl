
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


function FlatInfo(T, Arr, θpix, Nside, D)

    FFTW.set_num_threads(FFTW_NUM_THREADS)

    Nx, Ny = Nside .* (1,1)
    Δx   = T(deg2rad(θpix/60))
    FFT  = plan_rfft(Arr{T}(undef,Ny,Nx,(D==1 ? () : (D,))...), (1,2); (Arr <: Array ? (timelimit=FFTW_TIMELIMIT,) : ())...)
    Δℓx  = T(2π/(Nx*Δx))
    Δℓy  = T(2π/(Ny*Δx))
    nyq  = T(2π/(2Δx))
    Ωpix = T(Δx^2)
    ky   = ifftshift(-Ny÷2:(Ny-1)÷2) .* Δℓy
    kx   = ifftshift(-Nx÷2:(Nx-1)÷2) .* Δℓx
    kmag = @. sqrt(kx'^2 + ky^2)
    ϕ    = @. angle(kx' + im*ky)[1:Ny÷2+1,:]
    sin2ϕ, cos2ϕ = @. sin(2ϕ), cos(2ϕ)
    if iseven(Ny)
        sin2ϕ[end, end:-1:(Nx÷2+2)] .= sin2ϕ[end, 2:Nx÷2]
    end
    
    @namedtuple(T, θpix, Nx, Ny, Nside, Δx, Δℓx, Δℓy, nyq, Ωpix, kx, ky, kmag, sin2ϕ=Arr(sin2ϕ), cos2ϕ=Arr(cos2ϕ), FFT)

end

function Cℓ_to_2D(::Type{P}, ::Type{T}, Cℓ) where {T,N,P<:Flat{N}}
    Complex{T}.(nan2zero.(Cℓ.(fieldinfo(P,T).kmag[1:fieldinfo(P,T).Ny÷2+1,:])))
end


@doc doc"""
    pixwin(θpix, ℓ)

Returns the pixel window function for square flat-sky pixels of width `θpix` (in
arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
the power spectrum will be pixwin^2. 
"""
pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)
