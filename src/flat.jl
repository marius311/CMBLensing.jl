export Flat, FFTgrid

# a flat sky pixelization with `Nside` pixels per side and pixels of width `Θpix` arcmins 
abstract Flat{Θpix,Nside} <: Pix
Nside{P<:Flat}(::Type{P}) = P.parameters[2] #convenience method, will look less hacky in 0.6
Θpix₀ = 1 # default angular resolution used by a number of convenience constructors


immutable FFTgrid{dm, T}
    period::T
    nside::Int64
    Δx::T
    Δℓ::T
    nyq::T
    x::Array{T,1}
    k::Array{T,1}
    r::Array{T,dm}
    sincos2ϕ::Tuple{Array{T,dm},Array{T,dm}}
    FFT::FFTW.ScaledPlan{T,FFTW.rFFTWPlan{T,-1,false,dm},T}
end


"""
# Arguments
* `T`: 
* `period::Real`: the length of one side of the map
* `nside::Int`: the number of pixels in one side of the map 
* `dm::Integer=2`: the number of dimensions (i.e. 2 for map)
"""
function FFTgrid{T<:Real}(::Type{T}, period, nside, dm=2; flags=FFTW.ESTIMATE, timelimit=5)
    Δx  = period/nside
    FFTW.set_num_threads(Sys.CPU_CORES)
    FFT = (Δx/√(2π))^dm * plan_rfft(rand(T,fill(nside,dm)...); flags=flags, timelimit=timelimit)
    Δℓ  = 2π/period
    nyq = 2π/(2Δx)
    x,k = getxkside(Δx,Δℓ,period,nside)
    r   = sqrt.(.+((reshape(k.^2, (s=ones(Int,dm); s[i]=nside; tuple(s...))) for i=1:dm)...))
    ϕ   = angle.(k' .+ im*k)[1:nside÷2+1,:]
    sincos2ϕ = sin(2ϕ), cos(2ϕ)
    FFTgrid{dm,T}(period, nside, Δx, Δℓ, nyq, x, k, r, sincos2ϕ, FFT)
end

function getxkside(Δx,Δℓ,period,nside)
    x, ℓ = zeros(nside), zeros(nside)
    for j in 0:(nside-1)
        x[j+1] = ((j <= (nside-j)) ? j : j-nside)*Δx
        ℓ[j+1] = ((j <= (nside-j)) ? j : j-nside)*Δℓ
    end
    x, ℓ
end


# Use generated functions to get planned FFT's only once for any given (T, Θpix,
# Nside) combination
@generated function FFTgrid{T<:Real,P<:Flat}(::Type{T},::Type{P})
    Θpix, Nside = P.parameters
    FFTgrid(T, deg2rad(Θpix/60)*Nside, Nside)
end

abstract ℱ{P}

*{T,P}(::Type{ℱ{P}},x::Matrix{T}) = FFTgrid(T,P).FFT * x
\{T,P}(::Type{ℱ{P}},x::Matrix{Complex{T}}) = FFTgrid(T,P).FFT \ x


# Check map and fourier coefficient arrays are the right size
function checkmap{T,P}(::Type{P},A::Matrix{T})
    @assert ==(Nside(P),size(A)...) "Wrong size for a map."
    A
end
function checkfourier{T,P}(::Type{P},A::Matrix{Complex{T}})
    n,m = size(A)
    @assert m==Nside(P) && n==Nside(P)÷2+1 "Wrong size for a fourier transform."
    #todo: check symmetries
    A
end
