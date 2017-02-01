
# a flat sky pixelization with `Nside` pixels per side and pixels of width `Θpix` arcmins 
abstract Flat{Θpix,Nside} <: Pix
Nside{P<:Flat}(::Type{P}) = P.parameters[2] #convenience method, will look less hacky in 0.6

immutable FFTgrid{dm, T}
    period::T
    nside::Int64
    Δx::T
    Δk::T
    nyq::T
    x::Array{T,1}
    k::Array{T,1}
    r::Array{T,dm}
    sincosϕ::Tuple{Array{T,dm},Array{T,dm}}
    FFT::FFTW.ScaledPlan{T,FFTW.rFFTWPlan{T,-1,false,dm},T}
end

function FFTgrid{T<:Real}(::Type{T}, dm, period, nside; flags=FFTW.ESTIMATE, timelimit=5)
    Δx  = period/nside
    Δk  = 2π/period
    nyq = 2π/(2Δx)
    x,k = getxkside(Δx,Δk,period,nside)
    r = sqrt.(.+((reshape(k.^2, (s=ones(Int,dm); s[i]=nside; tuple(s...))) for i=1:dm)...)) # end
    ϕ2_l   = 2angle.(k .+ im*k')
    sincosϕ = sin(ϕ2_l), cos(ϕ2_l)
    FFT = (Δx/√(2π))^dm * plan_rfft(rand(T,fill(nside,dm)...); flags=flags, timelimit=timelimit)
    FFTgrid{dm,T}(period, nside, Δx, Δk, nyq, x, k, r, sincosϕ, FFT)
end

function getxkside(Δx,Δk,period,nside)
    x, k = zeros(nside), zeros(nside)
    for j in 0:(nside-1)
    x[j+1] = (j < nside/2) ? (j*Δx) : (j*Δx - period)
    k[j+1] = (j < nside/2) ? (j*Δk) : (j*Δk - 2π*nside/period)
    end
    x, k
end


# Use generated functions to get planned FFT's only once _for any given (T, Ωpix,
# Nside) combination
@generated function FFTgrid{T<:Real,P<:Flat}(::Type{T},::Type{P})
    Ωpix, Nside = P.parameters
    FFTgrid(T, 2, Ωpix*Nside*pi/(180*60), Nside)
end

abstract ℱ{P}

*{T,P}(::Type{ℱ{P}},x::Matrix{T}) = FFTgrid(T,P).FFT * x
\{T,P}(::Type{ℱ{P}},x::Matrix{Complex{T}}) = FFTgrid(T,P).FFT \ x
