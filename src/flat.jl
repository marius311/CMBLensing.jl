export Flat, FFTgrid

# a flat sky pixelization with `Nside` pixels per side and pixels of width `Θpix` arcmins 
abstract type Flat{Θpix,Nside} <: Pix end
Nside(::Type{P}) where {_,N,P<:Flat{_,N}} = N
Θpix₀ = 1 # default angular resolution used by a number of convenience constructors


struct FFTgrid{dm, T}
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
    sincos2ϕ = sin.(2ϕ), cos.(2ϕ)
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

abstract type ℱ{P} end

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


include("flat_s0.jl")
include("flat_s2.jl")

const FlatMap{T,P} = Union{FlatS0Map{T,P},FlatS2Map{T,P}}
const FlatFourier{T,P} = Union{FlatS0Fourier{T,P},FlatS2Fourier{T,P}}

# generic tovec works for all fields (each must define its own fromvec)
@generated function tovec(f::Union{FlatS0,FlatS2}) 
    :(vcat($((:(f.$x[:]) for x in fieldnames(f))...)))
end

# generic eltype
eltype(::Type{F}) where {T,P,F<:FlatMap{T,P}} = T
eltype(::Type{F}) where {T,P,F<:FlatFourier{T,P}} = Complex{T}

# we can broadcast a S0 field with an S2 one by just replicating the S0 part twice
@swappable promote_containertype{F0<:FlatS0Map,F2<:FlatS2Map}(::Type{F0},::Type{F2}) = F2
@swappable promote_containertype{F0<:FlatS0Fourier,F2<:FlatS2Fourier}(::Type{F0},::Type{F2}) = F2
broadcast_data(::Type{F2}, f::F0) where {F2<:FlatS2Map, F0<:FlatS0Map} = repeated(broadcast_data(F0,f)...,2)
broadcast_data(::Type{F2}, f::F0) where {F2<:FlatS2Fourier, F0<:FlatS0Fourier} = repeated(broadcast_data(F0,f)...,2)
@swappable *(a::FlatS0Map, b::FlatS2Map) = a.*b


# derivatives
∂Basis(::Type{<:FlatS0}) = Fourier
∂Basis(::Type{<:FlatS2}) = QUFourier
for F in (FlatS0Fourier,FlatS2QUFourier)
    @eval broadcast_data(::Type{$F{T,P}},::∂{:x}) where {T,P} = repeated(im * FFTgrid(T,P).k')
    @eval broadcast_data(::Type{$F{T,P}},::∂{:y}) where {T,P} = repeated(im * FFTgrid(T,P).k[1:Nside(P)÷2+1])
end
