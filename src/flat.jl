export Flat, FFTgrid, FlatIQUMap, get_Cℓ

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
    FFT = T((Δx/√(2π))^dm) * plan_rfft(rand(T,fill(nside,dm)...); flags=flags, timelimit=timelimit)
    Δℓ  = 2π/period
    nyq = 2π/(2Δx)
    x,k = (ifftshift(-nside÷2:(nside-1)÷2),) .* [Δx,Δℓ]'
    r   = sqrt.(.+((reshape(k.^2, (s=ones(Int,dm); s[i]=nside; tuple(s...))) for i=1:dm)...))
    ϕ   = angle.(k' .+ im*k)[1:nside÷2+1,:]
    sincos2ϕ = @. sin(2ϕ), cos(2ϕ)
    FFTgrid{dm,T}(period, nside, Δx, Δℓ, nyq, x, k, r, sincos2ϕ, FFT)
end

# Use generated functions to get planned FFT's only once for any given (T, Θpix,
# Nside) combination
@generated function FFTgrid(::Type{T},::Type{P}) where {Θpix, Nside, T<:Real,P<:Flat{Θpix, Nside}}
    FFTgrid(T, deg2rad(Θpix/60)*Nside, Nside)
end

""" Converts an (N÷2+1,N) fourier transform matrix to the full (N,N) one via symmetries """
function unfold(Tl::AbstractMatrix{Complex{T}}) where {T}
    m,n = size(Tl)
    @assert iseven(n) && m==n÷2+1
    Tlu = Array{Complex{T}}(n,n)
    Tlu[1:m,1:n] = Tl
    @inbounds for i=m+1:n
        Tlu[i,1] = Tl[2m-i, 1]'
        @simd for j=2:n
            Tlu[i,j] = Tl[2m-i, 2m-j]'
        end
    end
    Tlu
end


abstract type ℱ{P} end

*{T,P}(::Type{ℱ{P}},x::Matrix{T}) = FFTgrid(T,P).FFT * x
\{T,P}(::Type{ℱ{P}},x::Matrix{Complex{T}}) = FFTgrid(T,P).FFT \ x


# Check map and fourier coefficient arrays are the right size
function checkmap{T,P}(::Type{P},A::AbstractMatrix{T})
    @assert ==(Nside(P),size(A)...) "Wrong size for a map."
    A
end
checkfourier{T<:Real,P}(::Type{P},A::AbstractMatrix{T}) = checkfourier(P,complex(A))
function checkfourier{T,P}(::Type{P},A::AbstractMatrix{Complex{T}})
    n,m = size(A)
    @assert m==Nside(P) && n==Nside(P)÷2+1 "Wrong size for a fourier transform."
    #todo: check symmetries
    A
end

Cℓ_to_cov(::Type{P}, ::Type{S}, args::Vector{T}...) where {T,P,S<:Spin} = Cℓ_to_cov(T,P,S,args...)


include("flat_s0.jl")
include("flat_s2.jl")

const FlatMap{T,P} = Union{FlatS0Map{T,P},FlatS2Map{T,P}}
const FlatFourier{T,P} = Union{FlatS0Fourier{T,P},FlatS2Fourier{T,P}}

# generic f[:] that works for both fields (each must define its own fromvec)
@generated function getindex(f::Union{FlatS0,FlatS2},::Colon) 
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
DerivBasis(::Type{<:FlatS0}) = Fourier
DerivBasis(::Type{<:FlatS2}) = QUFourier
for F in (FlatS0Fourier,FlatS2QUFourier,FlatS2EBFourier)
    @eval broadcast_data(::Type{$F{T,P}},::∂{:x}) where {T,P} = repeated(im * FFTgrid(T,P).k',$(broadcast_length(F)))
    @eval broadcast_data(::Type{$F{T,P}},::∂{:y}) where {T,P} = repeated(im * FFTgrid(T,P).k[1:Nside(P)÷2+1],$(broadcast_length(F)))
end


const FlatIQUMap{T,P} = Field2Tuple{FlatS0Map{T,P},FlatS2QUMap{T,P}}

# todo: make this actually take into account TE
function Cℓ_to_cov{T,P}(::Type{T}, ::Type{P}, ::Type{S0}, ::Type{S2}, ℓ, CℓTT, CℓTE, CℓEE, CℓBB)
    FullDiagOp(FieldTuple(Cℓ_to_cov(T,P,S0,ℓ,CℓTT).f, Cℓ_to_cov(T,P,S2,ℓ,CℓEE,CℓBB).f))
end
