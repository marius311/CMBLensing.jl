export Flat, FFTgrid, FlatIQUMap, get_Cℓ

# a flat sky pixelization with `Nside` pixels per side and pixels of width `Θpix` arcmins
abstract type Flat{Θpix,Nside} <: Pix end
Nside(::Type{P}) where {_,N,P<:Flat{_,N}} = N
Θpix₀ = 1 # default angular resolution used by a number of convenience constructors


struct FFTgrid{dm, T, F}
    period::T
    nside::Int64
    Δx::T
    Δℓ::T
    nyq::T
    x::Vector{T}
    k::Vector{T}
    r::Array{T,dm}
    sincos2ϕ::Tuple{Array{T,dm},Array{T,dm}}
    FFT::F
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
    FFT = T((Δx/√(2π))^dm) * plan_rfft(Array{T}(fill(nside,dm)...); flags=flags, timelimit=timelimit)
    Δℓ  = 2π/period
    nyq = 2π/(2Δx)
    x,k = (ifftshift(-nside÷2:(nside-1)÷2),) .* [Δx,Δℓ]'
    r   = sqrt.(.+((reshape(k.^2, (s=ones(Int,dm); s[i]=nside; tuple(s...))) for i=1:dm)...))
    ϕ   = angle.(k' .+ im*k)[1:nside÷2+1,:]
    sincos2ϕ = @. sin(2ϕ), cos(2ϕ)
    FFTgrid(period, nside, Δx, Δℓ, nyq, x, k, r, sincos2ϕ, FFT)
end

# Use generated functions to get planned FFT's only once for any given (T, Θpix,
# Nside) combination
@generated function FFTgrid(::Type{T},::Type{P}) where {Θpix, Nside, T<:Real,P<:Flat{Θpix, Nside}}
    FFTgrid(T, deg2rad(Θpix/60)*Nside, Nside)
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


Cℓ_2D(ℓ, Cℓ, r) = extrapolate(interpolate((ℓ,),Cℓ,Gridded(Linear())),0)[r]
Cℓ_2D(::Type{P}, ℓ, Cℓ) where {N,P<:Flat{<:Any,N}} = Cℓ_2D(ℓ,Cℓ,FFTgrid(Float64,P).r)[1:N÷2+1,:]
Cℓ_to_cov(::Type{P}, ::Type{S}, args::Vector{T}...) where {T,P,S<:Spin} = Cℓ_to_cov(T,P,S,args...)

""" filter out the single row/column in the real FFT matrix `M` which
corresponds to exactly the nyquist frequency """
function Mnyq(::Type{T},::Type{P}, M) where {T,θ,N,P<:Flat{θ,N}}
    if iseven(N)
        inyq = first(indexin([-FFTgrid(T,P).nyq],FFTgrid(T,P).k))
        M[inyq,:] = M[:,inyq] = 0
    end
    M
end


include("flat_s0.jl")
include("flat_s2.jl")
include("flat_s0s2.jl")

const FlatFourier{T,P} = Union{FlatS0Fourier{T,P},FlatS2Fourier{T,P},FieldTuple{<:FlatS0Fourier{T,P},<:FlatS2Fourier{T,P}}}
const FlatMap{T,P} = Union{FlatS0Map{T,P},FlatS2Map{T,P},FieldTuple{<:Tuple{FlatS0Map{T,P},FlatS2Map{T,P}}}}
const FlatField{T,P} = Union{FlatS0{T,P},FlatS2{T,P},FlatS02{T,P}}

FFTgrid(::FlatField{T,P}) where {T,P} = FFTgrid(T,P)

# generic eltype
eltype(::Type{<:FlatField{T}}) where {T} = T

# we can broadcast a S0 field with an S2 one by just replicating the S0 part twice
@commutative promote_containertype{F0<:FlatS0Map,F2<:FlatS2Map}(::Type{F0},::Type{F2}) = F2
@commutative promote_containertype{F0<:FlatS0Fourier,F2<:FlatS2Fourier}(::Type{F0},::Type{F2}) = F2
broadcast_data(::Type{F2}, f::F0) where {F2<:FlatS2Map, F0<:FlatS0Map} = tuplejoin(repeated(broadcast_data(F0,f),2)...)
broadcast_data(::Type{F2}, f::F0) where {F2<:FlatS2Fourier, F0<:FlatS0Fourier} = tuplejoin(repeated(broadcast_data(F0,f),2)...)
@commutative *(a::FlatS0Map, b::FlatS2Map) = a.*b


# derivatives
DerivBasis(::Type{<:FlatS0}) = Fourier
DerivBasis(::Type{<:FlatS2}) = QUFourier
@generated broadcast_data(::Type{<:FlatFourier{T,P}},::∂{:x}) where {T,P} = (im * FFTgrid(T,P).k',)
@generated broadcast_data(::Type{<:FlatFourier{T,P}},::∂{:y}) where {T,P} = (im * FFTgrid(T,P).k[1:Nside(P)÷2+1],)
@generated broadcast_data(::Type{<:FlatFourier{T,P}},::∇²Op) where {T,P} = ((@. -FFTgrid(T,P).r[1:Nside(P)÷2+1,:]^2),)

# bandpass
broadcast_data(::Type{F}, op::BandPassOp) where {T,P,F<:FlatFourier{T,P}} =
    (Cℓ_2D(op.ℓ,op.Wℓ,FFTgrid(T,P).r)[1:Nside(P)÷2+1,:],)
