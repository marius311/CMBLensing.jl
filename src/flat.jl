export Flat, FFTgrid, FlatIQUMap, get_Cℓ

# a flat sky pixelization with `Nside` pixels per side and pixels of width `Θpix` arcmins
abstract type ∂modes end
struct fourier∂ <: ∂modes end
struct map∂ <: ∂modes end
abstract type Flat{Θpix,Nside,∂mode} <: Pix end
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
function FFTgrid(::Type{T}, period, nside, dm=2; flags=FFTW.ESTIMATE, timelimit=5) where {T<:Real}
    Δx  = period/nside
    FFTW.set_num_threads(Sys.CPU_THREADS)
    FFT = T((Δx/√(2π))^dm) * plan_rfft(Array{T}(undef,fill(nside,dm)...); flags=flags, timelimit=timelimit)
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

*(::Type{ℱ{P}},x::Matrix{T}) where {T,P} = FFTgrid(T,P).FFT * x
\(::Type{ℱ{P}},x::Matrix{Complex{T}}) where {T,P} = FFTgrid(T,P).FFT \ x


# Check map and fourier coefficient arrays are the right size
function checkmap(::Type{P},A::AbstractMatrix{T}) where {T,P}
    @assert ==(Nside(P),size(A)...) "Wrong size for a map."
    A
end
checkfourier(::Type{P},A::AbstractMatrix{T}) where {T<:Real,P} = checkfourier(P,complex(A))
function checkfourier(::Type{P},A::AbstractMatrix{Complex{T}}) where {T,P}
    n,m = size(A)
    @assert m==Nside(P) && n==Nside(P)÷2+1 "Wrong size for a fourier transform."
    #could check symmetries here?
    A
end


Cℓ_2D(ℓ, Cℓ, r) = LinearInterpolation(ℓ, Cℓ, extrapolation_bc = 0).(r)
Cℓ_2D(::Type{P}, ℓ, Cℓ) where {N,P<:Flat{<:Any,N}} = Cℓ_2D(ℓ,Cℓ,FFTgrid(Float64,P).r)[1:N÷2+1,:]
Cℓ_to_cov(::Type{P}, ::Type{S}, args::Vector{T}...) where {T,P,S<:Spin} = Cℓ_to_cov(T,P,S,args...)

""" filter out the single row/column in the real FFT matrix `M` which
corresponds to exactly the nyquist frequency """
function Mnyq(::Type{T},::Type{P}, M) where {T,θ,N,P<:Flat{θ,N}}
    if iseven(N)
        inyq = first((1:N)[@. FFTgrid(T,P).k ≈ -FFTgrid(T,P).nyq])
        M[inyq,:] = M[:,inyq] = 0
    end
    M
end

@doc doc"""
    pixwin(θpix, ℓ)

Returns the pixel window function for square flat-sky pixels of width `θpix` (in
arcmin) evaluated at some `ℓ`s. 

The pixel window function is defined so that if you start with white noise at
infinitely high resolution and pixelize it down a resolution `θpix`, its power
spectrum will be given by pixwin(θpix, ℓ)^2. 
"""
pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)

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
# @commutative promote_containertype{F0<:FlatS0Map,F2<:FlatS2Map}(::Type{F0},::Type{F2}) = F2
# @commutative promote_containertype{F0<:FlatS0Fourier,F2<:FlatS2Fourier}(::Type{F0},::Type{F2}) = F2
broadcast_data(::Type{F2}, f::F0) where {F2<:FlatS2Map, F0<:FlatS0Map} = (broadcast_data(F0,f),)
*(f0::FlatS0Map, f2::FlatS2Map) = f0 .* f2
*(f2::FlatS2Map, f0::FlatS2Map) = f0 .* f2


HarmonicBasis(::Type{<:FlatS0}) = Fourier

# repesentation of ∇ for FlatFields
*(::∇Op, f::FlatField) = @SVector[∂x*f, ∂y*f]
*(::AdjOp{∇Op}, v::SVector{<:Any,<:FlatField}) = @SVector[∂x,∂y]' * v
*(::AdjOp{∇Op}, v::SMatrix{<:Any,<:Any,<:FlatField}) = (@SMatrix[∂x ∂y] * v)[1,:]'

# fourier space derivatives
DerivBasis(::Type{<:FlatS0{T,Flat{θ,N,fourier∂}}}) where {T,θ,N} = Fourier
DerivBasis(::Type{<:FlatS2{T,Flat{θ,N,fourier∂}}}) where {T,θ,N} = QUFourier
@generated broadcast_data(::Type{<:FlatFourier{T,P}},::∂{:x}) where {T,P} = (im * FFTgrid(T,P).k',)
@generated broadcast_data(::Type{<:FlatFourier{T,P}},::∂{:y}) where {T,P} = (im * FFTgrid(T,P).k[1:Nside(P)÷2+1],)
@generated broadcast_data(::Type{<:FlatFourier{T,P}},::∇²Op) where {T,P} = ((@. -FFTgrid(T,P).r[1:Nside(P)÷2+1,:]^2),)
*(L::Union{∂,∇²Op}, f::FlatS0Fourier{T,<:Flat{θ,N,<:fourier∂}}) where {T,θ,N} = L .* f

# map space derivatives
DerivBasis(::Type{<:FlatS0{T,Flat{θ,N,map∂}}}) where {T,θ,N} = Map
DerivBasis(::Type{<:FlatS2{T,Flat{θ,N,map∂}}}) where {T,θ,N} = QUMap
function *(::∂{s}, f::FlatS0Map{T,<:Flat{θ,N,<:map∂}}) where {s,T,θ,N}
    f′ = similar(f)
    n,m = size(f.Tx)
    @unpack Δx = FFTgrid(f)
    if s==:x
        @inbounds for j=2:m-1
            @simd for i=1:n
                f′.Tx[i,j] = (f.Tx[i,j+1] - f.Tx[i,j-1])/2Δx
            end
        end
        @inbounds for i=1:n
            f′.Tx[i,1] = (f.Tx[i,2]-f.Tx[i,end])/2Δx
            f′.Tx[i,end] = (f.Tx[i,1]-f.Tx[i,end-1])/2Δx
        end
    else
        @inbounds for j=1:n
            @simd for i=2:m-1
                f′.Tx[i,j] = (f.Tx[i+1,j] - f.Tx[i-1,j])/2Δx
            end
            f′.Tx[1,j] = (f.Tx[2,j]-f.Tx[end,j])/2Δx
            f′.Tx[end,j] = (f.Tx[1,j]-f.Tx[end-1,j])/2Δx
        end
    end
    f′
end


# bandpass
broadcast_data(::Type{F}, op::BandPassOp) where {T,P,F<:FlatFourier{T,P}} =
    (Cℓ_2D(op.ℓ,op.Wℓ,FFTgrid(T,P).r)[1:Nside(P)÷2+1,:],)
