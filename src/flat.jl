export Flat, FFTgrid, FlatIQUMap, get_Cℓ

# derivatives can either be applied in fourier space by multipliying by im*k or
# in map space by finite differences. this tracks which option to use for a
# given field
abstract type ∂modes end
struct fourier∂ <: ∂modes end
struct map∂ <: ∂modes end
promote_type(::Type{map∂}, ::Type{fourier∂}) = fourier∂

# Flat{θpix,Nside,∂mode} is a flat sky pixelization with `Nside` pixels per side
# and pixels of width `Θpix` arcmins, where derivatives are done according to ∂mode
abstract type Flat{Θpix,Nside,∂mode<:∂modes} <: Pix end

# for convenience
Nside(::Type{P}) where {_,N,P<:Flat{_,N}} = N

# default angular resolution used by a number of convenience constructors
θpix₀ = 1 

# stores FFT plan and other info needed for manipulating a Flat map
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

# some syntatic sugar for applying the FFT plans stored in FFTgrid
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

""" filter out the single row/column in the real FFT matrix `M` which
corresponds to exactly the nyquist frequency """
function Mnyq(::Type{T},::Type{P}, M) where {T,θ,N,P<:Flat{θ,N}}
    if iseven(N)
        inyq = first((1:N)[@. FFTgrid(T,P).k ≈ -FFTgrid(T,P).nyq])
        M[inyq,:] .= 0
        M[:,inyq] .= 0
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

# Some Unions to specify various S0, S2, or S02 fields
const FlatMap{T,P}     = Union{FlatS0Map{T,P},     FlatS2Map{T,P},     FieldTuple{<:Tuple{FlatS0Map{T,P},    FlatS2Map{T,P}}}}
const FlatFourier{T,P} = Union{FlatS0Fourier{T,P}, FlatS2Fourier{T,P}, FieldTuple{<:Tuple{FlatS0Fourier{T,P},FlatS2Fourier{T,P}}}}
const FlatField{T,P}   = Union{FlatS0{T,P},        FlatS2{T,P},        FlatS02{T,P}}

# "Base" fields are the fields which have they own non-FieldTuple concrete types,
# i.e. S0 and S2. Fields like S02 are still flat fields but they're not "Base"
# because they are implemented as a FieldTuple of (S0,S2). 
const BaseFlatMap{T,P}       = Union{FlatS0Map{T,P},     FlatS2Map{T,P}}
const BaseFlatFourier{T,P}   = Union{FlatS0Fourier{T,P}, FlatS2Fourier{T,P}}
const BaseFlatField{T,P}     = Union{FlatS0{T,P},        FlatS2{T,P}}

## promotion

function promote(f1::F1, f2::F2) where {T1,θ1,N1,∂mode1,F1<:BaseFlatField{T1,Flat{θ1,N1,∂mode1}},T2,θ2,N2,∂mode2,F2<:BaseFlatField{T2,Flat{θ2,N2,∂mode2}}}
    T     = promote_type(T1,T2)
    B     = promote_type(basis(F1),basis(F2))
    ∂mode = promote_type(∂mode1,∂mode2)
    B(T(∂mode(f1))), B(T(∂mode(f2)))
end


## conversion

# e.g. Float32(f::FlatField) or Float64(f::FlatField)
(::Type{T′})(f::F) where {T′<:Real,T,P,F<:BaseFlatMap{T,P}} = 
    basetype(F){T′,P}(convert.(Matrix{T′}, fieldvalues(f))...)
(::Type{T′})(f::F) where {T′<:Real,T,P,F<:BaseFlatFourier{T,P}} = 
    basetype(F){T′,P}(convert.(Matrix{Complex{T′}}, fieldvalues(f))...)
# map∂(f::FlatField) or fourier∂(f::FlatField)
(::Type{∂mode})(f::F) where {∂mode<:∂modes,T,θ,N,F<:BaseFlatField{T,<:Flat{θ,N}}} = 
    basetype(F){T,Flat{θ,N,∂mode}}(fieldvalues(f)...)


FFTgrid(::FlatField{T,P}) where {T,P} = FFTgrid(T,P)

eltype(::Type{<:FlatField{T}}) where {T} = T

broadcast_data(::Type{F2}, f::F0) where {F2<:FlatS2Map, F0<:FlatS0Map} = broadcast_data(F0,f)
*(f0::FlatS0Map, f2::FlatS2Map) = f0 .* f2


## derivatives

# fourier space derivatives
DerivBasis(::Type{<:FlatS0{T,Flat{θ,N,fourier∂}}}) where {T,θ,N} = Fourier
DerivBasis(::Type{<:FlatS2{T,Flat{θ,N,fourier∂}}}) where {T,θ,N} = QUFourier
@generated function broadcast_data(::Type{<:FlatFourier{T,P}}, ∇i::Union{∇i{coord},AdjOp{<:∇i{coord}}}) where {coord,T,P}
    α = ∇i isa AdjOp ? -im : im
    if coord==0
        (α * FFTgrid(T,P).k',)
    elseif coord==1
        (α * FFTgrid(T,P).k[1:Nside(P)÷2+1],)
    end
end
@generated function broadcast_data(::Type{<:FlatFourier{T,P}}, ::∇²Op) where {coord,T,P}
    (FFTgrid(T,P).k' .^2 .+ FFTgrid(T,P).k[1:Nside(P)÷2+1].^2,)
end
mul!( f′::F, ∇i::Union{∇i,AdjOp{<:∇i}}, f::F) where {T,θ,N,F<:FlatFourier{T,<:Flat{θ,N,<:fourier∂}}} = @. f′ = ∇i * f
ldiv!(f′::F, ∇i::Union{∇i,AdjOp{<:∇i}}, f::F) where {T,θ,N,F<:FlatFourier{T,<:Flat{θ,N,<:fourier∂}}} = @. f′ = ∇i \ f

# map space derivatives
DerivBasis(::Type{<:FlatS0{T,Flat{θ,N,map∂}}}) where {T,θ,N} = Map
DerivBasis(::Type{<:FlatS2{T,Flat{θ,N,map∂}}}) where {T,θ,N} = QUMap
function mul!(f′::F, ∇::Union{∇i{coord},AdjOp{<:∇i{coord}}}, f::F) where {coord,T,θ,N,F<:FlatS0Map{T,<:Flat{θ,N,<:map∂}}}
    n,m = size(f.Tx)
    Δx = FFTgrid(f).Δx #* (∇ isa AdjOp ? -1 : 1) why doesn't this need to be here???
    if coord==0
        @inbounds for j=2:m-1
            @simd for i=1:n
                f′.Tx[i,j] = (f.Tx[i,j+1] - f.Tx[i,j-1])/2Δx
            end
        end
        @inbounds for i=1:n
            f′.Tx[i,1] = (f.Tx[i,2]-f.Tx[i,end])/2Δx
            f′.Tx[i,end] = (f.Tx[i,1]-f.Tx[i,end-1])/2Δx
        end
    elseif coord==1
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
function mul!(f′::F, ∇::Union{∇i{coord},AdjOp{<:∇i{coord}}}, f::F) where {coord,T,θ,N,F<:FlatS2Map{T,<:Flat{θ,N,<:map∂}}}
    mul!(f′.Q, ∇, f.Q)
    mul!(f′.U, ∇, f.U)
    f′
end


# specialized mul! to avoid allocation when doing `∇' * vector` when stuff is in
# the right the basis. expects memf′ is a preallocated memory that can be used
# to store an intermediate result. the default value uses v[1], hence destroys
# the original v. if this is not desired, provide a different field.
mul!(f′::F, ∇::Adjoint{∇i,<:∇Op}, v::FieldVector{F}, memf′::F=v[1]) where {F<:Union{FlatMap{<:Any,<:Flat{<:Any,<:Any,map∂}},FlatFourier{<:Any,<:Flat{<:Any,<:Any,fourier∂}}}} =
    (mul!(f′,∇[1],v[1]); mul!(memf′,∇[2],v[2]); (f′ .+= memf′))

# some other mul! which will eventually be generalized into broadcasted stuff
mul!(v::FlatS0Map{T,P}, a::AdjField{QUMap,S2,P,F}, b::F) where {T,P,F<:FlatS2QUMap{T,P}} = 
    ((@. v.Tx = (a').Qx*b.Qx + (a').Ux*b.Ux); v)
mul!(v::FlatS0Map{T,P}, a::AdjField{Map,S0,P,F}, b::F) where {T,P,F<:FlatS0Map{T,P}} = 
    ((@. v.Tx = (a').Tx*b.Tx); v)


# bandpass
HarmonicBasis(::Type{<:FlatS0}) = Fourier
HarmonicBasis(::Type{<:FlatS2}) = QUFourier
broadcast_data(::Type{F}, op::BandPassOp) where {T,P,F<:FlatFourier{T,P}} =
    (Cℓ_2D(op.ℓ,op.Wℓ,FFTgrid(T,P).r)[1:Nside(P)÷2+1,:],)


# allows std and var of a Vector of FlatFields to work
real(f::CMBLensing.FlatField) = f


# logdets
logdet(L::FullDiagOp{<:FlatS0Fourier})   = real(sum(nan2zero.(log.(unfold(L.f.Tl)))))
logdet(L::FullDiagOp{<:FlatS0Map})       = real(sum(nan2zero.(log.(complex(L.f.Tx)))))
logdet(L::FullDiagOp{<:FlatS2EBFourier}) = real(sum(nan2zero.(log.(unfold(L.f.El))) + nan2zero.(log.(unfold(L.f.Bl)))))
