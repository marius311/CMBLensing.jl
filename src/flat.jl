
# derivatives can either be applied in fourier space by multipliying by im*k or
# in map space by finite differences. this tracks which option to use for a
# given field
abstract type ∂modes end
struct fourier∂ <: ∂modes end
struct map∂ <: ∂modes end
promote_rule(::Type{map∂}, ::Type{fourier∂}) = fourier∂

# Flat{θpix,Nside,∂mode} is a flat sky pixelization with `Nside` pixels per side
# and pixels of width `θpix` arcmins, where derivatives are done according to ∂mode
abstract type Flat{Nside,θpix,∂mode<:∂modes} <: Pix end

# for convenience
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
    sincos2ϕ :: Tuple{Matrix{T},Matrix{T}}
    FFT :: F
end
 
# use @generated function to memoize FFTgrid for given (T,θ,Nside) combinations
FFTgrid(::Type{<:Flat{Nside,θpix}}, ::Type{T}) where {T, θpix, Nside} = FFTgrid(T, Val(θpix), Val(Nside))
@generated function FFTgrid(::Type{T}, ::Val{θpix}, ::Val{Nside}) where {T<:Real, θpix, Nside}
    Δx  = deg2rad(θpix/60)
    FFTW.set_num_threads(Sys.CPU_THREADS)
    FFT = T(Δx^2/(2π)) * plan_rfft(Matrix{T}(undef,Nside,Nside); flags=FFTW.ESTIMATE, timelimit=5)
    Δℓ  = 2π/(Nside*Δx)
    nyq = 2π/(2Δx)
    x,k = (ifftshift(-Nside÷2:(Nside-1)÷2),) .* [Δx,Δℓ]'
    r   = sqrt.(.+((reshape(k.^2, (s=ones(Int,2); s[i]=Nside; tuple(s...))) for i=1:2)...))
    ϕ   = angle.(k' .+ im*k)[1:Nside÷2+1,:]
    sincos2ϕ = @. sin(2ϕ), cos(2ϕ)
    FFTgrid{T,typeof(FFT)}(θpix, Nside, Δx, Δℓ, nyq, x, k, r, sincos2ϕ, FFT)
end



Cℓ_2D(ℓ, Cℓ, r) = LinearInterpolation(ℓ, Cℓ, extrapolation_bc = 0).(r)
Cℓ_2D(::Type{P}, ℓ, Cℓ) where {N,P<:Flat{N}} = Cℓ_2D(ℓ,Cℓ,FFTgrid(Float64,P).r)[1:N÷2+1,:]

""" filter out the single row/column in the real FFT matrix `M` which
corresponds to exactly the nyquist frequency """
function Mnyq(::Type{T},::Type{P}, M) where {T,N,P<:Flat{N}}
    if iseven(N)
        inyq = first((1:N)[@. FFTgrid(P,T).k ≈ -FFTgrid(P,T).nyq])
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

const FlatField{P,T,M} = Union{FlatS0{P,T,M},FlatS2{P,T,M}}

## promotion

function promote(f1::F1, f2::F2) where {T1,θ1,N1,∂mode1,F1<:FlatS0{Flat{N1,θ1,∂mode1},T1},T2,θ2,N2,∂mode2,F2<:FlatS0{Flat{θ2,N2,∂mode2},T2}}
    T     = promote_type(T1,T2)
    B     = promote_type(basis(F1),basis(F2))
    ∂mode = promote_type(∂mode1,∂mode2)
    B(T(∂mode(f1))), B(T(∂mode(f2)))
end

## conversion

(::Type{T})(f::FlatMap{P}) where {T<:Real,P} =  FlatMap{P}(convert(Matrix{T}, f.Ix))
(::Type{T})(f::FlatFourier{P}) where {T<:Real,P} =  FlatFourier{P}(convert(Matrix{Complex{T}}, f.Il))
(::Type{∂mode})(f::F) where {∂mode<:∂modes,θ,N,F<:FlatS0{<:Flat{θ,N}}} = basetype(F){Flat{θ,N,∂mode}}(fieldvalues(f)...)


FFTgrid(::FlatS0{P,T}) where {P,T} = FFTgrid(P,T)


# broadcast_data(::Type{F}, f::F) where {F<:BaseFlatField} = map(StridedView, fieldvalues(f))
# broadcast_data(::Type{F2}, f::F0) where {F2<:FlatS2Map, F0<:FlatS0Map} = broadcast_data(F0,f)
# *(f0::FlatS0Map, f2::FlatS2Map) = f0 .* f2
# 


LenseBasis(::Type{<:FlatS0}) = Map
LenseBasis(::Type{<:FlatS2}) = QUMap

# 
# ## derivatives
# 
# fourier space derivatives
DerivBasis(::Type{<:FlatS0{<:Flat{<:Any,<:Any,fourier∂}}}) =   Fourier
DerivBasis(::Type{<:FlatQU{<:Flat{<:Any,<:Any,fourier∂}}}) = QUFourier
DerivBasis(::Type{<:FlatEB{<:Flat{<:Any,<:Any,fourier∂}}}) = EBFourier


# α = im #∇i isa AdjOp ? -im : im
broadcastable(::Type{<:FlatFourier{P,T}}, ::∇i{0}) where {P,T} = im * FFTgrid(P,T).k'
broadcastable(::Type{<:FlatFourier{P,T}}, ::∇i{1}) where {P,T} = im * FFTgrid(P,T).k[1:Nside(P)÷2+1]

# @generated function broadcast_data(::Type{<:BaseFlatFourier{T,P}}, ::∇²Op) where {coord,T,P}
#     (FFTgrid(P,T).k' .^2 .+ FFTgrid(P,T).k[1:Nside(P)÷2+1].^2,)
# end
# mul!( f′::F, ∇i::Union{∇i,AdjOp{<:∇i}}, f::F) where {T,θ,N,F<:FlatFourier{T,<:Flat{θ,N,<:fourier∂}}} = @. f′ = ∇i * f
# ldiv!(f′::F, ∇i::Union{∇i,AdjOp{<:∇i}}, f::F) where {T,θ,N,F<:FlatFourier{T,<:Flat{θ,N,<:fourier∂}}} = @. f′ = ∇i \ f
# 
# # map space derivatives
# DerivBasis(::Type{<:FlatS0{T,Flat{θ,N,map∂}}}) where {T,θ,N} = Map
# DerivBasis(::Type{<:FlatS2{T,Flat{θ,N,map∂}}}) where {T,θ,N} = QUMap
# function mul!(f′::F, ∇::Union{∇i{coord},AdjOp{<:∇i{coord}}}, f::F) where {coord,T,θ,N,F<:FlatS0Map{T,<:Flat{θ,N,<:map∂}}}
#     n,m = size(f.Tx)
#     Δx = FFTgrid(f).Δx #* (∇ isa AdjOp ? -1 : 1) why doesn't this need to be here???
#     if coord==0
#         @inbounds for j=2:m-1
#             @simd for i=1:n
#                 f′.Tx[i,j] = (f.Tx[i,j+1] - f.Tx[i,j-1])/2Δx
#             end
#         end
#         @inbounds for i=1:n
#             f′.Tx[i,1] = (f.Tx[i,2]-f.Tx[i,end])/2Δx
#             f′.Tx[i,end] = (f.Tx[i,1]-f.Tx[i,end-1])/2Δx
#         end
#     elseif coord==1
#         @inbounds for j=1:n
#             @simd for i=2:m-1
#                 f′.Tx[i,j] = (f.Tx[i+1,j] - f.Tx[i-1,j])/2Δx
#             end
#             f′.Tx[1,j] = (f.Tx[2,j]-f.Tx[end,j])/2Δx
#             f′.Tx[end,j] = (f.Tx[1,j]-f.Tx[end-1,j])/2Δx
#         end
#     end
#     f′
# end
# function mul!(f′::F, ∇::Union{∇i{coord},AdjOp{<:∇i{coord}}}, f::F) where {coord,T,θ,N,F<:FlatS2Map{T,<:Flat{θ,N,<:map∂}}}
#     mul!(f′.Q, ∇, f.Q)
#     mul!(f′.U, ∇, f.U)
#     f′
# end
# 
# 
# # specialized mul! to avoid allocation when doing `∇' * vector` when stuff is in
# # the right the basis. expects memf′ is a preallocated memory that can be used
# # to store an intermediate result. the default value uses v[1], hence destroys
# # the original v. if this is not desired, provide a different field.
# mul!(f′::F, ∇::Adjoint{∇i,<:∇Op}, v::FieldVector{F}, memf′::F=v[1]) where {F<:Union{FlatMap{<:Any,<:Flat{<:Any,<:Any,map∂}},FlatFourier{<:Any,<:Flat{<:Any,<:Any,fourier∂}}}} =
#     (mul!(f′,∇[1],v[1]); mul!(memf′,∇[2],v[2]); (f′ .+= memf′))
# 
# # some other mul! which will eventually be generalized into broadcasted stuff
# mul!(v::FlatS0Map{T,P}, a::AdjField{QUMap,S2,P,F}, b::F) where {T,P,F<:FlatS2QUMap{T,P}} = 
#     ((@. v.Tx = (a').Qx*b.Qx + (a').Ux*b.Ux); v)
# mul!(v::FlatS0Map{T,P}, a::AdjField{Map,S0,P,F}, b::F) where {T,P,F<:FlatS0Map{T,P}} = 
#     ((@. v.Tx = (a').Tx*b.Tx); v)
# 
# 
# # bandpass
# HarmonicBasis(::Type{<:FlatS0}) = Fourier
# HarmonicBasis(::Type{<:FlatS2}) = QUFourier
# broadcast_data(::Type{F}, op::BandPassOp) where {T,P,F<:FlatFourier{T,P}} =
#     (Cℓ_2D(op.ℓ,op.Wℓ,FFTgrid(P,T).r)[1:Nside(P)÷2+1,:],)
# 
# 
# # allows std and var of a Vector of FlatFields to work
# real(f::CMBLensing.FlatField) = f
# 
# 
# # logdets
# logdet(L::FullDiagOp{<:FlatS0Fourier})   = real(sum(nan2zero.(log.(unfold(L.f.Tl)))))
# logdet(L::FullDiagOp{<:FlatS0Map})       = real(sum(nan2zero.(log.(complex(L.f.Tx)))))
# logdet(L::FullDiagOp{<:FlatS2EBFourier}) = real(sum(nan2zero.(log.(unfold(L.f.El))) + nan2zero.(log.(unfold(L.f.Bl)))))
