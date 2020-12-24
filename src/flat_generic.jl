const FlatField{P,T,M} = Union{FlatS0{P,T,M},FlatS2{P,T,M},FlatS02{P,T,M}}

const FlatFieldMap{P,T,M} = Union{FlatMap{P,T,M},FlatS2Map{P,T,M},FlatS02Map{P,T,M}}
const FlatFieldFourier{P,T,M} = Union{FlatFourier{P,T,M},FlatS2Fourier{P,T,M},FlatS02Fourier{P,T,M}}

### pretty printing
@show_datatype show_datatype(io::IO, t::Type{F}) where {N,θ,∂mode,D,T,M,F<:FlatField{Flat{N,θ,∂mode,D},T,M}} =
print(io, "$(pretty_type_name(F)){$(join(N .* (1,1), "×"))$(D==1 ? "" : "(×$D)") map, $(θ)′ pixels, $(∂mode.name.name), $M}")
for F in (:FlatMap, :FlatFourier, 
          :FlatQUMap, :FlatQUFourier, :FlatEBMap, :FlatEBFourier, 
          :FlatIQUMap, :FlatIQUFourier, :FlatIEBMap, :FlatIEBFourier)
    @eval pretty_type_name(::Type{<:$F}) = $(string(F))
end
function Base.summary(io::IO, f::FlatField{<:Flat{N,<:Any,<:Any,D}}) where {N,D}
    print(io, "$(length(f)÷D)", (D==1 ? "" : "(×$D)"), "-element ")
    showarg(io, f, true)
end

### field info
@memoize fieldinfo(::Type{P},::Type{T}=Float32,::Type{M}=Matrix) where {Nside,θpix,∂mode,D,P<:Flat{Nside,θpix,∂mode,D},T,M} = 
    (;FlatInfo(T,basetype(M),θpix,Nside,D)..., ∂mode=∂mode)
@memoize fieldinfo(::Type{F}) where {P<:Flat,T,M,F<:FlatField{P,T,M}} = 
    (;fieldinfo(P,T,M)..., P, M, B=basis(F), S=spin(F))
fieldinfo(::F) where {F<:FlatField} = fieldinfo(F)
fieldinfo(::DiagOp{F}) where {F<:FlatField} = fieldinfo(F)
get_storage(f::F) where {F<:FlatField} = basetype(fieldinfo(F).M)

### promotion & conversion
# note: we don't need to promote the eltype T here since that will be
# automatically handled in broadcasting
function promote(f1::F1, f2::F2) where {T1,θ1,N1,∂mode1,F1<:FlatField{<:Flat{N1,θ1,∂mode1},T1},T2,θ2,N2,∂mode2,F2<:FlatField{<:Flat{θ2,N2,∂mode2},T2}}
    B     = promote_type(basis(F1),basis(F2))
    ∂mode = promote_type(∂mode1,∂mode2)
    B(∂mode(f1)), B(∂mode(f2))
end
(::Type{∂mode})(f::F) where {∂mode<:∂modes,N,θ,D,F<:FlatS0{<:Flat{N,θ,<:Any,D}}} = basetype(F){Flat{N,θ,∂mode,D}}(fieldvalues(f)...)
(::Type{∂mode})(f::FieldTuple{B}) where {∂mode<:∂modes,B} = FieldTuple{B}(map(∂mode,f.fs))

### basis-like definitions
LenseBasis(::Type{<:FlatS0})  =    Map
LenseBasis(::Type{<:FlatS2})  =  QUMap
LenseBasis(::Type{<:FlatS02}) = IQUMap
DerivBasis(::Type{<:FlatS0{<:Flat{<:Any,<:Any,fourier∂}}})  =    Fourier
DerivBasis(::Type{<:FlatS2{<:Flat{<:Any,<:Any,fourier∂}}})  =  QUFourier
DerivBasis(::Type{<:FlatS02{<:Flat{<:Any,<:Any,fourier∂}}}) = IQUFourier
DerivBasis(::Type{<:FlatS0{<:Flat{<:Any,<:Any,map∂}}})      =    Map
DerivBasis(::Type{<:FlatS2{<:Flat{<:Any,<:Any,map∂}}})      =  QUMap
DerivBasis(::Type{<:FlatS02{<:Flat{<:Any,<:Any,map∂}}})     = IQUMap


### derivatives

## Fourier-space
# the use of @generated here is for memoization so that we don't have do the
# `prefactor * im * ...` each time. actually, since for the first two cases
# these are only 1-d arrays, that's a completely unnecessary optimization, but
# without the @generated it triggers an order-dependenant compilation bug which
# e.g. slows down LenseFlow by a factor of ~4 so we gotta keep it for now ¯\_(ツ)_/¯
@memoize function broadcastable(::Type{F}, ::∇diag{1,<:Any,prefactor}) where {P,T,M,prefactor,F<:FlatFourier{P,T,M}}
    @unpack kx = fieldinfo(F)
    @. prefactor * im * kx'
end
@memoize function broadcastable(::Type{F}, ::∇diag{2,<:Any,prefactor}) where {P,T,M,prefactor,F<:FlatFourier{P,T,M}}
    @unpack ky, Ny = fieldinfo(F)
    @. prefactor * im * ky[1:Ny÷2+1]
end
@memoize function broadcastable(::Type{F}, ::∇²diag) where {P,T,M,F<:FlatFourier{P,T,M}}
    @unpack kx, ky, Ny = fieldinfo(F)
    @. kx'^2 + ky[1:Ny÷2+1]^2
end

## Map-space
function copyto!(f′::F, bc::Broadcasted{<:Any,<:Any,typeof(*),Tuple{∇diag{coord,covariant,prefactor},F}}) where {coord,covariant,prefactor,T,θ,N,D,F<:FlatMap{Flat{N,θ,map∂,D},T}}
    D!=1 && error("Gradients of batched map∂ flat maps not implemented yet.")
    f = bc.args[2]
    n,m = size(f.Ix)
    α = 2 * prefactor * fieldinfo(f).Δx
    if coord==1
        @inbounds for j=2:m-1
            @simd for i=1:n
                f′[i,j] = (f[i,j+1] - f[i,j-1])/α
            end
        end
        @inbounds for i=1:n
            f′[i,1] = (f[i,2]-f[i,end])/α
            f′[i,end] = (f[i,1]-f[i,end-1])/α
        end
    elseif coord==2
        @inbounds for j=1:n
            @simd for i=2:m-1
                f′[i,j] = (f[i+1,j] - f[i-1,j])/α
            end
            f′[1,j] = (f[2,j]-f[end,j])/α
            f′[end,j] = (f[1,j]-f[end-1,j])/α
        end
    end
    f′
end



### bandpass
HarmonicBasis(::Type{<:FlatS0}) = Fourier
HarmonicBasis(::Type{<:FlatQU}) = QUFourier
HarmonicBasis(::Type{<:FlatEB}) = EBFourier
broadcastable(::Type{F}, bp::BandPass) where {P,T,F<:FlatFourier{P,T}} = Cℓ_to_2D(P,T,bp.Wℓ)
    

### logdets
logdet(L::Diagonal{<:Complex,<:FlatFourier}) = batch(real(sum_kbn(nan2zero.(log.(L.diag[:Il,full_plane=true])),dims=(1,2))))
logdet(L::Diagonal{<:Real,   <:FlatMap})     = batch(real(sum_kbn(nan2zero.(log.(complex.(L.diag.Ix))),dims=(1,2))))
### traces
tr(L::Diagonal{<:Complex,<:FlatFourier{<:Flat{N}}}) where {N} = batch(real(sum_kbn(L.diag[:Il,full_plane=true],dims=(1,2)))) / prod(N .* (1,1))
tr(L::Diagonal{<:Real,   <:FlatMap})                          = batch(real(sum_kbn(complex.(L.diag.Ix),dims=(1,2))))


### misc
Cℓ_to_Cov(f::FlatField{P,T,M}, args...; kwargs...) where {P,T,M} = adapt(M, Cℓ_to_Cov(P,T,spin(f),args...; kwargs...))

function pixwin(f::FlatField) 
    @unpack θpix,P,T,k = fieldinfo(f)
    Diagonal(FlatFourier{P,T}((pixwin.(θpix,k) .* pixwin.(θpix,k'))[1:end÷2+1,:]))
end

global_rng_for(::Type{<:FlatField{<:Any,<:Any,M}}) where {M} = global_rng_for(M)

"""
    fixed_white_noise(rng, F)

Like white noise but the amplitudes are fixed to unity, only the phases are
random. Currently only implemented when F is a Fourier basis. Note that unlike
[`white_noise`](@ref), fixed white-noise generated in EB and QU Fourier bases
are not statistically the same.
"""
fixed_white_noise(rng, F::Type{<:FlatFieldFourier}) =
     exp.(im .* angle.(basis(F)(white_noise(rng,F)))) .* fieldinfo(F).Nside



### UDGradeOp
function (L::UDGradeOp * f::FlatField)
    ud_grade(f, L.θout, mode=:map, deconv_pixwin=false, anti_aliasing=false)
end
function (L::Adjoint{<:Any,UDGradeOp} * f::FlatField{<:Any,T}) where {T}
    ud_grade(f, parent(L).θin, mode=:map, deconv_pixwin=false, anti_aliasing=false) / T(parent(L).θout/parent(L).θin)^2
end
