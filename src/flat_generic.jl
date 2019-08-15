const FlatField{P,T,M} = Union{FlatS0{P,T,M},FlatS2{P,T,M},FlatS02{P,T,M}}

### pretty printing
@show_datatype show_datatype(io::IO, t::Type{F}) where {N,θ,∂mode,T,M,F<:FlatField{Flat{N,θ,∂mode},T,M}} =
    print(io, "$(pretty_type_name(F)){$(N)×$(N) map, $(θ)′ pixels, $(∂mode.name.name), $(M.name.name){$(M.parameters[1])}}")
for F in (:FlatMap, :FlatFourier, 
          :FlatQUMap, :FlatQUFourier, :FlatEBMap, :FlatEBFourier, 
          :FlatIQUMap, :FlatIQUFourier, :FlatIEBMap, :FlatIEBFourier)
    @eval pretty_type_name(::Type{<:$F}) = $(string(F))
end
    
    


### promotion & conversion
# note: we don't need to promote the eltype T here since that will be
# automatically handled in broadcasting
function promote(f1::F1, f2::F2) where {T1,θ1,N1,∂mode1,F1<:FlatField{Flat{N1,θ1,∂mode1},T1},T2,θ2,N2,∂mode2,F2<:FlatField{Flat{θ2,N2,∂mode2},T2}}
    B     = promote_type(basis(F1),basis(F2))
    ∂mode = promote_type(∂mode1,∂mode2)
    B(∂mode(f1)), B(∂mode(f2))
end
(::Type{∂mode})(f::F) where {∂mode<:∂modes,N,θ,F<:FlatS0{<:Flat{N,θ}}} = basetype(F){Flat{N,θ,∂mode}}(fieldvalues(f)...)
(::Type{∂mode})(f::FieldTuple{B}) where {∂mode<:∂modes,B} = FieldTuple{B}(map(∂mode,f.fs))
FFTgrid(::FlatField{P,T}) where {P,T} = FFTgrid(P,T)

### basis-like definitions
LenseBasis(::Type{<:FlatS0}) = Map
LenseBasis(::Type{<:FlatS2}) = QUMap
DerivBasis(::Type{<:FlatS0{<:Flat{<:Any,<:Any,fourier∂}}}) =   Fourier
DerivBasis(::Type{<:FlatS2{<:Flat{<:Any,<:Any,fourier∂}}}) = QUFourier
DerivBasis(::Type{<:FlatS0{<:Flat{<:Any,<:Any,map∂}}})     =   Map
DerivBasis(::Type{<:FlatS2{<:Flat{<:Any,<:Any,map∂}}})     = QUMap

### derivatives

## Fourier-space
broadcastable(::Type{<:FlatFourier{P,T}}, ::∇diag{1,<:Any,prefactor}) where {P,T,prefactor} = 
    @. prefactor * im * $FFTgrid(P,T).k'
broadcastable(::Type{<:FlatFourier{P,T}}, ::∇diag{2,<:Any,prefactor}) where {N,P<:Flat{N},T,prefactor} = 
    @. prefactor * im * $FFTgrid(P,T).k[1:N÷2+1]
broadcastable(::Type{<:FlatFourier{P,T}}, ::∇²diag) where {N,P<:Flat{N},T} =
    @. $FFTgrid(P,T).k'^2 + $FFTgrid(P,T).k[1:N÷2+1]^2

## Map-space
function copyto!(f′::F, bc::Broadcasted{<:Any,<:Any,typeof(*),Tuple{∇diag{coord,covariant,prefactor},F}}) where {coord,covariant,prefactor,T,θ,N,F<:FlatMap{Flat{N,θ,map∂},T}}
    f = bc.args[2]
    n,m = size(f.Ix)
    α = 2 * prefactor * FFTgrid(f).Δx
    if coord==1
        @inbounds for j=2:m-1
            @simd for i=1:n
                f′.Ix[i,j] = (f.Ix[i,j+1] - f.Ix[i,j-1])/α
            end
        end
        @inbounds for i=1:n
            f′.Ix[i,1] = (f.Ix[i,2]-f.Ix[i,end])/α
            f′.Ix[i,end] = (f.Ix[i,1]-f.Ix[i,end-1])/α
        end
    elseif coord==2
        @inbounds for j=1:n
            @simd for i=2:m-1
                f′.Ix[i,j] = (f.Ix[i+1,j] - f.Ix[i-1,j])/α
            end
            f′.Ix[1,j] = (f.Ix[2,j]-f.Ix[end,j])/α
            f′.Ix[end,j] = (f.Ix[1,j]-f.Ix[end-1,j])/α
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
logdet(L::Diagonal{<:Complex,<:FlatFourier})   = real(sum(nan2zero∘log, unfold(L.diag.Il)))
logdet(L::Diagonal{<:Real,<:FlatMap})          = real(sum(nan2zero∘log, complex(L.diag.Tx)))
logdet(L::Diagonal{<:Complex,<:FlatEBFourier}) = real(sum(nan2zero∘log, unfold(L.diag.El)) + sum(nan2zero∘log, unfold(L.diag.Bl)))
### traces
tr(L::Diagonal{<:Complex,<:FlatFourier})   = real(sum(unfold(L.diag.Il)))
tr(L::Diagonal{<:Real,<:FlatMap})          = real(sum(complex(L.diag.Tx)))
tr(L::Diagonal{<:Complex,<:FlatEBFourier}) = real(sum(unfold(L.diag.El)) + sum(unfold(L.diag.Bl)))

# always do dot product in map basis
dot(a::FlatField{P}, b::FlatField{P}) where {P} = Ł(a)[:] ⋅ Ł(b)[:] * FFTgrid(a).Δx^2
