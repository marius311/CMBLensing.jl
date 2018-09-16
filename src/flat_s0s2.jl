export FlatIQUMap, FlatTEBFourier


# (S0,S2) fields don't need their own structs, they just use FieldTuples.
# here we define shorthand for the 2 often used ones, but any of the 8
# combinations of (Map,Fourier)_S0 x ((EB,QU)x(Map,Fourier))_S2 can be created
# via FieldTuple(...)
const FlatIQUMap{T,P} = FieldTuple{Tuple{FlatS0Map{T,P},FlatS2QUMap{T,P}},BasisTuple{Tuple{Map,QUMap}},SpinTuple{Tuple{S0,S2}},PixTuple{Tuple{P,P}}}
const FlatTEBFourier{T,P} = FieldTuple{Tuple{FlatS0Fourier{T,P},FlatS2EBFourier{T,P}},BasisTuple{Tuple{Fourier,EBFourier}},SpinTuple{Tuple{S0,S2}},PixTuple{Tuple{P,P}}}
# some convenience constructors
FlatIQUMap{T,P}(i,q,u) where {T,P} = FieldTuple(FlatS0Map{T,P}(i),FlatS2QUMap{T,P}(q,u))
FlatTEBFourier{T,P}(t,e,b) where {T,P} = FieldTuple(FlatS0Fourier{T,P}(t),FlatS2EBFourier{T,P}(e,b))

# any of the 8 possible (S0,S2) fields
const FlatS02{T,P} = FieldTuple{<:Tuple{FlatS0{T,P},FlatS2{T,P}}}

# a block TEB diagonal operator
struct FlatTEBCov{T,P} <: LinOp{BasisTuple{Tuple{Fourier,EBFourier}},SpinTuple{Tuple{S0,S2}},P}
    ΣTE :: SMatrix{2,2,Diagonal{T},4}
    ΣB :: Matrix{T}
    unsafe_invert :: Bool
    FlatTEBCov{T,P}(ΣTE,ΣB,unsafe_invert=true) where {T,P} = new{T,P}(ΣTE,ΣB,unsafe_invert)
end

# convenience constructor
function FlatTEBCov{T,P}(ΣTT::AbstractMatrix, ΣTE::AbstractMatrix, ΣEE::AbstractMatrix, ΣBB::AbstractMatrix) where {T,P}
    D(Σ) = Diagonal(Σ[:])
    FlatTEBCov{T,P}(@SMatrix([D(ΣTT) D(ΣTE); D(ΣTE) D(ΣEE)]), ΣBB)
end

# contructing from Cℓs
function Cℓ_to_cov(::Type{T}, ::Type{P}, ::Type{S0}, ::Type{S2}, ℓ, CℓTT, CℓEE, CℓBB, CℓTE; mask_nyquist=true) where {T,P}
    _Mnyq = mask_nyquist ? Mnyq : identity
    FlatTEBCov{T,P}(_Mnyq.(T,P,Cℓ_2D.(P,[ℓ],[CℓTT,CℓTE,CℓEE,CℓBB]))...)
end

# applying the operator
function *(L::FlatTEBCov{T,P}, f::FlatTEBFourier{T,P}) where {T,N,P<:Flat{<:Any,N}} 
    (t,e),b = (L.ΣTE * [@view(f.fs[1].Tl[:]), @view(f.fs[2].El[:])]), L.ΣB .* f.fs[2].Bl
    FieldTuple(FlatS0Fourier{T,P}(reshape(t,N÷2+1,N)),FlatS2EBFourier{T,P}(reshape(e,N÷2+1,N),b))
end
adjoint(L::F) where {F<:FlatTEBCov} = F(L.ΣTE',L.ΣB)
inv(L::F) where {F<:FlatTEBCov} = F((L.unsafe_invert ? (nan2zero.(inv(L.ΣTE)), nan2zero.(1./L.ΣB)) : (inv(L.ΣTE), 1/L.ΣB))...)
sqrtm(L::F) where {F<:FlatTEBCov} = F((L.unsafe_invert ? nan2zero.(sqrtm(L.ΣTE)) : sqrtm(L.ΣTE)), sqrt.(L.ΣB))
simulate(L::FlatTEBCov{T,P}) where {T,P} = sqrtm(L) * white_noise(FlatTEBFourier{T,P})
function Diagonal(L::FlatTEBCov{T,P}) where {T,N,P<:Flat{<:Any,N}}
    FullDiagOp(FlatTEBFourier{T,P}(reshape.(diag.([L.ΣTE[1,1], L.ΣTE[2,2]]),[(N÷2+1,N)])..., L.ΣB))
end


# multiplication by a Diag{TEB}
function *(L::FlatTEBCov{T,P}, D::FullDiagOp{FlatTEBFourier{T,P}}) where {T,P}
    t,e,b = Diagonal(real(D.f[:Tl][:])), Diagonal(real(D.f[:El][:])), real(D.f[:Bl])
    FlatTEBCov{T,P}(L.ΣTE * [[t] [0]; [0] [e]], b .* L.ΣB)
end
*(D::FullDiagOp{FlatTEBFourier{T,P}}, L::FlatTEBCov{T,P}) where {T,P} = (adjoint(L)*D)'
# multiplication of two FlatTEBCov
*(La::F, Lb::F) where {F<:FlatTEBCov} = F(La.ΣTE*Lb.ΣTE, La.ΣB.*Lb.ΣB)


# simple arithmetic with scalars
+(s::UniformScaling{<:Scalar}, L::FlatTEBCov) = L + s
+(L::F, s::UniformScaling{<:Scalar}) where {F<:FlatTEBCov} =
    F([[Diagonal(diag(L.ΣTE[1,1]).+s.λ)] [L.ΣTE[1,2]]; [L.ΣTE[2,1]] [Diagonal(diag(L.ΣTE[2,2]).+s.λ)]], L.ΣB.+s.λ, L.unsafe_invert)
*(s::Scalar, L::FlatTEBCov) = L * s
*(L::F, s::Scalar) where {F<:FlatTEBCov} = F(L.ΣTE .* s, L.ΣB .* s, L.unsafe_invert)

function get_Cℓ(f::FlatS02{T,P}; which=(:TT,:TE,:EE,:BB), kwargs...) where {T,P}
    Cℓs = [get_Cℓ((FlatS0Fourier{T,P}(f[Symbol(x,:l)]) for x=xs)...; kwargs...) for xs in string.(which)]
    (Cℓs[1][1], hcat(last.(Cℓs)...))
end

    
# convenience methods for getting components as S0
getproperty(f::FlatS02{T,P},::Val{:T}) where {T,P} = FlatS0Map{T,P}(f.Tx)
getproperty(f::FlatS02{T,P},::Val{:E}) where {T,P} = FlatS0Map{T,P}(f.Ex)
getproperty(f::FlatS02{T,P},::Val{:B}) where {T,P} = FlatS0Map{T,P}(f.Bx)
