
const FlatIQUMap{T,P} = Field2Tuple{FlatS0Map{T,P},FlatS2QUMap{T,P}}
const FlatTEBFourier{T,P} = Field2Tuple{FlatS0Fourier{T,P},FlatS2EBFourier{T,P}}


struct FlatTEBCov{T,P} <: LinDiagOp{P,Spin,Basis2Tuple{Fourier,EBFourier}}
    ΣTE :: SMatrix{2,2,Diagonal{T},4}
    ΣB :: Matrix{T}
end

function Cℓ_to_cov(::Type{T}, ::Type{P}, ::Type{S0}, ::Type{S2}, ℓ, CℓTT, CℓEE, CℓBB, CℓTE) where {T,P}
    Σ(Cℓ) = Diagonal(Cℓ_2D(P,ℓ,Cℓ)[:])
    FlatTEBCov{T,P}(
        @SMatrix([Σ(CℓTT) Σ(CℓTE);
                  Σ(CℓTE) Σ(CℓEE)]),
        Cℓ_2D(P,ℓ,CℓBB)
    )
end

function *(cv::FlatTEBCov{T,P}, f::FlatTEBFourier{T,P}) where {T,N,P<:Flat{<:Any,N}} 
    (t,e),b = (cv.ΣTE * [f.f1.Tl[:], f.f2.El[:]]), cv.ΣB .* f.f2.Bl
    FieldTuple(FlatS0Fourier{T,P}(reshape(t,N÷2+1,N)),FlatS2EBFourier{T,P}(reshape(e,N÷2+1,N),b))
end
\(cv::FlatTEBCov, f::FlatTEBFourier) = inv(cv)*f

literal_pow(^,cv::FlatTEBCov,::Type{Val{-1}}) = inv(cv)
inv(cv::FlatTEBCov{T,P}) where {T,P} = FlatTEBCov{T,P}(inv(cv.ΣTE),1./cv.ΣB)
sqrtm(cv::FlatTEBCov{T,P}) where {T,P} = FlatTEBCov{T,P}(nan2zero.(sqrtm(cv.ΣTE)),sqrt.(cv.ΣB))

simulate(cv::FlatTEBCov{T,P}) where {T,P} = sqrtm(cv) * white_noise(FlatTEBFourier{T,P})

# arithmetic with FlatTEBCov and scalars
broadcast_data(::Type{FlatTEBCov},s::Scalar) = repeated(s)
broadcast_data(::Type{FlatTEBCov},cv::FlatTEBCov) = fieldvalues(cv)
function broadcast(f,args::Union{_,FlatTEBCov{T,P},Scalar}...) where {T,P,_<:FlatTEBCov{T,P}}
    FlatTEBCov{T,P}(map(broadcast, repeated(f), map(broadcast_data, repeated(FlatTEBCov), args)...)...)
end


# pretty hacky but needed for StaticArrays to invert the 2x2 TE block matrix correctly
# maybe we can get a PR to StaticArrays accepted that removes the need for this
one(::Type{Diagonal{T}}) where {T} = Diagonal(Vector{T}(0))
one(::Type{Vector{T}}) where {T} = Vector{T}(0)
zero(::Type{Diagonal{T}}) where {T} = Diagonal(Vector{T}(0))
zero(::Type{Vector{T}}) where {T} = Vector{T}(0)

# so that sqrtm works
inv(d::Diagonal) = Diagonal(1./d.diag)

## also PR to StaticArrays, coded in that style
sqrtm(A::StaticMatrix) = _sqrtm(Size(A),A)
@generated function _sqrtm(::Size{(2,2)}, A::StaticMatrix)
    @assert size(A) == (2,2)
    T = typeof(sqrtm(one(eltype(A))))
    newtype = similar_type(A,T)
            
    quote
        a,b,c,d = A
        s = sqrtm(a*d-b*c)
        t = inv(sqrtm(a+d+2s))
        ($newtype)(t*(a+s), t*b, 
                   t*c,     t*(d+s))
    end
end
