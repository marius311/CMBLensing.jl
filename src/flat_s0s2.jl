
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

function get_Cℓ(f::Field2Tuple{<:FlatS0{T,P},<:FlatS2{T,P}}; which=(:TT,:TE,:EE,:BB), kwargs...) where {T,P}
    Cℓs = [get_Cℓ((FlatS0Fourier{T,P}(f[Symbol(x,:l)]) for x=xs)...; kwargs...) for xs in string.(which)]
    (Cℓs[1][1], hcat(last.(Cℓs)...))
end

getindex(f::Field2Tuple{<:Field{<:Flat,<:S0},<:Field{<:Flat,<:S2}},s::Symbol) = startswith(string(s),"T") ? f.f1[s] : f.f2[s]
    
    

# these are needed for StaticArrays to invert the 2x2 TE block matrix correctly
# we can hopefully remove this pending some sort of PR into StaticArrays to
# remove the need for this
one(::Type{Diagonal{T}}) where {T} = Diagonal(Vector{T}(0))
zero(::Type{Diagonal{T}}) where {T} = Diagonal(Vector{T}(0))

# this version puts Inf on diagonal for inverted 0 entries, the default throws a Singular error
inv(d::Diagonal) = Diagonal(1./d.diag)
