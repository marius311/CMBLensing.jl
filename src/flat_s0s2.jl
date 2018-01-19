export FlatIQUMap, FlatTEBFourier

const FlatIQUMap{T,P} = Field2Tuple{FlatS0Map{T,P},FlatS2QUMap{T,P}}
const FlatTEBFourier{T,P} = Field2Tuple{FlatS0Fourier{T,P},FlatS2EBFourier{T,P}}
const FlatS02{T,P} = Field2Tuple{<:FlatS0{T,P},<:FlatS2{T,P}}

# some convenience constructors
FlatIQUMap{T,P}(i,q,u) where {T,P} = Field2Tuple(FlatS0Map{T,P}(i),FlatS2QUMap{T,P}(q,u))
FlatTEBFourier{T,P}(t,e,b) where {T,P} = Field2Tuple(FlatS0Fourier{T,P}(t),FlatS2EBFourier{T,P}(e,b))


struct FlatTEBCov{T,P} <: LinOp{Basis2Tuple{Fourier,EBFourier},S02,P}
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

function Cℓ_to_cov(::Type{T}, ::Type{P}, ::Type{S0}, ::Type{S2}, ℓ, CℓTT, CℓEE, CℓBB, CℓTE) where {T,P}
    FlatTEBCov{T,P}(Cℓ_2D.(P,[ℓ],[CℓTT,CℓTE,CℓEE,CℓBB])...)
end

ctranspose(Σ::FlatTEBCov) = Σ

function *(cv::FlatTEBCov{T,P}, f::FlatTEBFourier{T,P}) where {T,N,P<:Flat{<:Any,N}} 
    (t,e),b = (cv.ΣTE * [f.f1.Tl[:], f.f2.El[:]]), cv.ΣB .* f.f2.Bl
    FieldTuple(FlatS0Fourier{T,P}(reshape(t,N÷2+1,N)),FlatS2EBFourier{T,P}(reshape(e,N÷2+1,N),b))
end

inv(cv::FlatTEBCov{T,P}) where {T,P} = FlatTEBCov{T,P}((cv.unsafe_invert ? (nan2zero.(inv(cv.ΣTE)), nan2zero.(1./cv.ΣB)) : (inv(cv.ΣTE), 1/cv.ΣB))...)
sqrtm(cv::FlatTEBCov{T,P}) where {T,P} = FlatTEBCov{T,P}((cv.unsafe_invert ? nan2zero.(sqrtm(cv.ΣTE)) : sqrtm(cv.ΣTE)), sqrt.(cv.ΣB))

simulate(cv::FlatTEBCov{T,P}) where {T,P} = sqrtm(cv) * white_noise(FlatTEBFourier{T,P})

# arithmetic with FlatTEBCov and scalars
broadcast_data(::Type{FlatTEBCov},s::Scalar) = repeated(s)
broadcast_data(::Type{FlatTEBCov},cv::FlatTEBCov) = (cv.ΣTE, cv.ΣB)
function broadcast(f,args::Union{_,FlatTEBCov{T,P},Scalar}...) where {T,P,_<:FlatTEBCov{T,P}}
    FlatTEBCov{T,P}(map(broadcast, repeated(f), map(broadcast_data, repeated(FlatTEBCov), args)...)...)
end

# can do TEB * Diag{TEB} explicilty 
@symarg function *{T,P}(cv::FlatTEBCov{T,P}, d::FullDiagOp{<:FlatTEBFourier{T,P}})
    all(isreal.(d.f[:])) || error("Can't multiply TEB cov by non positive-definite operator.")
    t,e = Diagonal(real(d.f[:Tl][:])), Diagonal(real(d.f[:El][:]))
    FlatTEBCov{T,P}(
        @SMatrix([t*cv.ΣTE[1,1] t*cv.ΣTE[1,2];
                  e*cv.ΣTE[2,1] e*cv.ΣTE[2,2]]),
        real(d.f[:Bl]) .* cv.ΣB
    )
end

# non-broadcasted algebra on FlatTEBCov's
for op in (:*,:\,:/)
    @eval ($op)(L::FlatTEBCov, s::Scalar)           = broadcast($op,L,s)
    @eval ($op)(s::Scalar,     L::FlatTEBCov)       = broadcast($op,s,L)
end
for op in (:+,:-)
    @eval ($op)(La::F, Lb::F) where {F<:FlatTEBCov} = broadcast($op,La,Lb)
end
    
function get_Cℓ(f::Field2Tuple{<:FlatS0{T,P},<:FlatS2{T,P}}; which=(:TT,:TE,:EE,:BB), kwargs...) where {T,P}
    Cℓs = [get_Cℓ((FlatS0Fourier{T,P}(f[Symbol(x,:l)]) for x=xs)...; kwargs...) for xs in string.(which)]
    (Cℓs[1][1], hcat(last.(Cℓs)...))
end

    

# these are needed for StaticArrays to invert the 2x2 TE block matrix correctly
# we can hopefully remove this pending some sort of PR into StaticArrays to
# remove the need for this
one(::Type{Diagonal{T}}) where {T} = Diagonal(Vector{T}(0))
zero(::Type{Diagonal{T}}) where {T} = Diagonal(Vector{T}(0))
/(n::Number, d::Diagonal{<:Number}) = Diagonal(n./d.diag)

# this version puts Inf on diagonal for inverted 0 entries, the default throws a Singular error
inv(d::Diagonal) = Diagonal(1./d.diag)


getindex(f::FlatS02{T,P},::Type{Val{:T}}) where {T,P} = FlatS0Map{T,P}(f[:Tx])
getindex(f::FlatS02{T,P},::Type{Val{:E}}) where {T,P} = FlatS0Map{T,P}(f[:Ex])
getindex(f::FlatS02{T,P},::Type{Val{:B}}) where {T,P} = FlatS0Map{T,P}(f[:Bx])
