# functions specific to creating "batched" fields, which are fields that
# simultaneously store and operate on several maps at a time. this is mainly
# useful on GPUs where operating on batches of fields is sometimes no slower
# than a single field. 

@doc doc"""
    batch(fs::FlatField...)
    batch(fs::Vector{<:FlatField})
    batch(fs::TUple{<:FlatField})
    
Turn a length-N array of `FlatField`'s into a single batch-length-N `FlatField`.
For the inverse operation, see [`unbatch`](@ref). 
"""
batch(fs::F...) where {N,θ,∂m,F<:FlatS0{<:Flat{N,θ,∂m}}} = 
    basetype(F){Flat{N,θ,∂m,length(fs)}}(cat(map(firstfield,fs)..., dims=3))
batch(fs::F...) where {F<:Union{FlatS2,FlatS02}} =
    FieldTuple{basis(F)}(map(batch, map(firstfield,fs)...))
batch(fs::Union{Vector{<:FlatField},Tuple{<:FlatField}}) = batch(fs...)

@doc doc"""
    batch(f::FlatField, D::Int)
    
Construct a batch-length-`D` `FlatField` from an unbatched `FlatField` which
will broadcast as if it were `D` copies of `f` (without actually making `D`
copies of the data in `f`)
"""    
batch(f::F, D::Int) where {N,θ,∂m,D′,F<:FlatS0{Flat{N,θ,∂m,D′}}} = 
    (D′==D || D′==1) ? basetype(F){Flat{N,θ,∂m,D}}(firstfield(f)) : error("Can't change batch-length from $(D′) to $D.")
batch(f::F, D::Int) where {F<:Union{FlatS2,FlatS02}} = FieldTuple{basis(F)}(map(f->batch(f,D), f.fs))
batch(x, ::Nothing) = x

@doc doc"""
    unbatch(f::FlatField)
    
If `f` is a batch-length-`D` field, return length-`D` vector of each batch
component, otherwise just return `f`. For the inverse operation, see
[`batch`](@ref).
"""
unbatch(f::FlatField{<:Flat{<:Any,<:Any,<:Any,1}}) = f
unbatch(f::FlatField{<:Flat{<:Any,<:Any,<:Any,D}}) where {D} = [batchindex(f,i) for i=1:D]

@doc """
    batchindex(f::FlatField, I)
    
Get the `I`th indexed batch (`I` can be a slice). 
"""
batchindex(f::F, I) where {N,θ,∂mode,P<:Flat{N,θ,∂mode},F<:FlatS0{P}} = 
    basetype(F){Flat{N,θ,∂mode,length(I)}}(f[:,:,I])
batchindex(f::FlatField, I) = 
    FieldTuple{basis(f)}(map(f->batchindex(f, I), f.fs))

@doc """
    batchlength(f::FlatField)
    
The number of batches of Fields in this object.
"""
batchsize(::FlatField{<:Flat{<:Any,<:Any,<:Any,D}}) where {D} = D

@doc doc"""
    BatchedReal(::Vector{<:Real}) <: Real

Holds a vector of real numbers and broadcasts algebraic operations over them,
as well as broadcasting with batched `FlatField`s, but is itself a `Real`. 
"""
struct BatchedVals{T,D,V<:AbstractVector{T}} <: Real
    vals :: V
    BatchedVals(v::V) where {T,V<:AbstractVector{T}} = new{T,length(v),V}(v)
end
const BatchedReal{D,V,T<:Real} = BatchedVals{T,D,V}
batch(r::Real) = r
batch(r::Real, D::Int) = batch(collect(repeated(r,D)))
batch(v::AbstractVector) = BatchedVals(v)
batch(rs::Real...) = BatchedVals(collect(rs))
batchindex(br::BatchedVals, I) = getindex(br.vals,I)
batchsize(::BatchedVals{<:Any,D}) where {D} = D
struct BatchedRealStyle{D} <: AbstractArrayStyle{0} end
BroadcastStyle(::Type{<:BatchedReal{D}}) where {D} = BatchedRealStyle{D}()
BroadcastStyle(::FlatS0Style{F,M}, ::BatchedRealStyle{D′}) where {D′,N,θ,∂m,D,M,F<:FlatS0{Flat{N,θ,∂m,D}}} = 
    (D==1 || D′==1 || D==D′) ? FlatS0Style{basetype(F){Flat{N,θ,∂m,max(D,D′)}},M}() : Broadcast.Unknown
BroadcastStyle(::FieldTupleStyle{B,Names,FS}, S2::BatchedRealStyle) where {B,Names,FS} = 
    FieldTupleStyle{B,Names,Tuple{map_tupleargs(S1->typeof(Broadcast.result_style(S1(),S2)), FS)...}}()
for op in [:+, :-, :*, :/]
    @eval begin
        ($op)(a::BatchedReal, b::BatchedReal) = batch(broadcast(($op), a.vals, b.vals))
        ($op)(a::BatchedReal, b::Real)        = batch(broadcast(($op), a.vals, b))
        ($op)(a::Real, b::BatchedReal)        = batch(broadcast(($op), a,      b.vals))
    end
end
-(br::BatchedReal) = batch(.-br.vals)
<(a::BatchedReal, b::BatchedReal) = all(a.vals .< b.vals)
<(a::BatchedReal, b::Real) = all(a.vals .< b)
sqrt(br::BatchedReal) = batch(sqrt.(br.vals))
eltype(::BatchedVals{T}) where {T} = T
broadcastable(::Type{<:FlatS0{<:Flat,T}}, br::BatchedReal) where {T} = reshape(T.(br.vals),1,1,length(br.vals))
one(br::BatchedReal) = batch(one.(br.vals))
unbatch(br::BatchedVals) = br.vals
unbatch(r::Real) = r
Base.show(io::IO, br::BatchedReal) = print(io, "Batched", br.vals)
(::Type{T})(br::BatchedReal) where {T<:Real} = batch(T.(br.vals))



batch(L::Diagonal{<:Any,<:FlatField}, D::Int) = Diagonal(batch(diag(L), D))
