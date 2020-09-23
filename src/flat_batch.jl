# functions specific to creating "batched" fields, which are fields that
# simultaneously store and operate on several maps at a time. this is mainly
# useful on GPUs where operating on batches of fields is sometimes no slower
# than a single field. 

"""
    batch(fs::FlatField...)
    batch(fs::Vector{<:FlatField})
    batch(fs::TUple{<:FlatField})
    
Turn a length-N array of `FlatField`'s into a single batch-length-N `FlatField`.
For the inverse operation, see [`unbatch`](@ref). 
"""
batch(f::F) where {N,θ,∂m,F<:FlatS0{<:Flat{N,θ,∂m}}} = f
batch(fs::F...) where {N,θ,∂m,F<:FlatS0{<:Flat{N,θ,∂m}}} = 
    basetype(F){Flat{N,θ,∂m,length(fs)}}(cat(map(firstfield,fs)..., dims=3))
batch(fs::F...) where {F<:Union{FlatS2,FlatS02}} =
    FieldTuple{basis(F)}(map(batch, map(firstfield,fs)...))
batch(fs::Union{Vector{<:FlatField},Tuple{<:FlatField}}) = batch(fs...)

"""
    batch(f::FlatField, D::Int)
    
Construct a batch-length-`D` `FlatField` from an unbatched `FlatField` which
will broadcast as if it were `D` copies of `f` (without actually making `D`
copies of the data in `f`)
"""    
batch(f::F, D::Int) where {N,θ,∂m,D′,F<:FlatS0{Flat{N,θ,∂m,D′}}} = 
    (D′==D || D′==1) ? basetype(F){Flat{N,θ,∂m,D}}(firstfield(f)) : error("Can't change batch-length from $(D′) to $D.")
batch(f::F, D::Int) where {F<:Union{FlatS2,FlatS02}} = FieldTuple{basis(F)}(map(f->batch(f,D), f.fs))
batch(x::T, D::Int) where {T} = D==1 ? x : error("batching $T not implemented.")
batch(x, ::Nothing) = x

"""
    batch_promote!(to, f)

Promote `f` to the same batch size as `to` by replication. If both are already
the same batch size, no copy is made and `f` is returned. If promotion needs to
happen, the answer is stored in-place in `to` and returned. 
"""
batch_promote!(to, f) = batchsize(to)==batchsize(f) ? f : (to .= f)

"""
    unbatch(f::FlatField)
    
If `f` is a batch-length-`D` field, return length-`D` vector of each batch
component, otherwise just return `f`. For the inverse operation, see
[`batch`](@ref).
"""
unbatch(f::FlatField{<:Flat{<:Any,<:Any,<:Any,1}}) = f
unbatch(f::FlatField{<:Flat{<:Any,<:Any,<:Any,D}}) where {D} = [batchindex(f,i) for i=1:D]

"""
    batchindex(f::FlatField, I)
    
Get the `I`th indexed batch (`I` can be a slice). 
"""
batchindex(f::F, I) where {N,θ,∂mode,P<:Flat{N,θ,∂mode},F<:FlatS0{P}} = 
    basetype(F){Flat{N,θ,∂mode,length(I)}}(f[:,:,I])
batchindex(f::FlatField, I) = 
    FieldTuple{basis(f)}(map(f->batchindex(f, I), f.fs))

"""
    batchsize(f::FlatField)
    
The number of batches of in this object.
"""
batchsize(::FlatField{<:Flat{<:Any,<:Any,<:Any,D}}) where {D} = D
batchsize(x) = 1



struct BatchedVals{T,D,V<:AbstractVector{T}} <: Real
    vals :: V
    BatchedVals(v::V) where {T,V<:AbstractVector{T}} = new{T,length(v),V}(v)
end

"""
    BatchedReal(::Vector{<:Real}) <: Real

Holds a vector of real numbers and broadcasts algebraic operations over them,
as well as broadcasting with batched `FlatField`s, but is itself a `Real`. 
"""
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
for op in [:+, :-, :*, :/, :<, :<=, :&, :|]
    @eval begin
        ($op)(a::BatchedReal, b::BatchedReal) = batch(broadcast(($op), a.vals, b.vals))
        ($op)(a::BatchedReal, b::Real)        = batch(broadcast(($op), a.vals, b))
        ($op)(a::Real,        b::BatchedReal) = batch(broadcast(($op), a,      b.vals))
    end
end
for op in [:-, :sqrt, :one, :zero, :isfinite, :eps]
    @eval ($op)(br::BatchedReal) = batch(broadcast(($op),br.vals))
end
for op in [:any, :all]
    @eval ($op)(br::BatchedReal) = ($op)(br.vals)
end
eltype(::BatchedVals{T}) where {T} = T
broadcastable(::Type{<:FlatS0{<:Flat,T}}, br::BatchedReal) where {T} = reshape(T.(br.vals),1,1,length(br.vals))
unbatch(br::BatchedVals) = br.vals
unbatch(r::Real) = r
Base.show(io::IO, br::BatchedReal) = print(io, "Batched", br.vals)
(::Type{T})(br::BatchedReal) where {T<:Real} = batch(T.(br.vals))
convert(::Type{<:BatchedVals{T,N}}, v::Bool) where {T,N} = batch(T(v),N)
Base.hash(bv::BatchedVals, h::UInt) = hash(bv.vals,hash(typeof(bv),h))

batch(L::Diagonal{<:Any,<:FlatField}, D::Int) = Diagonal(batch(diag(L), D))


"""
    batchmap(f, args...)

map function `f` over `args`, unbatching them first if they are batched, and
then batching the result.
"""
function batchmap(f, args...)
    _unbatch(x) = batchsize(x)==1 ? Ref(x) : unbatch(x)
    batch(broadcast(f, map(_unbatch, args)...)...)
end
