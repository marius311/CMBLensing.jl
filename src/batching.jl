
"""
    BatchedReal(::Vector{<:Real}) <: Real

Holds a vector of real numbers and broadcasts algebraic operations
over them, as well as broadcasting along the batch dimension of
`Field`s, but is itself a `Real`. 
"""
struct BatchedReal{T<:Real,V<:Vector{T}} <: Real
    vals :: V
end

batch(r::Real) = r
batch(rs::Real...) = BatchedReal(collect(rs))
batch(v::AbstractVector{<:Real}) = BatchedReal(collect(v))
batch_length(br::BatchedReal) = length(br.vals)
batch_length(::Real) = 1
batch_index(br::BatchedReal, I) = getindex(br.vals, I)
batch_index(r::Real, I) = r
for op in [:+, :-, :*, :/, :<, :<=, :&, :|, :(==), :≈]
    @eval begin
        ($op)(a::BatchedReal, b::BatchedReal) = batch(broadcast(($op), a.vals, b.vals))
        ($op)(a::BatchedReal, b::Real)        = batch(broadcast(($op), a.vals, b))
        ($op)(a::Real,        b::BatchedReal) = batch(broadcast(($op), a,      b.vals))
    end
end
for op in [:-, :!, :sqrt, :one, :zero, :isfinite, :eps, :exp, :log]
    @eval ($op)(br::BatchedReal) = batch(broadcast(($op),br.vals))
end
for op in [:any, :all]
    @eval ($op)(br::BatchedReal) = ($op)(br.vals)
end
eltype(::BatchedReal{T}) where {T} = T
unbatch(br::BatchedReal; dims=1) = reshape(br.vals, ntuple(_->1, dims-1)..., :)
unbatch(r::Real; dims=nothing) = r
Base.show(io::IO, br::BatchedReal) = print(io, "Batched", br.vals)
(::Type{T})(br::BatchedReal) where {T<:Real} = batch(T.(br.vals))
hash(bv::BatchedReal, h::UInt64) = foldr(hash, (typeof(bv), bv.vals), init=h)
hash(arr::AbstractArray{<:BatchedReal}, h::UInt64) = foldr(hash, arr, init=h)

# needed for ForwardDiff
one(::Type{<:BatchedReal{T}}) where {T} = one(T)
zero(::Type{<:BatchedReal{T}}) where {T} = zero(T)
convert(::Type{<:BatchedReal{T}}, x::T) where {T<:Real} = batch([x])


# used to denote a batch of things, no other functionality
struct BatchedVal{V<:Vector}
    vals :: V
end
batch(v::AbstractVector) = BatchedVal(v)
batch_index(bv::BatchedVal, I) = getindex(bv.vals, I)
batch_length(bv::BatchedVal) = length(bv.vals)


# mapping over a batch dimension
_batchargs(args, batch_size) = (map(batch∘collect, Base.Iterators.partition(arg, batch_size)) for arg in args)
batch_map(f, args...; batch_size) = mapreduce(unbatch, vcat, map(f, _batchargs(args, batch_size)...))
batch_pmap(f, args...; batch_size) = mapreduce(unbatch, vcat, pmap(f, _batchargs(args, batch_size)...))


# batched Tuples/NamedTuples
batch(ts::AbstractVector{<:Union{Tuple,NamedTuple}}) = map((t...) -> batch(collect(t)), ts...)
unbatch(t::Union{Tuple,NamedTuple}) = [map(x -> batch_index(x, i), t) for i=1:batch_length(t)]
batch_index(c::Union{Tuple,NamedTuple}, I) = map(x -> batch_index(x, I), c)
function batch_length(t::Union{Tuple,NamedTuple})
    reduce(map(batch_length, values(t))) do n, m
        n == m ? n : only(filter(!=(1), (n,m)))
    end
end

# batched ComponentArrays
function batch(cs::AbstractVector{<:ComponentArray})
    data = map(map(getdata, cs)...) do args...
        batch(collect(args))
    end
    axes = only(unique(map(getaxes, cs)))
    ComponentArray(data, axes)
end
function unbatch(c::ComponentArray)
    map(map(unbatch, getdata(c))...) do args...
        ComponentArray([args...], getaxes(c))
    end
end
function batch_length(c::ComponentArray)
    only(unique(filter(!=(1), map(batch_length, c))))
end
function batch_index(c::ComponentArray, I)
    map(x -> batch_index(x, I), c)
end