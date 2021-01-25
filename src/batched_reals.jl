
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
batch(rs::Real...) = BatchedReal(rs)
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
for op in [:-, :!, :sqrt, :one, :zero, :isfinite, :eps]
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
Base.hash(bv::BatchedReal, h::UInt) = foldr(hash, (typeof(bv), bv.vals), init=h)


# used to denote a batch of things, no other functionality
struct BatchedVal{V<:Vector}
    vals :: V
end
batch(v::AbstractVector) = BatchedVal(v)
batch_index(bv::BatchedVal, I) = getindex(bv.vals, I)