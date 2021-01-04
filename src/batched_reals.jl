# functions specific to creating "batched" fields, which are fields that
# simultaneously store and operate on several maps at a time. this is mainly
# useful on GPUs where operating on batches of fields is sometimes no slower
# than a single field. 

"""
    BatchedReal(::Vector{<:Real}) <: Real

Holds a vector of real numbers and broadcasts algebraic operations
over them, as well as broadcasting along the batch dimension of
`Field`s, but is itself a `Real`. 
"""
struct BatchedReal{T<:Real,V<:AbstractVector{T}} <: Real
    vals :: V
end

batch(r::Real) = r
batch(rs::Real...) = BatchedReal(collect(rs))
batch(v::AbstractVector) = BatchedReal(v)
getindex(br::BatchedReal, ::typeof(!), I) = getindex(br.vals, I)
batchlength(::BatchedReal) where {D} = D
for op in [:+, :-, :*, :/, :<, :<=, :&, :|, :(==)]
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
unbatch(br::BatchedReal) = br.vals
unbatch(r::Real) = r
Base.show(io::IO, br::BatchedReal) = print(io, "Batched", br.vals)
(::Type{T})(br::BatchedReal) where {T<:Real} = batch(T.(br.vals))
convert(::Type{<:BatchedReal{T,N}}, v::Bool) where {T,N} = batch(T(v),N)
Base.hash(bv::BatchedReal, h::UInt) = hash(bv.vals,hash(typeof(bv),h))
batch(Ls::Vector{<:Diagonal{<:Any,<:Field}}) = Diagonal(batch(map(diag,Ls)))