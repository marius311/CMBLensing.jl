

# allow converting Fields and LinOp to/from vectors


# f[:] or [a,b,c][:] where f,a,b,c are Fields gives you a single vector representation.
getindex(arr::Array{F},::Colon) where {F<:Field} = vcat((arr[i][:] for i=eachindex(arr))...)
getindex(t::NTuple{N,Field},::Colon) where {N} = vcat((t[i][:] for i=1:N)...)


# x[~f] or x[~(a,b,c)] converts vector x back into fields of type f or a,b,c,
# respectively, assuming the vector is the right length
@generated ~(args::Field...) = :(Tuple{$(args...)})
@generated function getindex(v::AbstractVector{<:Number}, i::Type{S}) where {S<:Tuple}
    lengths = [map(length,S.parameters)...]
    starts = 1+[0; cumsum(lengths)[1:end-1]]
    :(tuple($((:(fromvec($t,v[$(s:s+l-1)])) for (s,l,t) in zip(starts,lengths,S.parameters))...)))
end
getindex(v::AbstractVector{<:Number},::Type{Tuple{F}}) where {F<:Field} = fromvec(F,v)

# each field defines length(::Type{F}), this lets us call length() on a Field object itself
length(::F) where {F<:Field} = length(F)


# convert operators of Fields to operators on vectors
getindex(op::LinOp, ::Type{Tuple{F}}) where {F} = LazyVecApply{F}(op)
struct LazyVecApply{F}
    op::LinOp
end
*(lazy::LazyVecApply{F}, vec::AbstractVector) where {F} = convert(F,lazy.op*fromvec(F,vec))[:]
eltype(lz::LazyVecApply{F}) where {F} = eltype(F)
size(lz::LazyVecApply{F}) where {F} = (length(F),length(F))
size(lz::LazyVecApply{F}, d) where {F} = d<=2 ? length(F) : 1
