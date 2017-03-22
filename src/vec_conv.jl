

# allow converting Fields and LinOp to/from vectors


# f[:] or [a,b,c][:] where f,a,b,c are Fields gives you a single vector representation.
getindex(f::Field, i::Colon) = tovec(f)
getindex{F<:Field}(arr::Array{F},::Colon) = vcat((arr[i][:] for i=eachindex(arr))...)


# x[~f] or x[~(a,b,c)] converts vector x back into fields of type f or a,b,c,
# respectively, assuming the vector is the right length
@generated ~(args::Field...) = :(Tuple{$(args...)})
@generated function getindex(v::AbstractVector{<:Number}, i::Type{S}) where {S<:NTuple}
    lengths = [map(length,S.parameters)...]
    starts = 1+[0; cumsum(lengths)[1:end-1]]
    :(tuple($((:(fromvec($t,v[$(s:s+l-1)])) for (s,l,t) in zip(starts,lengths,S.parameters))...)))
end
getindex(v::AbstractVector{<:Number},::Type{Tuple{F}}) where {F<:Field} = fromvec(F,v)

# each field defines length(::Type{F}), this lets us call length() on a Field object itself
length{F<:Field}(::F) = length(F)


# convert operators of Fields to operators on vectors
getindex(op::LinOp, i::Tuple{DataType}) = LazyVecApply{i[1]}(op)
struct LazyVecApply{F}
    op::LinOp
end
*{F}(lazy::LazyVecApply{F}, vec::AbstractVector) = convert(F,lazy.op*fromvec(F,vec))[:]
eltype{F}(lz::LazyVecApply{F}) = eltype(F)
size{F}(lz::LazyVecApply{F}) = (length(F),length(F))
size{F}(lz::LazyVecApply{F}, d) = d<=2 ? length(F) : 1
