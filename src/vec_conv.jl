

# allow converting Fields and LinearOp to/from vectors


# f[:] or [a,b,c][:] where f,a,b,c are Fields gives you a single vector representation.
getindex(f::Field, i::Colon) = tovec(f)
getindex{F<:Field}(arr::Array{F},::Colon) = vcat((arr[i][:] for i=eachindex(arr))...)


# x[~f] or x[~(a,b,c)] converts vector x back into fields of type f or a,b,c,
# respectively, assuming the vector is the right length
~((args::Field)...) = map(typeof,args)
function getindex{T<:Number,N}(v::AbstractVector{T}, i::NTuple{N,DataType})
    lengths = [map(length,i)...]
    starts = 1+[0; cumsum(lengths)[1:end-1]]
    [fromvec(t,v[s:s+l-1]) for (s,l,t) in zip(starts,lengths,i)]
end
getindex{T<:Number}(v::AbstractVector{T},i::Tuple{DataType}) = fromvec(i[1],v)

# each field defines length(::Type{F}), this lets us call length() on a Field object itself
length{F<:Field}(f::F) = length(F)


# convert operators of Fields to operators on vectors
getindex(op::LinearOp, i::Tuple{DataType}) = LazyVecApply{i[1]}(op)
immutable LazyVecApply{F}
    op::LinearOp
end
*{F}(lazy::LazyVecApply{F}, vec::AbstractVector) = convert(F,lazy.op*fromvec(F,vec))[:]
eltype{F}(lz::LazyVecApply{F}) = eltype(F)
size{F}(lz::LazyVecApply{F}) = (length(F),length(F))
size{F}(lz::LazyVecApply{F}, d) = d<=2 ? length(F) : 1
