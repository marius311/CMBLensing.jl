

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
    [(println(s,l,t); fromvec(t,v[s:s+l-1])) for (s,l,t) in zip(starts,lengths,i)]
end
getindex{T<:Number}(v::AbstractVector{T},i::Tuple{DataType}) = fromvec(i[1],v)

# each field defines length(::Type{F}), this lets us call length() on a Field object itself
length{F<:Field}(f::F) = length(F)


# some of this may be broken....
function getindex{P,S,B}(op::LinearOp, f::Tuple{Field{P,S,B}})
    v = f[1][:]
    LazyVecApply{typeof(f[1]),B}(op,eltype(v),tuple(fill(length(v),2)...))
end
immutable LazyVecApply{F,B}
    op::LinearOp
    eltype::Type
    size::Tuple
end
*{F,B}(lazy::LazyVecApply{F,B}, vec::AbstractVector) = tovec(B(lazy.op*fromvec(F,vec,meta(lazy.op)...)))
eltype(lz::LazyVecApply) = lz.eltype
size(lz::LazyVecApply) = lz.size
size(lz::LazyVecApply, d) = d<=2 ? lz.size[d] : 1
