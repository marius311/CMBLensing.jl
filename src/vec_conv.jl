
# allow converting Fields and LinearOp to/from vectors
getindex(f::Field, i::Colon) = tovec(f)
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
~(f::Field) = (f,)

eltype(lz::LazyVecApply) = lz.eltype
size(lz::LazyVecApply) = lz.size
size(lz::LazyVecApply, d) = d<=2 ? lz.size[d] : 1
getindex{P,S,B}(vec::AbstractVector, s::Tuple{Field{P,S,B}}) = fromvec(typeof(s[1]),vec,meta(s[1])...)
