
### broadcasting over combinations of Scalars, Fields, and LinDiagOps

# 
# CMBLensing broadcasting basically works by turning expressions like:
# 
# ∂ * f + g
# 
# into:
# 
# broadcast_data(F,∂) .* broadcast_data(F,f) .+ broadcast_data(F,g)
# 
# where F is the inferred result type of the expression. For fields,
# `broadcast_data` just returns the matrix storing the map(s). The fact that F
# appears as an argument to `broadcast_data` means that things like derivative
# operators, ∂, can return an array appropriate for a given result type. 
# 


# We hook into the Julia broadcasting API (https://docs.julialang.org/en/latest/manual/interfaces/#man-interfaces-broadcasting-1)

# (1) Infer the type of the result, e.g. Field+Scalar=Field,
# LinDiagOp*Field=Field, or LinDiagOp+LinDiagOp=LinDiagOp. This is what the
# BroadcastStyle definitions below do, and they return a Style{F} where F is the
# final result type.
broadcastable(f::FieldOrOp) = f
BroadcastStyle(::Type{F}) where {F<:FieldOrOp} = Style{F}()
BroadcastStyle(::Style{F}, ::DefaultArrayStyle{0}) where {F<:FieldOrOp}   = Style{F}()
BroadcastStyle(::Style{F}, ::DefaultArrayStyle{n}) where {F<:FieldOrOp,n} = DefaultArrayStyle{n}()
BroadcastStyle(::Style{F}, ::Style{<:LinOp}) where {F<:Field} = Style{F}()
BroadcastStyle(::Style{F0}, ::Style{F2}) where {P,F0<:Field{Map,S0,P},F2<:Field{QUMap,S2,P}} = Style{F2}()
BroadcastStyle(::Style{F},  ::Style{F})  where {F<:Field} = Style{F}()

# (2) Call broadcast_data(F,⋅) on each of the arguments being broadcasted over
# to get the actual data which participates in the broadcast. This should return
# a tuple and in the end we broadcast over these tuples. Different types can
# specialize to return different things for different F's, e.g. ∂x returns a
# different sized array depending on the Nside of F. These are a few generic
# definitions:
broadcast_data(f::F) where {F<:FieldOrOp} = broadcast_data(F,f)
broadcast_data(::Type{F}, f::F) where {F<:FieldOrOp} = fieldvalues(f)
broadcast_data(::Type{F}, L::FullDiagOp{F}) where {F<:FieldOrOp} = broadcast_data(F, L.f)
broadcast_data(::Type{<:FieldOrOp}, s::Scalar) = s
broadcast_data(::Any, x::Ref) = (x,) 


# (3) Recursively reduce over any metadata that the fields may have
# todo: should probably switch to metadata_reduce(F, m1, m2) to allow easier
# customization for types F
metadata(::Type{<:FieldOrOp}, ::Any) = ()
metadata_reduce(m) = m
metadata_reduce(::Tuple{}, m::Tuple) = m
metadata_reduce(m::Tuple, ::Tuple{}) = m
metadata_reduce(::Tuple{}, ::Tuple{}) = ()
metadata_reduce(m1::Tuple, m2::Tuple) = m1==m2 ? m1 : error()
metadata_reduce(bc::Broadcasted) = metadata_reduce(map(metadata_reduce, bc.args)...)
metadata_reduce(a,b,c...) = metadata_reduce(metadata_reduce(a,b),c...)







# non-broadcasted algebra on fields just uses the broadcasted versions
# (although in a less efficient way than if you were to directly use
# broadcasting)
for op in (:+,:-), (T1,T2) in ((:Field,:Scalar),(:Scalar,:Field),(:Field,:Field))
    @eval ($op)(a::$T1, b::$T2) = broadcast($op,($T1==$T2 ? promote : tuple)(a,b)...)
end
for op in (:*,:/), (T1,T2) in ((:F,:Scalar),(:Scalar,:F),(:F,:F))
    @eval ($op)(a::$T1, b::$T2) where {F<:Field} = broadcast($(op),a,b)
end
-(f::Field) = .-(f)
dot(a::Field,b::Field) = dot(promote(a,b)...)


### basis conversion

# B(f) where B is a basis converts f to that basis. This is the fallback if the
# field is already in the right basis.
(::Type{B})(f::Field{B}) where {B} = f

# The abstract `Basis` type means "any basis", hence this conversion rule:
Basis(f::Field) = f

# B(f′, f) converts f to basis B and stores the result inplace in f′. If f is
# already in basis B, we just return f (but note, we never actually set f′ in
# this case, which is more efficient, but necessitates some care when using this
# construct)
(::Type{B})(f′::Field{B}, f::Field{B}) where {B} = f



# F(f) where F is some Field type defaults to just using the basis conversion
# and asserting that we end up with the right type, F
convert(::Type{F}, f::Field{B1}) where {B1,B2,F<:Field{B2}} = B2(f)::F

# this used to be the default in 0.6, bring it back because we use F(f) alot to
# mean convert f to a type of F
(::Type{F})(f::Field) where {F<:Field} = convert(F,f)



# a generic lazy adjoint
# note: the adjoint of a LinOp{B,S,P} is not necessarily ::LinOp{B,S,P}
# (consider e.g. the pixelization operator which produces a different P)
struct AdjOp{L<:LinOp} <: LinOp{Basis,Spin,Pix} 
    op::L
end
adjoint(L::LinOp) = AdjOp(L)
adjoint(L::AdjOp) = L.op
inv(L::AdjOp) = AdjOp(inv(L))
ud_grade(lz::AdjOp, args...; kwargs...) = AdjOp(ud_grade(lz.a,args...; kwargs...))


### linear algebra of Vectors and Matrices of Fields

# alot of work needed here to make various StaticArray stuff work / infer
# correctly... maybe at some point evaluate if its really worth it?


# useful since v .* f is not type stable
*(v::FieldVector, f::Field) = @SVector[v[1]*f, v[2]*f]
*(f::Field, v::FieldVector) = @SVector[f*v[1], f*v[2]]
# until StaticArrays better implements adjoints
*(v::FieldRowVector, M::FieldMatrix) = @SVector[v'[1]*M[1,1] + v'[2]*M[2,1], v'[1]*M[1,2] + v'[2]*M[2,2]]'
# and until StaticArrays better implements invereses... 
function inv(dst::FieldMatrix, src::FieldMatrix)
    a,b,c,d = src
    det⁻¹ = @. 1/(a*d-b*c)
    @. dst[1,1] =  det⁻¹*d
    @. dst[1,2] = -det⁻¹*b
    @. dst[2,1] = -det⁻¹*c
    @. dst[2,2] =  det⁻¹*a
    dst
end
mul!(f::Field, ::typeof(∇'), v::FieldVector) = f .= (∇*v[1])[1] .+ (∇*v[2])[2]

# helps StaticArrays infer various results correctly:
promote_rule(::Type{F}, ::Type{<:Scalar}) where {F<:Field} = F
arithmetic_closure(::F) where {F<:Field} = F
using LinearAlgebra: matprod
Base.promote_op(::typeof(adjoint), ::Type{T}) where {T<:∇i} = T
Base.promote_op(::typeof(matprod), ::Type{<:∇i}, ::Type{<:F}) where {F<:Field} = Base._return_type(*, Tuple{∇i{0,true},F})

ud_grade(s::Scalar, args...; kwargs...) = s
