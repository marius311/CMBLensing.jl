import Base.Broadcast: BroadcastStyle, materialize, materialize!, broadcastable
using Base.Broadcast: Broadcasted, Style, flatten, DefaultArrayStyle

### broadcasting over combinations of Scalars, Fields, and LinDiagOps

# Broadcasted expressions are evaluated in four phases, hooking into the Julia
# broadcasting API (https://docs.julialang.org/en/latest/manual/interfaces/#man-interfaces-broadcasting-1)
# The four phases are:

# (1) Infer the type of the result, e.g. Field+Scalar=Field,
# LinDiagOp*Field=Field, or LinDiagOp+LinDiagOp=LinDiagOp. This is what the
# BroadcastStyle definitions below do, and they return a Style{F} where F is the
# final result type.
broadcastable(f::Union{Field,LinOp}) = f
BroadcastStyle(::Type{F}) where {F<:Union{Field,LinOp}} = Style{F}()
BroadcastStyle(::Style{F}, ::DefaultArrayStyle{0}) where {F<:Union{Field,LinOp}}   = Style{F}()
BroadcastStyle(::Style{F}, ::DefaultArrayStyle{n}) where {F<:Union{Field,LinOp},n} = Style{DefaultArrayStyle{n}}()
BroadcastStyle(::Style{F}, ::Style{<:LinOp}) where {F<:Field} = Style{F}()
BroadcastStyle(::Style{F0}, ::Style{F2}) where {P,F0<:Field{Map,S0,P},F2<:Field{QUMap,S2,P}} = Style{F2}()
BroadcastStyle(::Style{F},  ::Style{F})  where {F<:Field} = Style{F}()

# (2) Call broadcast_data(F,⋅) on each of the arguments being broadcasted over
# to get the actual data which participates in the broadcast. This should return
# a tuple and in the end we broadcast over these tuples. Different types can
# specialize to return different things for different F's, e.g. ∂x returns a
# different sized array depending on the Nside of F. These are a few generic
# definitions:
broadcast_data(::Type{F}, f::F) where {F<:Field} = fieldvalues(f)
broadcast_data(::Type{F}, L::FullDiagOp{F}) where {F<:Field} = broadcast_data(F, L.f)
broadcast_data(::Type{<:Field}, s::Scalar) = s
broadcast_data(::Any, x::Ref) = (x,) 


# (3) Recursively reduce over any metadata that the fields may have
metadata(::Type{<:Field}, ::Any) = ()
metadata_reduce(m) = m
metadata_reduce(::Tuple{}, m::Tuple) = m
metadata_reduce(m::Tuple, ::Tuple{}) = m
metadata_reduce(::Tuple{}, ::Tuple{}) = ()
metadata_reduce(bc::Broadcasted) = metadata_reduce(map(metadata_reduce, bc.args)...)
metadata_reduce(a,b,c...) = metadata_reduce(metadata_reduce(a,b),c...)


# (4) Finally, we intercept the broadcast machinery at the materialize function,
# and modify the Broadcasted object there to replace all the args with
# broadcast_data(F,arg), and then forward the broadcast one level deeper to
# the tuples returned by broadcast_data.

# recursively replaces all arguments in a Broadcasted object with new_arg(arg)
# e.g. replace_bc_args(Broadcasted(+,(1,2)), x->2x) = Broadcasted(+,(2,4))
replace_bc_args(bc::Broadcasted{S}, new_arg) where {S} = Broadcasted{S}(bc.f, map(replace_bc_args, bc.args, map(_->new_arg, bc.args)))
replace_bc_args(arg, new_arg) = new_arg(arg)
# forward the broadcast one level "deeper", eg here is what happens when you
# then apply materialize to the output of this function:
# materialize(deepen_bc(Broadcasted(+,((1,2),(3,4))))) =  (Broadcasted(+,(1,3)), Broadcasted(+,(2,4)))
deepen_bc(bc::Broadcasted) = Broadcasted((x...)->Broadcasted(bc.f, tuple(x...)), map(deepen_bc, bc.args))
deepen_bc(x) = x

# now the custom materialize functions, these ones for when the result type is a Field
function _materialize(bc::Broadcasted{Style{F}}) where {F<:Field}
    meta = metadata_reduce(replace_bc_args(bc, arg->metadata(F,arg)))
    bc′ = materialize(deepen_bc(replace_bc_args(bc, arg->broadcast_data(F,arg))))
    meta, bc′
end
function materialize(bc::Broadcasted{Style{F}}) where {F<:Field}
    meta, bc′ = _materialize(bc)
    F(map(materialize, bc′)..., meta...)
end
function materialize!(dest::F, bc::Broadcasted{Style{F}}) where {F<:Field}
    meta, bc′ = _materialize(bc)
    @assert meta==metadata(F,dest)
    map(materialize!, broadcast_data(F,dest), bc′)
    dest
end
# and for when the result type is a FullDiagOp
materialize(bc::Broadcasted{<:Style{<:FullDiagOp{F}}}) where {F<:Field} = 
    FullDiagOp(materialize(convert(Broadcasted{Style{F}}, bc)))
materialize!(dest, bc::Broadcasted{<:Style{<:FullDiagOp{F}}}) where {F<:Field} = 
    (materialize!(dest.f, convert(Broadcasted{Style{F}}, bc)); dest)
# fallback for things we can't broadcast, e.g. `∂x .* ∂x` alone without a field
cant_broadcast() = error("Can't broadcast this expression.")
materialize(bc::Broadcasted{<:Style{<:LinDiagOp}}) where {F<:Field} = cant_broadcast()
materialize!(dest, bc::Broadcasted{<:Style{<:LinDiagOp}}) where {F<:Field} = cant_broadcast()



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

# F(f) where F is some Field type defaults to just using the basis conversion
# and asserting that we end up with the right type, F
convert(::Type{F}, f::Field{B1}) where {B1,B2,F<:Field{B2}} = B2(f)::F

# this used to be the default in 0.6, bring it back because we use F(f) alot to
# mean convert f to a type of F
(::Type{F})(f::Field) where {F<:Field} = convert(F,f)



### lazy evaluation

# we use LazyBinaryOps to create new operators composed from other operators
# which don't actually evaluate anything until they've been multiplied by a
# field
struct LazyBinaryOp{F,A<:Union{LinOp,Scalar},B<:Union{LinOp,Scalar}} <: LinOp{Basis,Spin,Pix}
    a::A
    b::B
    LazyBinaryOp(op,a::A,b::B) where {A,B} = new{op,A,B}(a,b)
end
# creating LazyBinaryOps
for op in (:+, :-, :*)
    @eval ($op)(a::LinOp,  b::LinOp)  = LazyBinaryOp($op,a,b)
    @eval ($op)(a::LinOp,  b::Scalar) = LazyBinaryOp($op,a,b)
    @eval ($op)(a::Scalar, b::LinOp)  = LazyBinaryOp($op,a,b)
end
/(op::LinOp, n::Real) = LazyBinaryOp(/,op,n)
literal_pow(::typeof(^), op::LinOp, ::Val{-1}) = inv(op)
literal_pow(::typeof(^), op::LinOp, ::Val{n}) where {n} = LazyBinaryOp(^,op,n)
inv(op::LinOp) = LazyBinaryOp(^,op,-1)
-(op::LinOp) = -1 * op
# evaluating LazyBinaryOps
for op in (:+, :-)
    @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
end
*(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
*(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)
*(lz::LazyBinaryOp{^}, f::Field) = foldr((lz.b>0 ? (*) : (\)), fill(lz.a,abs(lz.b)), init=f)
adjoint(lz::LazyBinaryOp{F}) where {F} = LazyBinaryOp(F,adjoint(lz.b),adjoint(lz.a))
ud_grade(lz::LazyBinaryOp{op}, args...; kwargs...) where {op} = LazyBinaryOp(op,ud_grade(lz.a,args...;kwargs...),ud_grade(lz.b,args...;kwargs...))

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


include("broadcast_expand.jl")

### linear algebra of Vectors and Matrices of Fields

# alot of work needed here to make various StaticArray stuff work / infer
# correctly... maybe at some point evaluate if its really worth it?


const FieldVector = StaticVector{2,<:Union{Field,LinOp}}
const FieldRowVector = Adjoint{<:Union{Field,LinOp},<:FieldVector}
const FieldMatrix = StaticMatrix{2,2,<:Union{Field,LinOp}}
const FieldArray = Union{FieldVector, FieldRowVector, FieldMatrix}
# useful since v .* f is not type stable
*(v::FieldVector, f::Field) = @SVector[v[1]*f, v[2]*f]
*(f::Field, v::FieldVector) = @SVector[f*v[1], f*v[2]]
# until StaticArrays better implements adjoints
*(v::FieldRowVector, M::FieldMatrix) = @SVector[v'[1]*M[1,1] + v'[1]*M[2,1], v'[2]*M[1,2] + v'[2]*M[2,2]]'
# and until StaticArrays better implements invereses... 
function inv(m::FieldMatrix)
    a,b,c,d = m
    invdet = @. 1/(a*d-b*c)
    @. @SMatrix [invdet*d -invdet*b; -invdet*c invdet*a]
end

# helps StaticArrays infer various results correctly:
promote_rule(::Type{F}, ::Type{<:Scalar}) where {F<:Field} = F
arithmetic_closure(::F) where {F<:Field} = F
using LinearAlgebra: matprod
Base.promote_op(::typeof(adjoint), ::Type{T}) where {T<:∇i} = T
Base.promote_op(::typeof(matprod), ::Type{<:∇i}, ::Type{<:F}) where {F<:Field} = Base._return_type(*, Tuple{∇i{0,true},F})

ud_grade(s::Scalar, args...; kwargs...) = s
