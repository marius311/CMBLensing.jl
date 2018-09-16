import Base.Broadcast: BroadcastStyle, materialize, materialize!, broadcastable
using Base.Broadcast: Broadcasted, Style, flatten, DefaultArrayStyle

### broadcasting over combinations of Scalars, Fields, and LinDiagOps

# Broadcasted expressions are evaluated in three phases, hooking into the Julia
# broadcasting API (https://docs.julialang.org/en/latest/manual/interfaces/#man-interfaces-broadcasting-1)
# The three phases are:

# (1) Infer the type of the result, e.g. Field+Scalar=Field,
# LinDiagOp*Field=Field, or LinDiagOp+LinDiagOp=LinDiagOp. This is what the
# BroadcastStyle definitions below do, and they return a Style{F} where F is the
# final result type.
broadcastable(f::Union{Field,LinDiagOp}) = f
BroadcastStyle(::Type{F}) where {F<:Union{Field,LinDiagOp}} = Style{F}()
BroadcastStyle(::Style{F}, ::DefaultArrayStyle{0}) where {F<:Union{Field,LinDiagOp}} = Style{F}()
BroadcastStyle(::Style{F}, ::Style{<:LinDiagOp}) where {F<:Field} = Style{F}()
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

# (3) Finally, we intercept the broadcast machinery at the materialize function,
# and modify the Broadcasted object there to replace all the args with
# broadcast_data(F,arg), and then forward the broadcast one level deeper to
# the tuples returned by broadcast_data.

# recursively replaces all arguments in a Broadcasted object with new_arg(arg)
# e.g. replace_bc_args(Broadcasted(+,(1,2)), x->2x) = Broadcasted(+,(2,4))
replace_bc_args(bc::Broadcasted, new_arg) = Broadcasted(bc.f, map(replace_bc_args, bc.args, map(_->new_arg, bc.args)))
replace_bc_args(arg, new_arg) = new_arg(arg)
# forward the broadcast one level "deeper", eg here is what happens when you
# then apply materialize to the output of this function:
# materialize(deepen_bc(Broadcasted(+,((1,2),(3,4))))) =  (Broadcasted(+,(1,3)), Broadcasted(+,(2,4)))
deepen_bc(bc::Broadcasted) = Broadcasted((x...)->Broadcasted(bc.f, tuple(x...)), map(deepen_bc, bc.args))
deepen_bc(x) = x

# now the custom materialize functions, these ones for when the result type is a Field
function materialize(bc::Broadcasted{Style{F}}) where {F<:Field}
    rbc = replace_bc_args(bc, arg->broadcast_data(F,arg))
    F(map(materialize, materialize(deepen_bc(rbc)))...)
end
function materialize!(dest::F, bc::Broadcasted) where {F<:Field}
    rbc = replace_bc_args(bc, arg->broadcast_data(F,arg))
    map(materialize!, broadcast_data(F,dest), materialize(deepen_bc(rbc)))
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
    @eval ($op)(a::$T1, b::$T2) = broadcast($op,($T1==$T2?promote:tuple)(a,b)...)
end
for op in (:*,:/), (T1,T2) in ((:F,:Scalar),(:Scalar,:F),(:F,:F))
    @eval ($op)(a::$T1, b::$T2) where {F<:Field} = broadcast($(op),a,b)
end
-(f::Field) = .-(f)
dot(a::Field,b::Field) = dot(promote(a,b)...)



### transposing

# our fields implicitly are column vectors, so transposing them technically
# should turn them into some sort of row vector object, but we can always tell
# if a field is supposed to be transposed depending on if its to the left or
# right of an operator. e.g. in x * Op its clear x is a transposed field
# (otherwise the expression doesn't make sense). since we can always infer this,
# we don't actually have a "TransposedField" object or anything like that.
adjoint(f::Field) = f

### basis conversion

# B(f) where B is a basis converts f to that basis. This is the fallback if the
# field is already in the right basis.
(::Type{B})(f::Field{B}) where {B} = f

# F(f) where F is some Field type defaults to just using the basis conversion
# and asserting that we end up with the right type, F
convert(::Type{F}, f::Field{B1}) where {B1,B2,F<:Field{B2}} = B2(f)::F

# this used to be the default in 0.6, bring it back because we use F(f) alot to
# mean convert f to a type of F
(::Type{F})(f) where {F<:Field} = convert(F,f)



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
for op in (:+, :-, :*, :Ac_mul_B)
    @eval ($op)(a::LinOp,  b::LinOp)  = LazyBinaryOp($op,a,b)
    @eval ($op)(a::LinOp,  b::Scalar) = LazyBinaryOp($op,a,b)
    @eval ($op)(a::Scalar, b::LinOp)  = LazyBinaryOp($op,a,b)
end
/(op::LinOp, n::Real) = LazyBinaryOp(/,op,n)
^(op::LinOp, n::Int) = LazyBinaryOp(^,op,n)
-(op::LinOp) = -1 * op
# evaluating LazyBinaryOps
for op in (:+, :-)
    @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
end
*(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
*(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)
*(lz::LazyBinaryOp{Ac_mul_B}, f::Field) = Ac_mul_B(lz.a,lz.b*f)
*(lz::LazyBinaryOp{^}, f::Field) = foldr((lz.b>0 ? (*) : (\)), f, fill(lz.a,abs(lz.b)))
adjoint(lz::LazyBinaryOp{F}) where {F} = LazyBinaryOp(F,adjoint(lz.b),adjoint(lz.a))
ud_grade(lz::LazyBinaryOp{op}, args...; kwargs...) where {op} = LazyBinaryOp(op,ud_grade(lz.a,args...;kwargs...),ud_grade(lz.b,args...;kwargs...))

# a generic lazy adjoint
struct LazyHermitian{A<:LinOp} <: LinOp{Basis,Spin,Pix}
    a::A
end
adjoint(L::LinOp) = LazyHermitian(L)
adjoint(L::LazyHermitian) = L.a
*(L::LazyHermitian, f::Field) = L.a'*f
inv(L::LazyHermitian) = LazyHermitian(inv(L))
ud_grade(lz::LazyHermitian, args...; kwargs...) = LazyHermitian(ud_grade(lz.a,args...; kwargs...))



### linear algebra of Vectors and Matrices of Fields

include("broadcast_expand.jl")

const Field2DVector = SVector{2,<:FieldOpScal}
const Field2DRowVector = Adjoint{<:FieldOpScal,<:Field2DVector}
const Field2DMatrix = SMatrix{2,2,<:FieldOpScal}

const 𝕀 = @SMatrix [1 0; 0 1]
⨳(a::Field2DMatrix, b::Field2DVector) = @. @SVector [a[1,1]*b[1]+a[1,2]*b[2], a[2,1]*b[1]+a[2,2]*b[2]]
⨳(a::Field2DRowVector, b::Field2DMatrix) = ((b') ⨳ (a'))' 
⨳(a::Field2DRowVector, b::Field2DVector) = @. a[1]*b[1] + a[2]*b[2]
⨳(a::Field2DVector, b::Field2DRowVector) = @SMatrix [a[1]*b[1] a[1]*b[2]; a[2]*b[1] a[2]*b[2]]
function ⨳(a::Field2DMatrix, b::Field2DMatrix)
    @. @SMatrix [(a[1,1]*b[1,1]+a[1,2]*b[2,1]) (a[1,1]*b[1,2]+a[1,2]*b[2,2]);
                 (a[2,1]*b[1,1]+a[2,2]*b[2,1]) (a[2,1]*b[1,2]+a[2,2]*b[2,2])]
end

*(a::Field2DVector, f::Field) = broadcast(*,a[1],f)#@SVector [a[1]*f, a[2]*f]
*(f::Field, a::Field2DVector) = @SVector [f*a[1], f*a[2]]
Ac_mul_B(f::Field, a::Field2DVector) = @SVector [f'*a[1], f'*a[2]]

# need this for type stability when converting bases of StaticArrays, seems like
# maybe a StaticArrays bug....
broadcast(::Type{B},a::StaticArray) where {B<:Basis} = map(x->B(x),a)

function inv(m::Field2DMatrix)
    a,b,c,d = m
    invdet = @. 1/(a*d-b*c)
    @. @SMatrix [invdet*d -invdet*b; -invdet*c invdet*a]
end

# needed by ODE.jl
norm(f::Field) = +(norm.(broadcast_data(containertype(f),f))...)
isnan(::Field) = false

ud_grade(s::Scalar, args...; kwargs...) = s
