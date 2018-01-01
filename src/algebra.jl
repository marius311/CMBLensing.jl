import Base: broadcast, broadcast!


### broadcasting over combinations of Scalars, Fields, and LinDiagOps

# scalars which are allowed in our expressions must be real because we
# implicitly assume our maps are real, and addition/multiplication by a complex
# number, even of the fourier transform, would break this.
const Scalar = Real
const FieldOpScal = Union{Field,LinOp,Scalar}

# the data which is broadcast over for Fields and Scalars
# (other objects can define their own methods for this, allowing a single type
# of object to be broadcast with many different types of Fields)
@generated fieldvalues(x) = :(tuple($((:(x.$f) for f=fieldnames(x))...)))
@generated function broadcast_length(::Type{F}) where {F} length(fieldnames(F)) end
broadcast_data(::Type{F}, n::Scalar) where {F<:Field} = repeated(n,broadcast_length(F))
broadcast_data(::Type{F}, f::F) where {F<:Field} = fieldvalues(f)
broadcast_data(::Type{F}, f::T) where {F,T} = error("Can't broadcast a $(shortname(T)) as a $(shortname(F)).
Try not using broadcasting or converting $(shortname(F)) to the right basis by hand.") #fall-back

# get the type of the final result in a type stable way (adapted from broadcast.jl)
containertype(::F) where {F<:Field} = F
containertype(::T) where {T<:Union{LinDiagOp,Scalar}} = Any
containertype(ct1, ct2) = promote_containertype(containertype(ct1), containertype(ct2))
@inline containertype(ct1, ct2, cts...) = promote_containertype(containertype(ct1), containertype(ct2, cts...))
promote_containertype(::Type{F}, ::Type{F}) where {F<:Field} = F
promote_containertype(::Type{Any}, ::Type{Any}) = Any
@symarg promote_containertype{F<:Field}(::Type{F}, ::Type{Any}) = F
promote_containertype(::Type{F1}, ::Type{F2}) where {F1,F2} = error("Can't broadcast together $(shortname(F1)) and $(shortname(F2)).
Try not using broadcasting or converting them to the same basis by hand.") #fall-back


# the actual broadcast functions which broadcast operations down to the
# underlying data as returned by broadcast_data. at least one Field must be
# present so we can infer the final type of the result
broadcast(op, args::Union{_,Field,LinDiagOp,Scalar}...) where {_<:Field} = _broadcast(op,args...)
# if there's no fields but at least one FullDiagOp we can still infer the final
# type (which will be a FullDiagOp of some kind)
broadcast(op, args::Union{_,LinDiagOp,Scalar}...) where {_<:FullDiagOp} = FullDiagOp(_broadcast(op,args...))
function _broadcast(op, args...)
    F = containertype(args...)
    F(@tmap(broadcast, repeated(op), map(broadcast_data, repeated(F), args)...)...)
end

broadcast!(op, X::Field, args::Union{Field,LinDiagOp,Scalar}...) = begin
    F = containertype(X)
    @tmap(broadcast!, repeated(op), broadcast_data(F,X), map(broadcast_data, repeated(F), args)...)
    X
end


### old-style (slow) non-broadcasted algebra

for T in (:Field,:LinDiagOp), op in (:+,:-), (T1,T2) in ((T,:Scalar),(:Scalar,T),(T,T))
    @eval ($op)(a::$T1, b::$T2) = broadcast($(op),promote(a,b)...)
end
for op in (:*,:/)
    for (T1,T2) in ((:T,:Scalar),(:Scalar,:T),(:T,:T)), T in (:Field,:LinDiagOp)
        @eval ($op)(a::$T1, b::$T2) where {T<:$T} = broadcast($(op),a,b)
    end
end
^(f::Field,n::Real) = f.^n
^(f::Field,n::Int) = f.^n #needed to avoid ambiguity error
-(f::Field) = .-(f)
dot(a::Field,b::Field) = dot(promote(a,b)...)


### transposing

# our fields implicitly are column vectors, so transposing them technically
# should turn them into some sort of row vector object, but we can always tell
# if a field is supposed to be transposed depending on if its to the left or
# right of an operator. e.g. in x * Op its clear x is a transposed field
# (otherwise the expression doesn't make sense). since we can always infer this,
# we don't actually have a "TransposedField" object or anything like that.
transpose(f::Field) = f

# there is one exception, sometimes we write f1' * f2 with f1 and f2 as Fields
# (this comes up in the transposed lensing operators). for S0 this means Tx .*
# Tx, for S2 it means Qx .* Qx + Ux .* Ux, etc... in these cases, we overload
# the '* operator (Ac_mul_B). since we don't have a TransposedField object, this
# means we can't store the transpose of a field then later multiply it, i.e.
# `y=x'; y*x` doesn't work. in this case, the `y*x` does *not* do this transpose
# multiplication.
# this is the fallback:
# Ac_mul_B(x::Field, y::Field) = x*y


### basis conversion

# B(f) where B is a basis converts f to that basis. This is the fallback if the
# field is already in the right basis.
@∷ (::Type{B})(f::Field{∷,∷,B}) where {B} = f

# F(f) where F is some Field type defaults to just using the basis conversion
# and asserting that we end up with the right type, F
@∷ convert(::Type{F}, f::Field{∷,∷,B1}) where {B1,B2,F<:Field{∷,∷,B2}} = B2(f)::F



### lazy evaluation

# we use LazyBinaryOps to create new operators composed from other operators
# which don't actually evaluate anything until they've been multiplied by a
# field
struct LazyBinaryOp{F,A<:Union{LinOp,Scalar},B<:Union{LinOp,Scalar}} <: LinOp{Pix,Spin,Basis}
    a::A
    b::B
    LazyBinaryOp(op,a::A,b::B) where {A,B} = new{op,A,B}(a,b)
end
# creating LazyBinaryOps
for op in (:+, :-, :*, :Ac_mul_B)
    @eval         ($op)(a::LinOp, b::LinOp)  = LazyBinaryOp($op,a,b)
    @eval @symarg ($op)(a::LinOp, b::Scalar) = LazyBinaryOp($op,a,b)
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
ctranspose(lz::LazyBinaryOp{F}) where {F} = LazyBinaryOp(F,ctranspose(lz.b),ctranspose(lz.a))
ud_grade(lz::LazyBinaryOp{op}, args...; kwargs...) where {op} = LazyBinaryOp(op,ud_grade(lz.a,args...;kwargs...),ud_grade(lz.b,args...;kwargs...))

# a generic lazy ctranspose
struct LazyHermitian{A<:LinOp} <: LinOp{Pix,Spin,Basis}
    a::A
end
ctranspose(L::LinOp) = LazyHermitian(L)
ctranspose(L::LazyHermitian) = L.a
*(L::LazyHermitian, f::Field) = L.a'*f
inv(L::LazyHermitian) = LazyHermitian(inv(L))
ud_grade(lz::LazyHermitian, args...; kwargs...) = LazyHermitian(ud_grade(lz.a,args...; kwargs...))



### linear algebra of Vectors and Matrices of Fields

include("broadcast_expand.jl")

const Field2DVector = SVector{2,<:FieldOpScal}
const Field2DRowVector = RowVector{<:FieldOpScal,<:ConjArray{<:FieldOpScal,1,<:Field2DVector}}
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

*(a::Field2DVector, f::Field) = @SVector [a[1]*f, a[2]*f]
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
