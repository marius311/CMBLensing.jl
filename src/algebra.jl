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
Try not using broadcasting or converting $F to the right basis by hand.") #fall-back

# get the type of the final result in a type stable way (adapted from broadcast.jl)
containertype(::F) where {F<:Field} = F
containertype(::T) where {T<:Union{LinDiagOp,Scalar}} = Any
containertype(ct1, ct2) = promote_containertype(containertype(ct1), containertype(ct2))
@inline containertype(ct1, ct2, cts...) = promote_containertype(containertype(ct1), containertype(ct2, cts...))
promote_containertype(::Type{F}, ::Type{F}) where {F<:Field} = F
promote_containertype(::Type{Any}, ::Type{Any}) = Any
@swappable promote_containertype{F<:Field}(::Type{F}, ::Type{Any}) = F
promote_containertype(::Type{F1}, ::Type{F2}) where {F1,F2} = error("Can't broadcast together $(shortname(F1)) and $(shortname(F2)). 
Try not using broadcasting or converting them to the same basis by hand.") #fall-back


# the actual broadcast functions which broadcast operations down to the
# underlying data as returned by broadcast_data. at least one Field must be
# present so we can infer the final type of the result
function broadcast(op, args::Union{_,Field,LinDiagOp,Scalar}...) where {_<:Field}
    F = containertype(args...)
    F(map(broadcast, repeated(op), map(broadcast_data, repeated(F), args)...)...)
end

broadcast!(op, X::Field, args::Union{Field,LinDiagOp,Scalar}...) = begin
    F = containertype(X)
    for (x,d)=zip(broadcast_data(F,X),zip(map(broadcast_data,repeated(F),args)...))
        broadcast!(op,x,d...)
    end 
    X
end


### old-style (slow) non-broadcasted algebra

for op in (:+,:-), (T1,T2) in ((:Field,:Scalar),(:Scalar,:Field),(:Field,:Field))
    @eval ($op)(a::$T1, b::$T2) = broadcast($(op),promote(a,b)...)
end
for op in (:*,:/)
    for (T1,T2) in ((:F,:Scalar),(:Scalar,:F),(:F,:F))
        @eval ($op)(a::$T1, b::$T2) where {F<:Field} = broadcast($(op),a,b)
    end
    @eval ($op)(a::Field, b::Field) = error("Fields must be put into same basis before they can be multiplied.")
end
^(f::Field,n::Real) = f.^n
^(f::Field,n::Int) = f.^n #needed to avoid ambiguity error
-(f::Field) = .-(f)
dot(a::Field,b::Field) = dot(promote(a,b)...)


### basis conversion

# B(f) where B is a basis converts f to that basis
(::Type{B})(f::Field{P,S,B}) where {P,S,B} = f
#todo: probably want to have convert(::Type{T},..)::T rather than the following...
convert(::Type{F}, f::Field{P,S,B1}) where {P,S,B1,B2,F<:Field{P,S,B2}} = B2(f)


# convert Fields to right basis before feeding into a LinOp
for op=(:*,:\)
    @eval @∷ ($op){B1,B2}(O::LinOp{∷,∷,B1}, f::Field{∷,∷,B2}) = $(op)(O,B1(f))
end


### lazy evaluation
struct LazyBinaryOp{Op,TA<:Union{LinOp,Scalar},TB<:Union{LinOp,Scalar}} <: LinOp{Pix,Spin,Basis}
    a::TA
    b::TB
    LazyBinaryOp(a::TA,::Op,b::TB) where {Op<:Function,TA,TB} = new{Op,TA,TB}(a,b)
    # maybe assert metadata is the same here? 
end

# do these ops lazily in these cases
for op in (:+, :-, :*)
    @eval ($op)(a::LinOp, b::LinOp) = LazyBinaryOp(a,$op,b)
    @eval @swappable ($op)(a::LinOp, b::Real) = LazyBinaryOp(a,$op,b)
end
/(op::LinOp, n::Real) = LazyBinaryOp(op,/,n)
-(op::LinOp) = LazyBinaryOp(-1,*,op)
^(op::LinOp, n::Int) = n<0 ? error("Can't raise $T to negative ($n) power") : n==0 ? 1 : n==1 ? op : *(repeated(op,n)...)

# evaluation rules when finally applying a lazy op to a field
for op in (:+, :-)
    @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
end
*(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
*(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)


### linear algebra of Vectors and Matrices of Fields
include("broadcast_expand.jl")
import Base: Ac_mul_B, A_mul_Bc, broadcast, transpose, inv, ctranspose

const Field2DVector = SVector{2,<:FieldOpScal}
const Field2DRowVector = RowVector{<:FieldOpScal,<:Field2DVector}
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

ctranspose(v::Field2DVector) = RowVector(v)

function inv(m::Field2DMatrix)
    a,b,c,d = m
    invdet = @. 1/(a*d-b*c)
    @. @SMatrix [invdet*d -invdet*b; -invdet*c invdet*a]
end


transpose(f::Union{Field,LinOp}) = f #todo: this should probably conjugate the field but need to think about exactly what that impacts....

# needed by ODE.jl
norm(f::Field) = +(norm.(broadcast_data(containertype(f),f))...)
isnan(::Field) = false
