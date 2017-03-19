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
broadcast_data(::Type{F}, n::Scalar) where {F<:Field} = repeated(n)
broadcast_data(::Type{F}, f::F) where {F<:Field} = fieldvalues(f)

# fall-back
broadcast_data(::Type{F}, f::T) where {F,T} = error("Can't broadcast $T as a $F.
Try not using broadcasting or converting $F to the right basis by hand.")


# we have a separate promotion system for broadcasting, similar to the regular
# one, which allows a slightly different set of combinations of Fields to be
# broadcast together
broadcast_promote_type(::Type{<:Field}) = F
broadcast_promote_type(::Type{F},::Type{F}) where {F<:Field} = F
@swappable broadcast_promote_type{F<:Field,S<:Scalar}(::Type{F},::Type{S}) = F
@swappable broadcast_promote_type{D<:LinDiagOp,S<:Scalar}(::Type{D},::Type{S}) = D
broadcast_promote_type(a::Type,b::Type,cs::Type...) = broadcast_promote_type(broadcast_promote_type(a,b),cs...)
# LinDiagOps who's {P,S,B} are supertypes of the Field's {P,S,B} are OK. this
# lets through some bad cases though, which are caught by the fall-back
# broadcast_data above.
@swappable broadcast_promote_type{DP,DS,DB,D<:LinDiagOp{DP,DS,DB},FP<:DP,FS<:DS,FB<:DB,F<:Field{FP,FS,FB}}(::Type{D},::Type{F}) = F
# fall-back
broadcast_promote_type(::Type{T1},::Type{T2}) where {T1,T2} = error("Can't broadcast $T1 
and $T2 together. Try not broadcasting, or converting them to a different basis.") 


# the actual broadcast functions which broadcast operations down to the
# underlying data as returned by broadcast_data. at least one Field must be
# present so we can infer the final type of the result
broadcast(op, args::Union{_,Field,LinDiagOp,Scalar}...) where {_<:Union{Field,LinDiagOp}} = begin
    F = broadcast_promote_type(map(typeof,args)...)
    F((broadcast(op,d...) for d=zip(map(broadcast_data,repeated(F),args)...))...)::F
end
broadcast!(op, X::F, args::Union{Field,LinDiagOp,Scalar}...) where {F<:Field} = begin
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
(::Type{B}){P,S,B}(f::Field{P,S,B}) = f
convert(::Type{F}, f::Field{P,S,B1}) where {P,S,B1,B2,F<:Field{P,S,B2}} = B2(f)


# convert Fields to right basis before feeding into a LinOp
for op=(:*,:\)
    @eval ($op){P,S,B1,B2}(O::LinOp{P,S,B1}, f::Field{P,S,B2}) = $(op)(O,B1(f))
    @eval ($op){P,S,B}(O::LinOp{P,S,B}, f::Field{P,S,B}) = throw(MethodError($op,(O,f)))
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

×(a::Field2DMatrix, b::Field2DVector) = @. @SVector [a[1,1]*b[1]+a[1,2]*b[2], a[1,1]*b[1]+a[1,2]*b[2]]
×(a::Field2DRowVector, b::Field2DMatrix) = @. (@SVector [a[1,1]*b[1]+a[1,2]*b[2], a[1,1]*b[1]+a[1,2]*b[2]])'
×(a::Field2DRowVector, b::Field2DVector) = @. a[1]*b[1] + a[2]*b[2]
    
A_mul_Bc(a::Field2DVector, b::Field2DVector) = @SMatrix [a[1]*b[1] a[1]*b[2]; a[2]*b[1] a[2]*b[2]]
*(a::Field2DVector, f::Field) = @SVector [a[1]*f, a[2]*f]

ctranspose(v::Field2DVector) = RowVector(v)

function inv(m::Field2DMatrix)
    a,b,c,d = m
    invdet = @. 1/(a*d-b*c)
    @. @SMatrix [invdet*d -invdet*b; -invdet*c invdet*a]
end

const 𝕀 = @SMatrix [1 0; 0 1]

# (::Type{B}){B<:Basis,F<:Field}(a::AbstractArray{F}) = map(B,a)
transpose(f::Union{Field,LinOp}) = f #todo: this should probably conjugate the field but need to think about exactly what that impacts....
