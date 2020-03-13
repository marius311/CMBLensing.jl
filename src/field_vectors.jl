
# length-2 StaticVectors or 2x2 StaticMatrices of Fields or LinOps are what we
# consider "Field or Op" arrays, for which we define some special behavior
# note: these are non-concrete types to accomodate ∇ which is a custom
# StaticVector rather than an SVector (this doesn't hurt performance anywhere)
const FieldOrOpVector{F<:FieldOrOp}    = StaticVector{2,F}
const FieldOrOpMatrix{F<:FieldOrOp}    = StaticMatrix{2,2,F}
const FieldOrOpRowVector{F<:FieldOrOp} = Adjoint{<:Any,<:FieldOrOpVector{F}}
const FieldOrOpArray{F<:FieldOrOp}     = Union{FieldOrOpVector{F}, FieldOrOpMatrix{F}, FieldOrOpRowVector{F}}
# or just Fields (in this case, they are concrete)
const FieldVector{F<:Field}    = SVector{2,F}
const FieldMatrix{F<:Field}    = SMatrix{2,2,F,4}
const FieldRowVector{F<:Field} = FieldOrOpRowVector{F}
const FieldArray{F<:Field}     = FieldOrOpArray{F}




### broadcasting
# FieldArray broadcasting wins over Field and FieldTuple broadcasting, so go
# through the broadcast expression, wrap all Field and FieldTuple args in Refs,
# then forward to StaticArrayStyle broadcasting
struct FieldOrOpArrayStyle{N} <: AbstractArrayStyle{N} end
(::Type{<:FieldOrOpArrayStyle})(::Val{N}) where {N} = FieldOrOpArrayStyle{N}()
BroadcastStyle(::Type{<:FieldOrOpVector}) = FieldOrOpArrayStyle{1}()
BroadcastStyle(::Type{<:FieldOrOpMatrix}) = FieldOrOpArrayStyle{2}()
BroadcastStyle(S::FieldOrOpArrayStyle, ::FieldTupleStyle) = S
instantiate(bc::Broadcasted{<:FieldOrOpArrayStyle}) = bc
function copy(bc::Broadcasted{FieldOrOpArrayStyle{N}}) where {N}
    bc′ = convert(Broadcasted{StaticArrayStyle{N}}, map_bc_args(fieldvector_data,bc))
    materialize(bc′)
end
fieldvector_data(f::FieldOrOp) = Ref(f)
fieldvector_data(x) = x


# non-broadcasted algebra
for f in (:/, :\, :*)
    if f != :/
        @eval ($f)(A::Field, B::FieldOrOpArray) = broadcast($f, A, B)
    end
    if f != :\
        @eval ($f)(A::FieldOrOpArray, B::Field) = broadcast($f, A, B)
    end
end





# this makes Vector{Diagonal}' * Vector{Field} work right
dot(D::DiagOp{<:Field{B}}, f::Field) where {B} = conj(D.diag) .* B(f)

# needed since v .* f is not type stable
*(v::FieldOrOpVector, f::Field) = @SVector[v[1]*f, v[2]*f]
*(v::FieldOrOpVector, w::FieldOrOpRowVector) = @SMatrix[v[1]*w[1] v[1]*w[2]; v[2]*w[1] v[2]*w[2]]
*(f::Field, v::FieldOrOpVector) = @SVector[f*v[1], f*v[2]]
*(f::Field, v::FieldOrOpRowVector) = @SVector[(f*v[1])', (f*v[2])']'
*(v::FieldOrOpRowVector, w::FieldOrOpVector) = v[1]*w[1] + v[2]*w[2]

# ffs how is something this simple broken in StaticArrays...
adjoint(L::FieldOrOpMatrix) = @SMatrix[L[1,1]' L[2,1]'; L[1,2]' L[2,2]']

# eventually replace having to do this by hand with Cassette-based solution
mul!(f::Field, v::FieldOrOpRowVector{<:Diagonal}, w::FieldVector) = 
    ((@. f = v[1].diag * w[1] + v[2].diag * w[2]); f)
mul!(f::Field, v::FieldOrOpRowVector{<:Diagonal}, x::Diagonal, w::FieldVector) = 
    ((@. f = x.diag * (v[1].diag * w[1] + v[2].diag * w[2])); f)
mul!(v::FieldOrOpVector{<:Diagonal}, M::FieldOrOpMatrix{<:Diagonal}, w::FieldOrOpVector{<:Diagonal}) = 
    ((@. v[1].diag = M[1,1].diag*w[1].diag + M[1,2].diag*w[2].diag); (@. v[2].diag = M[2,1].diag*w[1].diag + M[2,2].diag*w[2].diag); v)
mul!(v::FieldVector, M::FieldOrOpMatrix{<:Diagonal}, w::FieldVector) = 
    ((@. v[1] = M[1,1].diag*w[1] + M[1,2].diag*w[2]); (@. v[2] = M[2,1].diag*w[1] + M[2,2].diag*w[2]); v)
mul!(v::FieldVector, w::FieldOrOpVector{<:Diagonal}, f::Field) = 
    ((@. v[1] = w[1].diag * f); (@. v[2] = w[2].diag * f); v)
mul!(v::FieldVector, x::Diagonal, w::FieldOrOpVector{<:Diagonal}, f::Field) = 
    ((@. v[1] = x.diag * w[1].diag * f); (@. v[2] = x.diag * w[2].diag * f); v)
# only thing needed for TupleAdjoints
mul!(v::FieldVector, f::TupleAdjoint, w::FieldVector) = (mul!(v[1], f, w[1]); mul!(v[2], f, w[2]); v)

function pinv!(dst::FieldOrOpMatrix, src::FieldOrOpMatrix)
    a,b,c,d = src
    det⁻¹ = pinv(@. a*d-b*c)
    @. dst[1,1] =  det⁻¹ * d
    @. dst[1,2] = -det⁻¹ * b
    @. dst[2,1] = -det⁻¹ * c
    @. dst[2,2] =  det⁻¹ * a
    dst
end



promote_rule(::Type{F}, ::Type{<:Scalar}) where {F<:Field} = F
arithmetic_closure(::F) where {F<:Field} = F
