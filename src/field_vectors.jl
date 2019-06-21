
# length-2 StaticVectors or 2x2 StaticMatrices of Fields or LinOps are what we
# consider "Field or Op" arrays, for which we define some special behavior
const FieldOrOpVector{F<:FieldOrOp}    = StaticVector{2,F}
const FieldOrOpRowVector{F<:FieldOrOp} = Adjoint{<:Adjoint{<:Any,F},<:FieldOrOpVector{F}}
const FieldOrOpMatrix{F<:FieldOrOp}    = StaticMatrix{2,2,F}
const FieldOrOpArray{F<:FieldOrOp}     = Union{FieldOrOpVector{F}, FieldOrOpMatrix{F}, FieldOrOpRowVector{F}}
# or just Fields: 
const FieldVector{F<:Field}    = FieldOrOpVector{F}
const FieldRowVector{F<:Field} = FieldOrOpRowVector{F}
const FieldMatrix{F<:Field}    = FieldOrOpMatrix{F}
const FieldArray{F<:Field}     = FieldOrOpArray{F}




### broadcasting
# FieldArray broadcasting wins over Field and FieldTuple broadcasting, so go
# through the broadcast expression, wrap all Field and FieldTuple args in Refs,
# then forward to DefaultArrayStyle broadcasting
struct FieldOrOpArrayStyle{N} <: AbstractArrayStyle{N} end
(::Type{<:FieldOrOpArrayStyle})(::Val{N}) where {N} = FieldOrOpArrayStyle{N}()
BroadcastStyle(::Type{<:FieldOrOpVector}) = FieldOrOpArrayStyle{1}()
BroadcastStyle(::Type{<:FieldOrOpMatrix}) = FieldOrOpArrayStyle{2}()
BroadcastStyle(::FieldOrOpArrayStyle{N}, ::ArrayStyle{F}) where {N,F<:Field} = FieldOrOpArrayStyle{N}()
BroadcastStyle(::FieldOrOpArrayStyle{N}, ::Style{FT}) where {N,FT<:FieldTuple} = FieldOrOpArrayStyle{N}()
instantiate(bc::Broadcasted{<:FieldOrOpArrayStyle}) = bc
function copy(bc::Broadcasted{FieldOrOpArrayStyle{N}}) where {N}
    bc′ = flatten(bc)
    bc″ = Broadcasted{DefaultArrayStyle{N}}(bc′.f, map(fieldvector_data,bc′.args))
    materialize(bc″)
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


# needed since v .* f is not type stable
*(v::FieldOrOpVector, f::Field) = @SVector[v[1]*f, v[2]*f]
*(f::Field, v::FieldOrOpVector) = @SVector[f*v[1], f*v[2]]

# eventually replace having to do this by hand with Cassette-based solution
mul!(f::Field, v::FieldOrOpRowVector, w::FieldOrOpVector) = (@. f = v.parent[1]*w[1] + v.parent[2]*w[2])

# # until StaticArrays better implements adjoints
# *(v::FieldRowVector, M::FieldMatrix) = @SVector[v'[1]*M[1,1] + v'[2]*M[2,1], v'[1]*M[1,2] + v'[2]*M[2,2]]'
# # and until StaticArrays better implements invereses... 
function inv(dst::FieldMatrix, src::FieldMatrix)
    a,b,c,d = src
    det⁻¹ = @. 1/(a*d-b*c)
    @. dst[1,1] =  det⁻¹*d
    @. dst[1,2] = -det⁻¹*b
    @. dst[2,1] = -det⁻¹*c
    @. dst[2,2] =  det⁻¹*a
    dst
end
# mul!(f::Field, ::typeof(∇'), v::FieldVector) = f .= (∇*v[1])[1] .+ (∇*v[2])[2]

# helps StaticArrays infer various results correctly:
promote_rule(::Type{F}, ::Type{<:Scalar}) where {F<:Field} = F
arithmetic_closure(::F) where {F<:Field} = F
# using LinearAlgebra: matprod
# Base.promote_op(::typeof(adjoint), ::Type{T}) where {T<:∇i} = T
# Base.promote_op(::typeof(matprod), ::Type{<:∇i}, ::Type{<:F}) where {F<:Field} = Base._return_type(*, Tuple{∇i{0,true},F})