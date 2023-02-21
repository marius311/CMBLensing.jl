
# imlements some behavior for length-2 StaticVectors and 2x2
# StaticMatrices of Fields or FieldOps. these come up when you do
# gradients, e.g. ∇*ϕ is a vector of Fields. 

# where elements are Fields or FieldOps
const FieldOrOpVector{F<:FieldOrOp}    = StaticVector{2,F}
const FieldOrOpMatrix{F<:FieldOrOp}    = StaticMatrix{2,2,F}
const FieldOrOpRowVector{F<:FieldOrOp} = Adjoint{<:Any,<:FieldOrOpVector{F}}
const FieldOrOpArray{F<:FieldOrOp}     = Union{FieldOrOpVector{F}, FieldOrOpMatrix{F}, FieldOrOpRowVector{F}}
# where elements are just Fields
const FieldVector{F<:Field}    = SVector{2,F}
const FieldMatrix{F<:Field}    = SMatrix{2,2,F,4}
const FieldRowVector{F<:Field} = FieldOrOpRowVector{F}
const FieldArray{F<:Field}     = FieldOrOpArray{F}


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
# only thing needed for SpinAdjoints
mul!(v::FieldVector, f::SpinAdjoint, w::FieldVector) = (mul!(v[1], f, w[1]); mul!(v[2], f, w[2]); v)

promote_rule(::Type{F}, ::Type{<:Scalar}) where {F<:Field} = F


# StaticArrays really sucks when !isbitstype(eltype(A)), so we need to
# write a bunch of this stuff by hand to make it work :(
# note: `a,c,b,d = A` is the right unpack order for A = [a b; c d]

@auto_adjoint function *(A::StaticMatrix{2,2,<:DiagOp}, x::StaticVector{2,<:Union{Field,DiagOp}})
    SizedVector{2}([A[1,1]*x[1]+A[1,2]*x[2], A[2,1]*x[1]+A[2,2]*x[2]])
end

@auto_adjoint function sqrt(A::SA) where {SA<:StaticMatrix{2,2,<:DiagOp}}
    a,c,b,d = A
    s = sqrt(a*d-b*c)
    t = pinv(sqrt(a+(d+2s)))
    SA([t*(a+s) t*b; t*c t*(d+s)])
end

@auto_adjoint function det(A::StaticMatrix{2,2,<:DiagOp})
    a,c,b,d = A
    a*d-b*c
end

@auto_adjoint function pinv(A::SA) where {SA<:StaticMatrix{2,2,<:DiagOp}}
    a,c,b,d = A
    idet = pinv(a*d-b*c)
    SA([d*idet -(b*idet); -(c*idet) a*idet])
end

function pinv!(dst::StaticMatrix{2,2,<:DiagOp}, src::StaticMatrix{2,2,<:DiagOp})
    a,c,b,d = src
    det⁻¹ = pinv(@. a*d-b*c)
    @. dst[1,1] =  det⁻¹ * d
    @. dst[1,2] = -det⁻¹ * b
    @. dst[2,1] = -det⁻¹ * c
    @. dst[2,2] =  det⁻¹ * a
    dst
end




@adjoint (::Type{SA})(x::AbstractArray) where {SA<:SizedArray} = SA(x), Δ -> (Δ.data,)

# needed for autodiff to avoid calling setindex! on StaticArrays with !isbitstype(eltype(x))
function ChainRules.∇getindex(x::StaticArray, dy, inds...)
    T = Union{typeof(dy), ChainRules.ZeroTangent}
    plain_inds = CartesianIndex(Base.to_indices(x, inds))
    StaticArrays.sacollect(StaticArrays.similar_type(x, T, axes(x)), (I==plain_inds ? dy : ChainRules.ZeroTangent() for I in CartesianIndices(x)))
end
if isdefined(Zygote, :∇getindex) # to be deleted by https://github.com/FluxML/Zygote.jl/pull/1328
    function Zygote.∇getindex(x::StaticArray, inds)
        getindex_pullback(dy) = (ChainRules.∇getindex(x, dy, inds...), map(_->nothing, inds)...)
        getindex_pullback
    end
end
