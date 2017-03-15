import Base: broadcast, broadcast!


### broadcasting over combinations of Scalars, Fields, and LinDiagOps

# scalars which are allowed in our expressions must be real because we
# implicitly assume our maps are real, and addition/multiplication by a complex
# number, even of the fourier transform, would break this. 
const Scalar = Real

# the data which is broadcast over for Fields and Scalars
# (other objects can define their own methods for this, allowing a single type
# of object to be broadcast with many different types of Fields)
broadcast_data(::Type{F}, n::Scalar) where {F<:Field} = repeated(n)
broadcast_data(::Type{F}, f::F) where {F<:Field} = fieldvalues(f)

# fall-back
broadcast_data(::Type{F}, f::T) where {F,T} = error("Can't broadcast $T as a $F.
Try not using broadcasting or converting $F to the right basis by hand.")


# the actual broadcast functions which broadcast operations down to the
# underlying data as returned by broadcast_data.
broadcast(op, args::Union{F,LinDiagOp,Scalar}...) where {F<:Field} = begin
    F((broadcast(op,d...) for d=zip(map(broadcast_data,repeated(F),args)...))...)
end
broadcast!(op, X::F, args::Union{F,LinDiagOp,Scalar}...) where {F<:Field} = begin
    for (x,d)=zip(broadcast_data(F,X),zip(map(broadcast_data,repeated(F),args)...))
        broadcast!(op,x,d...)
    end
    X
end

# catch-alls to give more helpful error messages
broadcast(op, ::Union{D,Scalar}...) where {D<:LinDiagOp} = error("Broadcast expression must contain at least one Field.")
broadcast(op, ::Union{F,Field,LinDiagOp,Scalar}...) where {F<:Field} = error("Fields in a broadcast expression must all be the same type.")



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
^(f::Field,n::Int) = f.^n #needed to avoid ambiguity
-(f::Field) = .-(f)
dot(a::Field,b::Field) = dot(promote(a,b)...)


# B(f) where B is a basis converts f to that basis
(::Type{B}){P,S,B}(f::Field{P,S,B}) = f
convert(::Type{F}, f::Field{P,S,B1}) where {P,S,B1,B2,F<:Field{P,S,B2}} = B2(f)


# convert Fields to right basis before feeding into a LinOp
for op=(:*,:\)
    @eval ($op){P,S,B1,B2}(O::LinOp{P,S,B1}, f::Field{P,S,B2}) = $(op)(O,B1(f))
    @eval ($op){P,S,B}(O::LinOp{P,S,B}, f::Field{P,S,B}) = throw(MethodError($op,(O,f)))
end





# # type for allowing composition of LinOps
# struct LazyBinaryOp{Op} <: LinOp{Pix,Spin,Basis}
#     a::Union{LinOp,Real}
#     b::Union{LinOp,Real}
#     # maybe assert metadata is the same here? 
# end
# 
# # do these ops lazily in these cases
# for op in (:+, :-, :*)
#     @eval ($op)(a::LinOp, b::LinOp) = LazyBinaryOp{$op}(a,b)
#     @eval @swappable ($op)(a::LinOp, b::Real) = LazyBinaryOp{$op}(a,b)
# end
# # /(op::LinOp, n::Real) = LazyBinaryOp{/}(op,n)
# # -(op::LinOp) = LazyBinaryOp{*}(-1,op)
# ^(op::LinOp, n::Real) = n<0 ? error("Can't raise $T to negative ($n) power") : n==0 ? 1 : n==1 ? op : *(repeated(op,n)...)

# 
# # evaluation rules when finally applying a lazy op to a field
# for op in (:+, :-)
#     @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
# end
# *(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
# *(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)
# 
# # getting metadata
# linop(lz::LazyBinaryOp) = isa(lz.a, LinOp) ? lz.a : lz.b
# meta(lz::LazyBinaryOp) = meta(linop(lz))
# size(lz::LazyBinaryOp) = size(linop(lz))


# import Base: inv
# function inv{T<:Field}(m::Matrix{T})
#     n,n = size(m)
#     @assert n==2
#     a,b,c,d = (m[i] for i=eachindex(m))
#     invdet = 1./(a*d-b*c)
#     [invdet*d -invdet*b; -invdet*c invdet*a] :: Matrix{T}
# end


# linear algebra of Vector{T} and Matrix{T} where T<:Union{Field,LinOp}
import Base: Ac_mul_B, A_mul_Bc, broadcast, transpose
# function *{T1<:Union{Field,LinOp},T2<:Union{Field,LinOp}}(a::AbstractVecOrMat{T1}, b::AbstractVecOrMat{T2})
#     @assert size(a,2)==size(b,1) "Dimension mismatch"
#     ans = [sum(a[i,j]*b[j,k] for j=1:size(b,1)) for i=1:size(a,1), k=1:size(b,2)]
#     size(ans)==(1,1) ? ans[1,1] : ans
# end
# Ac_mul_B{T1<:Union{Field,LinOp},T2<:Union{Field,LinOp}}(a::AbstractVecOrMat{T1}, b::AbstractVecOrMat{T2}) = (at=a'; at*b)
# Ac_mul_B{F<:Field}(f::Field, m::AbstractArray{F}) = broadcast(Ac_mul_B,[f],m)

A_mul_Bc(a::Vector{<:Field}, b::Vector{<:Field}) = [a[i].*b[j] for i=eachindex(a), j=eachindex(b)]
A_mul_Bc(a::Vector{<:LinOp}, b::Vector{<:Field}) = [a[i].*b[j] for i=eachindex(a), j=eachindex(b)]
*{T<:LinOp}(m::AbstractArray{T}, f::Field) = broadcast(*,m,[f])
# *{F<:Field}(f::Field, m::AbstractArray{F}) = broadcast(*,[f],m)
# *{F<:Field}(f::LinOp, m::AbstractArray{F}) = broadcast(*,[f],m)
# (::Type{B}){B<:Basis,F<:Field}(a::AbstractArray{F}) = map(B,a)
transpose(f::Union{Field,LinOp}) = f #todo: this should probably conjugate the field but need to think about exactly what that impacts....
