
# algebra with Fields and LinearOps


# addition and subtraction of two Fields or LinearDiagOps
for op in (:+, :-)
    @eval ($op)(a::Field, b::Field) = ($op)(promote(a,b)...)
    for F in (Field,LinearDiagOp)
        @eval ($op){T<:($F)}(a::T, b::T) = T(map($op,map(data,(a,b))...)..., meta(a)...)
        @eval ($op){T<:($F)}(a::T) = T(map($op,data(a))..., meta(a)...)
    end
end

# element-wise multiplication or division of two Fields
# no promotion should be done here since a.*b isn't linear alegrbra 
# (i.e. it's not independent of which basis its done in)
for op in (:*, :/), F in (Field,LinearDiagOp)
    @eval ($op){T<:($F)}(a::T, b::T) = T(map($(Symbol(:.,op)),map(data,(a,b))...)..., meta(a)...)
end


# ops with a Field or LinearDiagOp and a scalar
for op in (:+, :-, :.+, :.-, :*, :/, :.*, :./), F in (Field,LinearDiagOp)
    @eval ($op){T<:($F)}(f::T, n::Number) = T(map($op,data(f),repeated(n))..., meta(f)...)
    @eval ($op){T<:($F)}(n::Number, f::T) = T(map($op,repeated(n),data(f))..., meta(f)...)
end

# Can raise these guys to powers explicitly since they're diagonal
# for the case of f::Field, this is another instance where we're not really doing linear algebra
^{T<:Union{LinearDiagOp,Field}}(f::T, n::Number) = T(map(.^,data(f),repeated(n))..., meta(f)...)
sqrt{T<:Union{LinearDiagOp,Field}}(f::T) = T(map(sqrt,data(f))..., meta(f)...)

# B(f) where B is a basis converts f to that basis
(::Type{B}){P,S,B}(f::Field{P,S,B}) = f
function convert{T<:Field,P1,S1,B1}(::Type{T}, f::Field{P1,S1,B1})
    if T.abstract
        f::T
    else
        P2,S2,B2 = supertype(T).parameters
        @assert P1==P2 && S1==S2
        B2(f)
    end
end


# convert Fields to right basis before feeding into a LinearOp
*{P,S,B1,B2}(op::LinearOp{P,S,B1}, f::Field{P,S,B2}) = op * B1(f)


# type for allowing composition of LinearOps
immutable LazyBinaryOp{Op} <: LinearOp
    a::Union{LinearOp,Number}
    b::Union{LinearOp,Number}
    # maybe assert metadata is the same here? 
end

# do these ops lazily in these cases
for op in (:+, :-, :*)
    @eval ($op)(a::LinearOp, b::LinearOp) = LazyBinaryOp{$op}(a,b)
    @eval @swappable ($op)(a::LinearOp, b::Number) = LazyBinaryOp{$op}(a,b)
end
/(op::LinearOp, n::Number) = LazyBinaryOp{/}(op,n)
-(op::LinearOp) = LazyBinaryOp{*}(-1,op)
^(op::LinearOp, n::Integer) = n==0 ? 1 : n==1 ? op : *(repeated(op,n)...)

# evaluation rules when finally applying a lazy op to a field
for op in (:+, :-)
    @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
end
*(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
*(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)

# getting metadata
linop(lz::LazyBinaryOp) = isa(lz.a, LinearOp) ? lz.a : lz.b
meta(lz::LazyBinaryOp) = meta(linop(lz))
size(lz::LazyBinaryOp) = size(linop(lz))


import Base: inv
function inv{T<:Field}(m::Matrix{T})
    n,n = size(m)
    @assert n==2
    a,b,c,d = m[:]
    invdet = 1./(a*d-b*c)
    [invdet*d -invdet*b; -invdet*c invdet*a] :: Matrix{T}
end


# linear algebra of Vector{T} and Matrix{T} where T<:Union{Field,LinearOp}
import Base: Ac_mul_B, A_mul_Bc, broadcast, transpose
function *{T1<:Union{Field,LinearOp},T2<:Union{Field,LinearOp}}(a::AbstractVecOrMat{T1}, b::AbstractVecOrMat{T2})
    @assert size(a,2)==size(b,1) "Dimension mismatch"
    ans = [sum(a[i,j]*b[j,k] for j=1:size(b,1)) for i=1:size(a,1), k=1:size(b,2)]
    size(ans)==(1,1) ? ans[1,1] : ans
end
Ac_mul_B{T1<:Union{Field,LinearOp},T2<:Union{Field,LinearOp}}(a::AbstractVecOrMat{T1}, b::AbstractVecOrMat{T2}) = (at=a'; at*b)
A_mul_Bc{T1<:Union{Field,LinearOp},T2<:Union{Field,LinearOp}}(a::AbstractVecOrMat{T1}, b::AbstractVecOrMat{T2}) = (bt=b'; a*bt)
*{T<:LinearOp}(m::AbstractArray{T}, f::Field) = broadcast(*,m,[f])
(::Type{B}){B<:Basis,F<:Field}(a::AbstractArray{F}) = map(B,a)
transpose(f::Union{Field,LinearOp}) = f
