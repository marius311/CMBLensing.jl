
# algebra with Fields and LinearFieldOps


# addition and subtraction of two Fields or FieldCovs
for op in (:+, :-)
    @eval ($op)(a::Field, b::Field) = ($op)(promote(a,b)...)
    for F in (Field,FieldCov)
        @eval ($op){T<:($F)}(a::T, b::T) = T(map($op,map(data,(a,b))...)..., meta(a)...)
    end
end

# element-wise multiplication or division of two Fields
for op in (:.*, :./)
    @eval ($op){T<:Field}(a::T, b::T) = T(map($op,map(data,(a,b))...)..., meta(a)...)
end

# ops with a Field or FieldCov and a scalar
for op in (:+, :-, :*, :/), F in (Field,FieldCov)
    @eval ($op){T<:($F)}(f::T, n::Number) = T(map($op,data(f),repeated(n))..., meta(f)...)
    @eval ($op){T<:($F)}(n::Number, f::T) = T(map($op,repeated(n),data(f))..., meta(f)...)
end


# B(f) where B is a basis converts f to that basis (each field )
(::Type{B}){P,S,B}(f::Field{P,S,B}) = f
# (::Type{T}){T<:Basis}(f::Field) = T(f)
function convert{T<:Field,P1,S1,B1}(::Type{T}, f::Field{P1,S1,B1})
    P2,S2,B2 = supertype(T).parameters
    @assert P1==P2 && S1==S2
    B2(f)
end


# convert Fields to right basis before feeding into a LinearFieldOp
*{P,S,B1,B2}(op::LinearFieldOp{P,S,B1}, f::Field{P,S,B2}) = op * B1(f)


# allow composition of LinearFieldOps
immutable LazyBinaryOp{Op} <: LinearFieldOp
    a::Union{LinearFieldOp,Number}
    b::Union{LinearFieldOp,Number}
    # function LazyBinaryOp(a::LinearFieldOp,b::LinearFieldOp)
    #     # @assert meta(a)==meta(b) "Can't '$Op' two operators with different metadata"
    #     new(a,b)
    # end
    # @swappable LazyBinaryOp(op::LinearFieldOp, n::Number) = new(op,n)
end

## construct them with operators
for op in (:+, :-, :*)
    @eval ($op)(a::LinearFieldOp, b::LinearFieldOp) = LazyBinaryOp{$op}(a,b)
    @eval @swappable ($op)(a::LinearFieldOp, b::Number) = LazyBinaryOp{$op}(a,b)
end
/(op::LinearFieldOp, n::Number) = LazyBinaryOp{/}(op,n)

## evaluation rules when applying them
for op in (:+, :-)
    @eval *(lz::LazyBinaryOp{$op}, f::Field) = ($op)(lz.a * f, lz.b * f)
end
*(lz::LazyBinaryOp{/}, f::Field) = (lz.a * f) / lz.b
*(lz::LazyBinaryOp{*}, f::Field) = lz.a * (lz.b * f)

## getting metadata
linop(lz::LazyBinaryOp) = isa(lz.a, LinearFieldOp) ? lz.a : lz.b
meta(lz::LazyBinaryOp) = meta(linop(lz))
size(lz::LazyBinaryOp) = size(linop(lz))
