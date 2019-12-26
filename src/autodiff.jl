    
### algebra

# lazy outer products of Fields, which comes up alot in automatic differentiation
*(x::Field, y::Adjoint{<:Any, <:Field}) = FuncOp(op=z->x*(y*z), opᴴ=z->y'*(x'*z))

@adjoint sum(f::Field) = sum(f), Δ -> (Δ*one(f),)

@adjoint (::Type{B})(f::Field{B′}) where {B<:Basis, B′} = B(f), Δ -> (B′(Δ),)

#---
# todo: try and figure out a way to forward adjoints like these that doesn't
# involve duplicating code in Base?  see also:
# https://discourse.julialang.org/t/how-to-deal-with-zygote-sometimes-pirating-its-own-adjoints-with-worse-ones

# this makes it so we only have to define adjoints for L*f, and the f'*L adjoint just uses that
@adjoint *(f::Adjoint{<:Any,<:Field}, D::Diagonal)   = Zygote.pullback((f,D)->(D'*f')', f, D)
# this adjoint through StaticArray's complicated machinery doesn't work
@adjoint *(a::FieldOrOpRowVector, b::FieldVector) = Zygote.pullback((a,b)->(a[1]*b[1] + a[2]*b[2]), a.parent, b)
#---


# this is a specialized version of the suggestion here:
#   https://github.com/FluxML/Zygote.jl/issues/316
# which does necessary basis conversions
@adjoint *(D::DiagOp{<:Field{B}}, v::Field{B′}) where {B,B′} = D.diag .* B(v),
    Δ -> (Diagonal(unbroadcast(D.diag, B(B′(Δ) .* conj.(v)))), unbroadcast(v, B′(B(Δ) .* conj.(D.diag))))

@adjoint *(∇::∇Op, f::Field{B}) where {B} = ∇*f, Δ->(nothing, B(∇'*Δ))

# this does basis promotion, unlike Zygote's default for AbstractArrays
Zygote.accum(a::Field, b::Field) = a+b


# eventually we need to implement these to allow gradients w.r.t. θ:
@adjoint pinv(D::LinOp) = pinv(D), Δ->nothing
@adjoint logdet(L::LinOp, θ) = logdet(L,θ), Δ->nothing
@adjoint (ds::DataSet)(args...; kwargs...) = ds(args...; kwargs...), Δ->nothing

# some stuff which arguably belongs in Zygote or ChainRules
# see also: https://github.com/FluxML/Zygote.jl/issues/316

@adjoint broadcasted(::typeof(-), x ::Numeric, y::Numeric) =
    broadcast(-, x, y), Δ -> (nothing, unbroadcast(x, Δ), unbroadcast(y, -Δ))

@adjoint broadcasted(::typeof(\), x ::Numeric, y::Numeric) =
    broadcast(\, x, y), Δ -> (nothing, unbroadcast(x, @. -Δ*y/x^2), unbroadcast(y, @. Δ/x))

@adjoint (::Type{SA})(tup) where {SA<:SArray} = SA(tup), Δ->(tuple(Δ...),)
