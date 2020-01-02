    
# lazy outer products of Fields, which comes up alot in automatic differentiation
*(x::Field, y::Adjoint{<:Any, <:Field}) = OuterProdOp(x,y.parent)

@adjoint sum(f::Field{B}) where {B} = sum(f), Δ -> (Δ*one(B(f)),)
@adjoint norm(f::Field) = Zygote.pullback(f->sqrt(dot(f,f)), f)

# it doesnt look like we _need_ these, but without them, the default pullbacks
# don't put things in a basis consistent with the input arguments (auto-basis
# conversions seem to take care of things afterwards though)
@adjoint dot(f::Field{B1}, g::Field{B2}) where {B1,B2} = dot(f,g), Δ -> (Δ*B1(g), Δ*B2(f))
@adjoint *(f::Adjoint{<:Any,<:Field}, g::Field) = Zygote.pullback((f,g)->dot(f',g),f,g)


@adjoint (::Type{B})(f::Field{B′}) where {B<:Basis, B′} = B(f), Δ -> (B′(Δ),)

# this makes it so we only have to define adjoints for L*f, and the f'*L adjoint just uses that
@adjoint *(f::Adjoint{<:Any,<:Field}, D::Diagonal)   = Zygote.pullback((f,D)->(D'*f')', f, D)

@adjoint function *(x::FieldOrOpRowVector, y::FieldVector)
    
    z = x * y
    
    # when x is a vector of Fields
    back(Δ) = (Δ * y', x' * Δ)
    
    # when x is a vector of Diagonals. in this case, Δ * basis(Δ)(y)' create an
    # OuterProdOp in the same basis as the Diagonals in x
    back(Δ::Field{B}) where {B} = (Δ * basis(Δ)(y)'), (x' * Δ)
    
    z, back
    
end


@adjoint function *(A::FieldOrOpMatrix, x::FieldOrOpVector)

    z = A * x

    back(Δ::FieldVector) = (B=basis(eltype(z)); (B(Δ) * B(x)', A' * Δ))
    back(Δ::FieldOrOpVector) = (Δ * x', A' * Δ) # not sure if anything special needed here?
    back(Δ) = (Δ * x', A' * Δ)

    z, back
    
end

# see Zygote/lib/array.jl:311
@adjoint function pinv(M::FieldOrOpMatrix) 
    M⁻¹ = pinv(M)
    M⁻¹, Δ->(-M⁻¹*Δ*M⁻¹,)
    # M⁻¹, Δ->(-M⁻¹' * Δ * M⁻¹' + (- M * M⁻¹ * Δ' * M⁻¹ * M⁻¹' + Δ' * M⁻¹ * M⁻¹') + (M⁻¹' * M⁻¹ * Δ' - M⁻¹' * M⁻¹ * Δ' * M⁻¹ * M),)
end

# without this we get a segfault for I + Hessian(ϕ) like in the LenseFlow velocity
@adjoint +(I::UniformScaling, M::FieldOrOpMatrix) = I+M, Δ->(nothing, Δ)


### apparently we don't need either of these anymore?: 
# this is a specialized version of the suggestion here:
#   https://github.com/FluxML/Zygote.jl/issues/316
# which also does necessary basis conversions
# @adjoint *(D::DiagOp{<:Field{B}}, v::Field{B′}) where {B,B′} = D.diag .* B(v),
#     Δ -> (Diagonal(unbroadcast(D.diag, B(B′(Δ) .* conj.(v)))), unbroadcast(v, B′(B(Δ) .* conj.(D.diag))))
# @adjoint *(D::DiagOp{<:Field{B}}, v::Field{B′}) where {B,B′} = Zygote.pullback((D,v) -> diag(D) .* B(v), D, v)
### 

@adjoint *(∇::DiagOp{<:∇diag}, f::Field{B}) where {B} = ∇*f, Δ->(nothing, B(∇'*Δ))

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
