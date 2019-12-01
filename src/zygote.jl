@eval using Zygote: unbroadcast, Numeric, @adjoint

@eval begin
    
# lenseflow

@adjoint function *(Lϕ::LenseFlowOp, f::Field{B}) where {B}
    f̃ = Lϕ * f
    function back(Δ)
        δf, δϕ = δf̃ϕ_δfϕ(Lϕ,f̃,f)' * FΦTuple(Δ,zero(getϕ(Lϕ)))
        LenseFlow(δϕ), B(δf)
    end
    f̃, back
end

@adjoint function \(Lϕ::LenseFlowOp, f̃::Field{B}) where {B}
    f = Lϕ \ f̃
    function back(Δ)
        δf, δϕ = δfϕ_δf̃ϕ(Lϕ,f,f̃)' * FΦTuple(Δ,zero(getϕ(Lϕ)))
        LenseFlow(δϕ), B(δf)
    end
    f, back
end

@adjoint (::Type{L})(ϕ) where {L<:LenseFlowOp} = L(ϕ), Δ -> (Δ.ϕ,)
@adjoint cache(Lϕ::LenseFlowOp, f) = cache(Lϕ,f), Δ->(Δ,nothing)
@adjoint cache!(cL::CachedLenseFlow, ϕ) where {L<:LenseFlowOp} = cache!(cL,ϕ), Δ -> (nothing,Δ.ϕ)


# algebra

@adjoint (::Type{B})(f::Field{B′}) where {B<:Basis, B′} = B(f), Δ -> (B′(Δ),)
@adjoint (*)(L::LinOp, f::Field{B}) where {B} = L*f, Δ -> (nothing, B(L'*Δ))
@adjoint (\)(L::LinOp, f::Field{B}) where {B} = L\f, Δ -> (nothing, B(L'\Δ))
@adjoint (*)(a::Adjoint{<:Any,<:Field{B1}}, b::Field{B2}) where {B1,B2} = a*b, Δ -> (B1(Δ*b)',  B2(Δ*a'))
@adjoint function (*)(x::Adjoint{<:Any,<:Field{B1}}, D::Union{Diagonal,OuterProdOp}, y::Field{B2}) where {B1,B2}
    z = x*D*y
    back = if parent(x)===y
        function (Δ)
            g = B1(Δ*(D*y))
            (g', Δ*z, g)
        end
    else
        function (Δ)
            (B1(Δ*(D*y))', Δ*z, B2(Δ*(D*x')))
        end
    end
    z, back
end

# eventually we need to implement these to allow gradients w.r.t. θ:
@adjoint pinv(D::LinOp) = pinv(D), Δ->nothing
@adjoint logdet(L::LinOp, θ) = logdet(L,θ), Δ->nothing
@adjoint (ds::DataSet)(args...; kwargs...) = ds(args...; kwargs...), Δ -> nothing

# some stuff which arguably belongs in Zygote or ChainRules
# see also: https://github.com/FluxML/Zygote.jl/issues/316
@adjoint *(D::Diagonal, v::AbstractVector) = D.diag .* v,
    Δ -> (Diagonal(unbroadcast(D.diag, Δ .* conj.(v))), unbroadcast(v, Δ .* conj.(D.diag)))

@adjoint broadcasted(::typeof(-), x ::Numeric, y::Numeric) =
    broadcast(-, x, y), Δ -> (nothing, unbroadcast(x, Δ), unbroadcast(y, -Δ))

@adjoint broadcasted(::typeof(\), x ::Numeric, y::Numeric) =
    broadcast(\, x, y), Δ -> (nothing, unbroadcast(x, @. -Δ*y/x^2), unbroadcast(y, @. Δ/x))


end
