

# Eventually we want proper Zygote.jl reverse-mode automatic differentiation
# through our posterior. This is close but not fully working yet, so for now we
# define the following simple API and implement it using finite differences. In
# the future, we can then transparently switch to a Zygote-based solution,
# once that is adequately working.


function gradient(f::Function; relϵ=nothing, absϵ=nothing) 
    
    _gradient(x::Number) = gradient_1vecarg(x->f(x[1]))([x])[1]
    _gradient(x::Vector) = gradient_1vecarg(f)(x)
    
    function gradient_1vecarg(f::Function)
        function (x::Vector)
            g = similar(x)
            if relϵ==nothing && absϵ==nothing
                ϵ = sqrt(eps(eltype(x)))
            elseif absϵ!=nothing
                @assert relϵ==nothing "Only one of `relϵ` or `absϵ` must be specifed"
                ϵ = absϵ
            end
            for i in eachindex(x)
                if relϵ!=nothing
                    ϵ = relϵ*x[i]
                end
                x₊,x₋ = copy(x), copy(x)
                x₊[i] += ϵ
                x₋[i] -= ϵ
                g[i] = (f(x₊) - f(x₋))/(2ϵ)
            end
            return g
        end
    end
    
    (args...) -> _gradient(args...)
    
end

function hessian(f::Function; relϵ=nothing, absϵ=nothing) 
    
    _hessian(x::Number) = gradient_1vecarg(x->f(x[1]))([x])[1]
    _hessian(x::Vector) = gradient_1vecarg(f)(x)
    
    function gradient_1vecarg(f::Function)
        function (x::Vector)
            length(x)==1 || throw(ArgumentError("multi-dimensional hessian not implement yet"))
            g = similar(x)
            if relϵ==nothing && absϵ==nothing
                ϵ = sqrt(sqrt(eps(eltype(x))))
            elseif absϵ!=nothing
                @assert relϵ==nothing "Only one of `relϵ` or `absϵ` must be specifed"
                ϵ = absϵ
            end
            for i in eachindex(x)
                if relϵ!=nothing
                    ϵ = relϵ*x[i]
                end
                x₊,x₋ = copy(x), copy(x)
                x₊[i] += ϵ
                x₋[i] -= ϵ
                g[i] = (f(x₊) - 2f(x) + f(x₋))/(ϵ^2)
            end
            return g
        end
    end
    
    (args...) -> _hessian(args...)

    
end



## Some adjoint rules for the work-in-progress Zygote-based derivatives

@init @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" @eval begin

using .Zygote: unbroadcast, Numeric, @adjoint


@adjoint function *(Lϕ::LenseFlowOp, f::Field{B}) where {B}
    f̃ = Lϕ * f
    function pullback(Δ)
        δf, δϕ = δf̃ϕ_δfϕ(Lϕ,f̃,f)' * FΦTuple(Δ,zero(ϕ))
        LenseFlow(δϕ), B(δf)
    end
    f̃, pullback
end

@adjoint function \(Lϕ::LenseFlowOp, f̃::Field{B}) where {B}
    f = Lϕ \ f̃
    function pullback(Δ)
        δf, δϕ = δfϕ_δf̃ϕ(Lϕ,f,f̃)' * FΦTuple(Δ,zero(ϕ))
        LenseFlow(δϕ), B(δf)
    end
    f, pullback
end

@adjoint (::Type{L})(ϕ) where {L<:LenseFlowOp} = L(ϕ), Δ -> (Δ.ϕ,)

@adjoint (::Type{B})(f::Field{B′}) where {B<:Basis, B′} = B(f), Δ -> (B′(Δ),)

@adjoint (*)(L::LinOp, f::Field{B}) where {B} = L*f, Δ -> (nothing, B(L'*Δ))

@adjoint (\)(L::LinOp, f::Field{B}) where {B} = L*f, Δ -> (nothing, B(L'\Δ))

@adjoint function (*)(x::Adjoint{<:Any,<:Field{B1}}, D::Diagonal, y::Field{B2}) where {B1,B2}
    z = x*D*y
    pullback = if parent(x)===y
        function (Δ)
            g = B1(Δ*(D*y))
            (g', Δ*z, g)
        end
    else
        function (Δ)
            (B1(Δ*(D*y))', Δ*z, B2(Δ*(D*x')))
        end
    end
    z, pullback
end

@adjoint (*)(a::Adjoint{<:Any,<:Field{B1}}, b::Field{B2}) where {B1,B2} = a*b, Δ -> (B1(Δ*b)',  B2(Δ*a'))


# some stuff which arguably belong in Zygote or ChainRules
# see also: https://github.com/FluxML/Zygote.jl/issues/316
@adjoint *(D::Diagonal, v::AbstractVector) = D.diag .* v,
    Δ -> (Diagonal(unbroadcast(D.diag, Δ .* conj.(v))), unbroadcast(v, Δ .* conj.(D.diag)))

@adjoint broadcasted(::typeof(-), x ::Numeric, y::Numeric) =
    broadcast(-, x, y), Δ -> (nothing, unbroadcast(x, Δ), unbroadcast(y, -Δ))

@adjoint broadcasted(::typeof(\), x ::Numeric, y::Numeric) =
    broadcast(\, x, y), Δ -> (nothing, unbroadcast(x, @. -Δ*y/x^2), unbroadcast(y, @. Δ/x))


end
