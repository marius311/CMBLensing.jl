
abstract type FlowOp{I,t₀,t₁} <: ImplicitOp{Basis,Spin,Pix} end
abstract type FlowOpWithAdjoint{I,t₀,t₁} <: FlowOp{I,t₀,t₁} end

# interface
function velocity end
function velocityᴴ end
function negδvelocityᴴ end


# if no custom caching is defined
cache(L::FlowOp, f) = L
cache(L::Adjoint{<:Any,<:FlowOp}, f) = cache(L',f)'

# define integrations for L*f, L'*f, L\f, and L'\f
*(Lϕ::                FlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = odesolve(I,  velocity(cache(Lϕ, f),f)..., t₀, t₁)
*(Lϕ::Adjoint{<:Any,<:FlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = odesolve(I, velocityᴴ(cache(Lϕ',f),f)..., t₁, t₀)
\(Lϕ::                FlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = odesolve(I,  velocity(cache(Lϕ, f),f)..., t₁, t₀)
\(Lϕ::Adjoint{<:Any,<:FlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = odesolve(I, velocityᴴ(cache(Lϕ',f),f)..., t₀, t₁)


@adjoint (::Type{L})(ϕ) where {L<:FlowOp} = L(ϕ), Δ -> (Δ,)
@adjoint (Lϕ::FlowOp)(ϕ′) = Lϕ(ϕ′), Δ -> (nothing, Δ)

@adjoint function *(Lϕ::FlowOpWithAdjoint{I,t₀,t₁}, f::Field{B}) where {I,t₀,t₁,B}
    cLϕ = cache(Lϕ,f)
    f̃ = cLϕ * f
    function back(Δ)
        (_,δf,δϕ) = odesolve(I, negδvelocityᴴ(cLϕ, FieldTuple(f̃,Δ))..., t₀, t₁)
        δϕ, B(δf)
    end
    f̃, back
end

@adjoint function \(Lϕ::FlowOpWithAdjoint{I,t₀,t₁}, f̃::Field{B}) where {I,t₀,t₁,B}
    cLϕ = cache(L,f)
    f = cLϕ \ f̃
    function back(Δ)
        (_,δf,δϕ) = odesolve(I, negδvelocityᴴ(cLϕ), FieldTuple(f,Δ), t₁, t₀)
        δϕ, B(δf)
    end
    f̃, back
end
