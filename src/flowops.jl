
abstract type FlowOp{T} <: ImplicitOp{T} end
abstract type FlowOpWithAdjoint{T} <: FlowOp{T} end

# interface
function velocity end
function velocityᴴ end
function negδvelocityᴴ end

# define integrations for L*f, L'*f, L\f, and L'\f
*(Lϕ::                FlowOp,  f::Field) = @⌛ Lϕ.odesolve( velocity(precompute!!(Lϕ,  f), f)..., Lϕ.t₀ => Lϕ.t₁)
*(Lϕ::Adjoint{<:Any,<:FlowOp}, f::Field) = @⌛ Lϕ.odesolve(velocityᴴ(precompute!!(Lϕ', f), f)..., Lϕ.t₁ => Lϕ.t₀)
\(Lϕ::                FlowOp,  f::Field) = @⌛ Lϕ.odesolve( velocity(precompute!!(Lϕ,  f), f)..., Lϕ.t₁ => Lϕ.t₀)
\(Lϕ::Adjoint{<:Any,<:FlowOp}, f::Field) = @⌛ Lϕ.odesolve(velocityᴴ(precompute!!(Lϕ', f), f)..., Lϕ.t₀ => Lϕ.t₁)


@adjoint (::Type{L})(ϕ) where {L<:FlowOp} = L(ϕ), Δ -> (Δ,)
@adjoint (Lϕ::FlowOp)(ϕ′) = Lϕ(ϕ′), Δ -> (nothing, Δ)

# for FlowOps (without adjoint), use Zygote to take a gradient through the ODE solver

@adjoint *(Lϕ::FlowOp, f::Field{B}) where {B} = 
    Zygote.pullback((Lϕ, f) -> Lϕ.odesolve(velocity(precompute!!(Lϕ, f), f)..., Lϕ.t₀ => Lϕ.t₁), Lϕ, f)
    
@adjoint \(Lϕ::FlowOp, f::Field{B}) where {B} = 
    Zygote.pullback((Lϕ, f) -> Lϕ.odesolve(velocity(precompute!!(Lϕ, f), f)..., Lϕ.t₁ => Lϕ.t₀), Lϕ, f)


# FlowOpWithAdjoint provide their own velocity for computing the gradient


# note the weird use of task_local_storage below is b/c if we dont
# need the pullback w.r.t. ϕ, we can save time by running the
# transpose flow, rather than the transpose-δ flow. however, Zygote
# has no capibility to know that here in the code, so we use this ugly
# hack which requires the higher level function to have specified :ϕ
# is constant by setting a task_local_storage. this can be made much
# more clean once we switch to Diffractor

@adjoint function *(Lϕ::FlowOpWithAdjoint, f::Field{B}) where {B}
    Lϕ_fwd = precompute!!(Lϕ, f)
    f̃ = Lϕ_fwd * f
    function back(Δ)
        Lϕ_back = precompute!!(Lϕ_fwd, Δ)
        if :ϕ in get(task_local_storage(), :AD_constants, ())
            nothing, B(Lϕ_back' * Δ)
        else
            (_, δf, δϕ) = @⌛ Lϕ.odesolve(negδvelocityᴴ(Lϕ_back, FieldTuple(f̃, Δ))..., Lϕ.t₁ => Lϕ.t₀)
            δϕ, B(δf)
        end
    end
    f̃, back
end

@adjoint function \(Lϕ::FlowOpWithAdjoint, f̃::Field{B}) where {B}
    Lϕ_fwd = precompute!!(Lϕ, f)
    f = Lϕ_fwd \ f̃
    function back(Δ)
        Lϕ_back = precompute!!(Lϕ_fwd, Δ)
        if :ϕ in get(task_local_storage(), :AD_constants, ())
            nothing, B(Lϕ_back' \ Δ)
        else
            (_, δf, δϕ) = @⌛ Lϕ.odesolve(negδvelocityᴴ(Lϕ_back, FieldTuple(f, Δ))..., Lϕ.t₀ => Lϕ.t₁)
            δϕ, B(δf)
        end
    end
    f, back
end
