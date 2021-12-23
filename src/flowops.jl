
abstract type FlowOp{I,t₀,t₁,T} <: ImplicitOp{T} end
abstract type FlowOpWithAdjoint{I,t₀,t₁,T} <: FlowOp{I,t₀,t₁,T} end

# interface
function velocity end
function velocityᴴ end
function negδvelocityᴴ end



# define integrations for L*f, L'*f, L\f, and L'\f
*(Lϕ::                FlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = @⌛ odesolve(I,  velocity(cache(Lϕ, f),f)..., t₀, t₁)
*(Lϕ::Adjoint{<:Any,<:FlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = @⌛ odesolve(I, velocityᴴ(cache(Lϕ',f),f)..., t₁, t₀)
\(Lϕ::                FlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = @⌛ odesolve(I,  velocity(cache(Lϕ, f),f)..., t₁, t₀)
\(Lϕ::Adjoint{<:Any,<:FlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = @⌛ odesolve(I, velocityᴴ(cache(Lϕ',f),f)..., t₀, t₁)


@adjoint (::Type{L})(ϕ) where {L<:FlowOp} = L(ϕ), Δ -> (Δ,)
@adjoint (Lϕ::FlowOp)(ϕ′) = Lϕ(ϕ′), Δ -> (nothing, Δ)

# for FlowOps (without adjoint), use Zygote to take a gradient through the ODE solver

@adjoint *(Lϕ::FlowOp{I,t₀,t₁}, f::Field{B}) where {I,t₀,t₁,B} = 
    Zygote.pullback((Lϕ,f)->odesolve(I, velocity(cache(Lϕ, f),f)..., t₀, t₁), Lϕ, f)
    
@adjoint \(Lϕ::FlowOp{I,t₀,t₁}, f::Field{B}) where {I,t₀,t₁,B} = 
    Zygote.pullback((Lϕ,f)->odesolve(I, velocity(cache(Lϕ, f),f)..., t₁, t₀), Lϕ, f)


# FlowOpWithAdjoint provide their own velocity for computing the gradient


# note the weird use of task_local_storage below is b/c if we dont
# need the pullback w.r.t. ϕ, we can save time by running the
# transpose flow, rather than the transpose-δ flow. however, Zygote
# has no capibility to know that here in the code, so we use this ugly
# hack which requires the higher level function to have specified :ϕ
# is constant by setting a task_local_storage. this can be made much
# more clean once we switch to Diffractor

@adjoint function *(Lϕ::FlowOpWithAdjoint{I,t₀,t₁}, f::Field{B}) where {I,t₀,t₁,B}
    cLϕ = cache(Lϕ,f)
    f̃ = cLϕ * f
    function back(Δ)
        if :ϕ in get(task_local_storage(), :AD_constants, ())
            nothing, B(cLϕ' * Δ)
        else
            (_,δf,δϕ) = @⌛ odesolve(I, negδvelocityᴴ(cLϕ, FieldTuple(f̃,Δ))..., t₁, t₀)
            δϕ, B(δf)
        end
    end
    f̃, back
end

@adjoint function \(Lϕ::FlowOpWithAdjoint{I,t₀,t₁}, f̃::Field{B}) where {I,t₀,t₁,B}
    cLϕ = cache(Lϕ,f̃)
    f = cLϕ \ f̃
    function back(Δ)
        if :ϕ in get(task_local_storage(), :AD_constants, ())
            nothing, B(cLϕ' \ Δ)
        else
            (_,δf,δϕ) = @⌛ odesolve(I, negδvelocityᴴ(cLϕ, FieldTuple(f,Δ))..., t₀, t₁)
            δϕ, B(δf)
        end
    end
    f, back
end
