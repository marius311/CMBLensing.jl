
export LenseFlow

abstract type ODESolver end

struct LenseFlow{I<:ODESolver,t₀,t₁,F<:Field} <: LenseOp
    ϕ::F
    ∇ϕ::SVector{2,F}
    Hϕ::SMatrix{2,2,F,4}
end

LenseFlow{I}(ϕ::Field{<:Any,<:S0}) where {I} = LenseFlow{I,0.,1.}(Map(ϕ), gradhess(ϕ)...)
LenseFlow{I,t₀,t₁}(ϕ::F,∇ϕ,Hϕ) where {I,t₀,t₁,F} = LenseFlow{I,t₀,t₁,F}(ϕ,∇ϕ,Hϕ)
LenseFlow(args...) = LenseFlow{jrk4{7}}(args...)

# the ODE solvers
abstract type ode45{reltol,abstol,maxsteps,debug} <: ODESolver  end
abstract type ode4{nsteps} <: ODESolver  end
abstract type jrk4{nsteps} <: ODESolver  end

function ode45{ϵr,ϵa,N,dbg}(vel,x₀,ts) where {ϵr,ϵa,N,dbg}
    ys = ODE.ode45(
        (t,y)->F!(similar(y₀),t,y), y₀, linspace(t₀,t₁,N+1),
        norm=pixstd, reltol=ϵr, abstol=ϵa, minstep=1/N, points=((dbg[1] || dbg[2]) ? :all : :specified)
    )
    dbg && info("ode45 took $(length(ys[2])) steps")
    dbg ? ys : ys[2][end]
end
ode4{N}(F!,y₀,t₀,t₁) where {N} = ODE.ode4((t,y)->F!(similar(y₀),t,y),y₀,linspace(t₀,t₁,N+1))[2][end]
jrk4{N}(F!,y₀,t₀,t₁) where {N} = jrk4(F!,y₀,t₀,t₁,N)

""" ODE velocity for LenseFlow """
velocity!(v::Field, L::LenseFlow, f::Field, t::Real) = (v .= @⨳ L.∇ϕ' ⨳ inv(𝕀 + t*L.Hϕ) ⨳ $Ł(∇*Ð(f)))
velocityᴴ!(v::Field, L::LenseFlow, f::Field, t::Real) = (v .= Ł(@⨳ ∇ᵀ ⨳ $Ð(@⨳ $Ł(f) * (inv(𝕀 + t*L.Hϕ) ⨳ L.∇ϕ))))

@∷ _getindex(L::LenseFlow{I,∷,∷,F}, ::→{t₀,t₁}) where {I,t₀,t₁,F} = LenseFlow{I,t₀,t₁,F}(L.ϕ,L.∇ϕ,L.Hϕ)
*(L::LenseFlow{I,t₀,t₁}, f::Field) where {I,t₀,t₁} = I((v,t,f)->velocity!(v,L,f,t), Ł(f), t₀, t₁)
\(L::LenseFlow{I,t₀,t₁}, f::Field) where {I,t₀,t₁} = I((v,t,f)->velocity!(v,L,f,t), Ł(f), t₁, t₀)
*(f::Field, L::LenseFlow{I,t₀,t₁}) where {I,t₀,t₁} = I((v,t,f)->velocityᴴ!(v,L,f,t), Ł(f), t₀, t₁)
\(f::Field, L::LenseFlow{I,t₀,t₁}) where {I,t₀,t₁} = I((v,t,f)->velocityᴴ!(v,L,f,t), Ł(f), t₁, t₀)


## LenseFlow Jacobian operators

*(J::δfϕₛ_δfϕₜ{s,t,<:LenseFlow}, fϕ::FΦTuple) where {s,t} = δfϕₛ_δfϕₜ(J.L,Ł(J.fₜ),Ł(fϕ)...,s,t)
*(fϕ::FΦTuple, J::δfϕₛ_δfϕₜ{s,t,<:LenseFlow}) where {s,t} = δfϕₛ_δfϕₜᴴ(J.L,Ł(J.fₛ),Ł(fϕ)...,s,t)


## Jacobian

""" (δfϕₛ(fₜ,ϕ)/δfϕₜ) * (δf,δϕ) """
function δfϕₛ_δfϕₜ(L::LenseFlow{I}, fₜ::Field, δf::Field, δϕ::Field, s::Real, t::Real) where {I}
    FieldTuple(I((v,t,y)->δvelocity!(v,L,y...,δϕ,t,Ł.(gradhess(δϕ))...),FieldTuple(fₜ,δf),t,s)[2], δϕ)
end

""" ODE velocity for the Jacobian flow """
function δvelocity!(f_δf′::Field2Tuple, L::LenseFlow, f::Field, δf::Field, δϕ::Field, t::Real, ∇δϕ, Hδϕ)

    @unpack ∇ϕ,Hϕ = L
    M⁻¹ = Ł(inv(𝕀 + t*Hϕ))
    ∇f  = Ł(∇*Ð(f))
    ∇δf = Ł(∇*Ð(δf))

    f_δf′[1] .= @⨳ ∇ϕ' ⨳ M⁻¹ ⨳ ∇f
    f_δf′[2] .= (∇ϕ' ⨳ M⁻¹ ⨳ ∇δf) + (∇δϕ' ⨳ M⁻¹ ⨳ ∇f) - t*(∇ϕ' ⨳ M⁻¹ ⨳ Hδϕ ⨳ M⁻¹ ⨳ ∇f)

end


## transpose Jacobian

""" Compute (δfϕₛ(fₛ,ϕ)/δfϕₜ)' * (δf,δϕ) """
function δfϕₛ_δfϕₜᴴ(L::LenseFlow{I}, fₛ::Field, δf::Field, δϕ::Field, s::Real, t::Real) where {I}
    FieldTuple(I((v,t,y)->negδvelocityᴴ!(v,L,y...,t),FieldTuple(fₛ,δf,δϕ), s,t)[2:3]...)
end


""" ODE velocity for the negative transpose Jacobian flow """
function negδvelocityᴴ!(f_δf_δϕ′::Field3Tuple, L::LenseFlow, f::Field, δf::Field, δϕ::Field, t::Real)

    Łδf        = Ł(δf)
    M⁻¹        = Ł(inv(𝕀 + t*L.Hϕ))
    ∇f         = Ł(∇*Ð(f))
    M⁻¹_δfᵀ_∇f = Ł(M⁻¹ ⨳ (Łδf'*∇f))
    M⁻¹_∇ϕ     = Ł(M⁻¹ ⨳ L.∇ϕ)

    f_δf_δϕ′[1] .= @⨳ L.∇ϕ' ⨳ M⁻¹ ⨳ ∇f
    f_δf_δϕ′[2] .= Ł(@⨳ ∇ᵀ ⨳ $Ð(Łδf*M⁻¹_∇ϕ))
    f_δf_δϕ′[3] .= Ł(@⨳ ∇ᵀ ⨳ $Ð(M⁻¹_δfᵀ_∇f) + t*(∇ᵀ ⨳ ((∇ᵀ ⨳ $Ð(M⁻¹_∇ϕ ⨳ M⁻¹_δfᵀ_∇f'))')))

end



"""
Solve for y(t₁) with 4th order Runge-Kutta assuming dy/dt = F(t,y) and y(t₀) = y₀

Arguments
* F! : a function F!(v,t,y) which sets v=F(t,y)
"""
function jrk4(F!::Function, y₀, t₀, t₁, nsteps)
    h = (t₁-t₀)/nsteps
    y = copy(y₀)
    k₁, k₂, k₃, k₄ = @repeated(similar(y₀),4)
    for t in linspace(t₀,t₁,nsteps+1)[1:end-1]
        @! k₁ = F!(t, y)
        @! k₂ = F!(t + (h/2), y + (h/2)*k₁)
        @! k₃ = F!(t + (h/2), y + (h/2)*k₂)
        @! k₄ = F!(t +   (h), y +   (h)*k₃)
        @. y += h*(k₁ + 2k₂ + 2k₃ + k₄)/6
    end
    return y
end
