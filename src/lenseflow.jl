
export LenseFlow, CachedLenseFlow

abstract type ODESolver end

abstract type LenseFlowOp{I<:ODESolver,t₀,t₁} <: LenseOp end

struct LenseFlow{I<:ODESolver,t₀,t₁,F<:Field} <: LenseFlowOp{I,t₀,t₁}
    ϕ::F
    ∇ϕ::SVector{2,F}
    Hϕ::SMatrix{2,2,F,4}
end

# constructors
LenseFlow{I}(ϕ::Field{<:Any,<:S0}) where {I<:ODESolver} = LenseFlow{I,0,1}(ϕ)
LenseFlow{n}(ϕ::Field{<:Any,<:S0}) where {n} = LenseFlow{jrk4{n},0,1}(ϕ)
LenseFlow{I,t₀,t₁}(ϕ::Field{<:Any,<:S0}) where {I,t₀,t₁} = LenseFlow{I,t₀,t₁}(Map(ϕ), gradhess(ϕ)...)
LenseFlow{I,t₀,t₁}(ϕ::F,∇ϕ,Hϕ) where {I,t₀,t₁,F} = LenseFlow{I,float(t₀),float(t₁),F}(ϕ,∇ϕ,Hϕ)
LenseFlow(args...) = LenseFlow{jrk4{7}}(args...)

# only one single ODE solver implemented for now, a simple custom RK4
abstract type jrk4{nsteps} <: ODESolver  end
jrk4{N}(F!,y₀,t₀,t₁) where {N} = jrk4(F!,y₀,t₀,t₁,N)

# todo, remove this `→` crap, maybe
@∷ _getindex(L::LenseFlow{I,∷,∷,F}, ::→{t₀,t₁}) where {I,t₀,t₁,F} = LenseFlow{I,t₀,t₁,F}(L.ϕ,L.∇ϕ,L.Hϕ)

# Define integrations for L*f, L'*f, L\f, and L'\f
*(L::        LenseFlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = I((v,t,f)->velocity!( v,L, f,t), Ł(f), t₀, t₁)
*(L::AdjOp{<:LenseFlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = I((v,t,f)->velocityᴴ!(v,L',f,t), Ð(f), t₁, t₀)
\(L::        LenseFlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = I((v,t,f)->velocity!( v,L, f,t), Ł(f), t₁, t₀)
\(L::AdjOp{<:LenseFlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = I((v,t,f)->velocityᴴ!(v,L',f,t), Ð(f), t₀, t₁)
# Define integrations for Jacobian
*(J::δfϕₛ_δfϕₜ{s,t,<:LenseFlowOp{I}}, (δf,δϕ)::FΦTuple) where {s,t,I} = 
    (gh = Ł.(gradhess(δϕ)); FieldTuple(I((v,t,y)->δvelocity!(v,J.L,y...,δϕ,t,gh...),Ł(FieldTuple(J.fₜ,δf)),t,s)[2], δϕ))
*(J::AdjOp{<:δfϕₛ_δfϕₜ{s,t,<:LenseFlowOp{I}}}, (δf,δϕ)::FΦTuple) where {s,t,I} =
    FieldTuple(I((v,t,y)->negδvelocityᴴ!(v,J'.L,y...,t),FieldTuple(Ł(J'.fₛ),Ð(δf),Ð(δϕ)),s,t)[2:3]...)


# lensing velocities
 velocity!(v::Field, L::LenseFlow, f::Field, t::Real) = (@. v = L.∇ϕ' ⨳ $(inv(I + t*L.Hϕ)) ⨳ $(Ł(∇ᵢ*f)))
velocityᴴ!(v::Field, L::LenseFlow, f::Field, t::Real) = (@. v = Ł(∇ᵢ' ⨳ (Ł(f) * (inv(I + t*L.Hϕ) ⨳ L.∇ϕ))))
# Jacobian velocities
function δvelocity!((f′,δf′)::FieldTuple, L::LenseFlow, f::Field, δf::Field, δϕ::Field, t::Real, ∇δϕ, Hδϕ)

    @unpack ∇ϕ,Hϕ = L
    M⁻¹ = Ł(inv(I + t*Hϕ))
    ∇f  = Ł(∇*f)
    ∇δf = Ł(∇*δf)

    @. f′  =  ∇ϕ' ⨳ M⁻¹ ⨳ ∇f
    @. δf′ = (∇ϕ' ⨳ M⁻¹ ⨳ ∇δf) + (∇δϕ' ⨳ M⁻¹ ⨳ ∇f) - t*(∇ϕ' ⨳ M⁻¹ ⨳ Hδϕ ⨳ M⁻¹ ⨳ ∇f)

end
""" ODE velocity for the negative transpose Jacobian flow """
function negδvelocityᴴ!((f′,δf′,δϕ′)::FieldTuple, L::LenseFlow, f::Field, δf::Field, δϕ::Field, t::Real)

    Łδf        = Ł(δf)
    M⁻¹        = Ł(inv(I + t*L.Hϕ))
    ∇f         = Ł(∇*f)
    M⁻¹_δfᵀ_∇f = Ł(M⁻¹ * (Łδf' * ∇f))
    M⁻¹_∇ϕ     = Ł(M⁻¹ * L.∇ϕ)

    @. f′  = L.∇ϕ' ⨳ M⁻¹ ⨳ ∇f
    @. δf′ = Ł(∇' ⨳ (Łδf ⨳ M⁻¹_∇ϕ))
    @. δϕ′ = Ł(∇' ⨳ (M⁻¹_δfᵀ_∇f) + t⨳(∇' ⨳ (∇' ⨳ (M⁻¹_∇ϕ ⨳ M⁻¹_δfᵀ_∇f'))'))

end


## CachedLenseFlow

# This is a version of LenseFlow that precomputes the inverse magnification
# matrix, M⁻¹, and the p vector, p = M⁻¹⋅∇ϕ, when it is constructed. The regular
# version of LenseFlow computes these on the fly during the integration, which
# is faster if you only apply the lensing operator or its Jacobian once.
# However, *this* version is faster is you apply the operator or its Jacobian
# several times for a given ϕ. This is useful, for example, during Wiener
# filtering with a fixed ϕ, or computing the likelihood gradient which involves
# lensing and 1 or 2 (depending on parametrization) Jacobian evaluations all
# with the same ϕ.


struct CachedLenseFlow{N,t₀,t₁,Φ<:Field,ŁF<:Field,ÐF<:Field} <: LenseFlowOp{jrk4{N},t₀,t₁}
    L   :: LenseFlow{jrk4{N},t₀,t₁,Φ}
    p   :: Dict{Float16,SVector{2,Φ}}
    M⁻¹ :: Dict{Float16,SMatrix{2,2,Φ}}
    memŁv  :: SVector{2,ŁF}
    memŁv′ :: SVector{2,ŁF}
    memÐv  :: SVector{2,ÐF}
    memŁf  :: ŁF
    memÐf  :: ÐF
end
CachedLenseFlow{N}(ϕ) where {N} = cache(LenseFlow{jrk4{N}}(ϕ))
function cache(L::LenseFlow{jrk4{N},t₀,t₁,Φ},f) where {N,t₀,t₁,Φ}
    ts = linspace(t₀,t₁,2N+1)
    p, M⁻¹ = Dict(), Dict()
    for (t,τ) in zip(ts,τ.(ts))
        M⁻¹[τ] = inv(I + t*L.Hϕ) #TODO: remove need for Float32
        p[τ]  = M⁻¹[τ] ⨳ L.∇ϕ
    end
    CachedLenseFlow{N,t₀,t₁,Φ,typeof(Ł(f)),typeof(Ð(f))}(L,p,M⁻¹,Ł(∇*f),Ł(∇*f),Ð(∇*f),copy(Ł(f)),copy(Ð(f)))
end
cache(L::CachedLenseFlow) = L
τ(t) = Float16(t)

# velocities for CachedLenseFlow which use the precomputed quantities and preallocated memory

# the way these velocities work is that they unpack the preallocated fields
# stored in L.mem* into variables with more meaningful names, which are then
# used in a bunch of in-place (eg mul!, Ł!, etc...) functions. note the use of
# the @! macro, which just switches @! x = f(y) to f!(x,y) for easier reading. 

function velocity!(v::Field, L::CachedLenseFlow, f::Field, t::Real)
    Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐv,  L.memŁv
    p = L.p[τ(t)]
    
    @! Ðf  = Ð(f)
    @! Ð∇f = ∇*Ðf
    @! Ł∇f = Ł(Ð∇f)
    @⨳ v  = p' ⨳ Ł∇f
end

function velocityᴴ!(v::Field, L::CachedLenseFlow, f::Field, t::Real)
    Łf, Łf_p, Ð_Łf_p, tmpÐf = L.memŁf, L.memŁv, L.memÐv, L.memÐv[1]
    p = L.p[τ(t)]
    
    @! Łf = Ł(f)
    @! Łf_p = Łf * p
    @! Ð_Łf_p = Ð(Łf_p)
    @! v = mul!(∇', Ð_Łf_p, tmpÐf)
end

function negδvelocityᴴ!(v_f_δf_δϕ′::FieldTuple, L::CachedLenseFlow, f::Field, δf::Field, δϕ::Field, t::Real)

    Łδf        = Ł(δf)
    M⁻¹        = L.M⁻¹[τ(t)]
    ∇f         = Ł(∇*Ð(f))
    M⁻¹_δfᵀ_∇f = Ł(M⁻¹ ⨳ (Łδf'*∇f))
    M⁻¹_∇ϕ     = L.p[τ(t)]

    v_f_δf_δϕ′.fs[1] .= @⨳ M⁻¹_∇ϕ' ⨳ ∇f
    v_f_δf_δϕ′.fs[2] .= Ł(@⨳ ∇' ⨳ $Ð(Łδf*M⁻¹_∇ϕ))
    # split into two terms due to inference limit:
    tmp = @⨳ ∇' ⨳ $Ð(M⁻¹_δfᵀ_∇f)
    tmp .+= @⨳ t*(∇' ⨳ ((∇' ⨳ $Ð(M⁻¹_∇ϕ ⨳ M⁻¹_δfᵀ_∇f'))'))
    v_f_δf_δϕ′.fs[3] .= Ł(tmp)

end
# no specialized version for these (yet):
δvelocity!(v_f_δf, L::CachedLenseFlow, args...) = δvelocity!(v_f_δf, L.L, args...)

# changing integration endpoints causes a re-caching (although swapping them does not)
_getindex(L::CachedLenseFlow{N,t₀,t₁}, ::→{t₀,t₁}) where {t₀,t₁,N} = L
_getindex(L::CachedLenseFlow{N,t₁,t₀}, ::→{t₀,t₁}) where {t₀,t₁,N} = CachedLenseFlow(L.L[t₀→t₁],L.p,L.M⁻¹)
_getindex(L::CachedLenseFlow,          ::→{t₀,t₁}) where {t₀,t₁}   = cache(L.L[t₀→t₁])

# ud_grading lenseflow ud_grades the ϕ map
ud_grade(L::LenseFlow{I,t₀,t₁}, args...; kwargs...) where {I,t₀,t₁} = LenseFlow{I,t₀,t₁}(ud_grade(L.ϕ,args...;kwargs...))
ud_grade(L::CachedLenseFlow, args...; kwargs...)  = cache(ud_grade(L.L,args...;kwargs...))

"""
Solve for y(t₁) with 4th order Runge-Kutta assuming dy/dt = F(t,y) and y(t₀) = y₀

Arguments
* F! : a function F!(v,t,y) which sets v=F(t,y)
"""
function jrk4(F!::Function, y₀, t₀, t₁, nsteps)
    h = (t₁-t₀)/nsteps
    y = copy(y₀)
    k₁, k₂, k₃, k₄, y′ = @repeated(similar(y₀),5)
    for t in linspace(t₀,t₁,nsteps+1)[1:end-1]
        @! k₁ = F!(t, y)
        @! k₂ = F!(t + (h/2), (@. y′ = y + (h/2)*k₁))
        @! k₃ = F!(t + (h/2), (@. y′ = y + (h/2)*k₂))
        @! k₄ = F!(t +   (h), (@. y′ = y +   (h)*k₃))
        @. y += h*(k₁ + 2k₂ + 2k₃ + k₄)/6
    end
    return y
end
