
abstract type LenseFlowOp{S<:ODESolver,T} <: FlowOpWithAdjoint{T} end

# `L = LenseFlow(ϕ)` just creates a wrapper holding ϕ. Then when you do `L*f` or
# `precompute!!(L,f)` we create a CachedLenseFlow object which holds all the
# precomputed quantities and preallocated memory needed to do the lense.

"""
    LenseFlow(ϕ, [n=7])

`LenseFlow` is the ODE-based lensing algorithm from [Millea, Anderes,
& Wandelt, 2019](https://arxiv.org/abs/1708.06753). The number of
steps in the ODE solver is controlled by `n`. The action of the
operator, as well as its adjoint, inverse, inverse-adjoint, and
gradient of any of these w.r.t. `ϕ` can all be computed. The
log-determinant of the operation is zero independent of `ϕ`, in the
limit of `n` high enough.
"""
struct LenseFlow{S<:ODESolver,T} <: LenseFlowOp{S,T}
    ϕ :: Field
    odesolve :: S
    t₀ :: T
    t₁ :: T
    function LenseFlow(ϕ::Field, odesolve::S, t₀, t₁) where {S<:ODESolver}
        rT = ForwardDiff.valtype(real(eltype(ϕ)))
        new{S,rT}(ϕ, odesolve, rT(0), rT(1))
    end
end
LenseFlow(ϕ::Field, nsteps::Int=7) = LenseFlow(ϕ, RK4Solver(nsteps), 0, 1)
LenseFlow(nsteps::Int) = ϕ -> LenseFlow(ϕ, nsteps)


struct CachedLenseFlow{S<:ODESolver, T, D<:Diagonal, ŁΦ<:Field, ÐΦ<:Field, ŁF<:Field, ÐF<:Field} <: LenseFlowOp{S, T}
    
    # save ϕ to know when to trigger recaching
    ϕ :: Ref{Any}
    needs_precompute :: Ref{Bool}

    # ODE solver
    odesolve :: S
    t₀ :: T
    t₁ :: T
    
    # p and M⁻¹ quantities precomputed at every time step
    p   :: Dict{Float16,SVector{2,D}}
    M⁻¹ :: Dict{Float16,SMatrix{2,2,D,4}}
    
    # preallocated "wide" f-type memory
    memŁf  :: ŁF
    memÐf  :: ÐF
    memŁvf :: FieldVector{ŁF}
    memÐvf :: FieldVector{ÐF}
    
    # preallocated "wide" ϕ-type memory
    memŁϕ  :: ŁΦ
    memÐϕ  :: ÐΦ
    memŁvϕ :: FieldVector{ŁΦ}
    memÐvϕ :: FieldVector{ÐΦ}

end



### printing
size(L::CachedLenseFlow) = length(L.memŁf) .* (1, 1)


# convenience for getting the actual ϕ map
getϕ(L::LenseFlow) = L.ϕ
getϕ(L::CachedLenseFlow) = L.ϕ[]

# if the type and ϕ are the same, its the same op
hash(L::LenseFlowOp, h::UInt64) = foldr(hash, (typeof(L), getϕ(L)), init=h)


### caching

τ(t) = Float16(t)

function precompute!!(Lϕ::LenseFlow{S,T}, f) where {S<:RK4Solver, T}
    
    @unpack (ϕ, t₀, t₁, odesolve) = Lϕ
    @unpack nsteps = odesolve
    
    # p & M precomputed matrix elements will use exactly same type as ϕ
    Łϕ, Ðϕ = Ł(ϕ), Ð(ϕ)
    p, M⁻¹ = Dict(), Dict()
    τs     = τ.(range(t₀, stop=t₁, length=2nsteps+1))
    p      = Dict(map(τ -> (τ => Diagonal.(similar.(@SVector[Łϕ, Łϕ]))),       τs))
    M⁻¹    = Dict(map(τ -> (τ => Diagonal.(similar.(@SMatrix[Łϕ Łϕ; Łϕ Łϕ]))), τs))

    # preallocated memory need to be "wide" enough to handle the
    # batching and/or Dual-ness of both f and ϕ. this is a kind of
    # hacky way to get fields that are this wide given the input f and ϕ:
    f′ = Ł(ϕ) .* Ł(f)
    ϕ′ = spin_adjoint(f′) * f′
    Łϕ′, Ðϕ′ = Ł(ϕ′), Ð(ϕ′)
    Łf′, Ðf′ = Ł(f′), Ð(f′)

    cLϕ = CachedLenseFlow(
        Ref{Any}(ϕ), Ref(false),
        odesolve, t₀, t₁,
        p, M⁻¹, 
        similar(Łf′), similar(Ðf′), similar.(@SVector[Łf′,Łf′]), similar.(@SVector[Ðf′,Ðf′]),
        similar(Łϕ′), similar(Ðϕ′), similar.(@SVector[Łϕ′,Łϕ′]), similar.(@SVector[Ðϕ′,Ðϕ′]),
    )

    return precompute!(cLϕ)
end

function precompute!!(Lϕ::CachedLenseFlow, f)
    if real(eltype(f)) == real(eltype(Lϕ.memŁf))
        if Lϕ.needs_precompute[]
            precompute!(Lϕ)
            Lϕ.needs_precompute[] = false
        end
        return Lϕ
    else
        return precompute!!(LenseFlow(Lϕ.ϕ[], Lϕ.odesolve, Lϕ.t₀, Lϕ.t₁), f)
    end
end

function (Lϕ::CachedLenseFlow)(ϕ::Field)
    if Lϕ.ϕ[] !== ϕ
        Lϕ.ϕ[] = ϕ
        Lϕ.needs_precompute[] = true
    end
    Lϕ
end

function precompute!(Lϕ::CachedLenseFlow{S,T}) where {S,T}
    # @info "Precomputing $T"
    @unpack (ϕ, t₀, t₁, odesolve) = Lϕ
    @unpack nsteps = odesolve
    ts = range(t₀, stop=t₁, length=2nsteps+1)
    ∇ϕ, ∇∇ϕ = map(Ł, gradhess(ϕ[]))
    for (t, τ) in zip(ts,τ.(ts))
        @! Lϕ.M⁻¹[τ] = pinv(Diagonal.(I + T(t)*∇∇ϕ))
        @! Lϕ.p[τ] = Lϕ.M⁻¹[τ]' * Diagonal.(∇ϕ)
    end
    Lϕ
end

# the way these velocities work is that they unpack the preallocated fields
# stored in L.mem* into variables with more meaningful names, which are then
# used in a bunch of in-place (eg mul!, Ł!, etc...) functions. note the use of
# the @! macro, which just rewrites @! x = f(y) to x = f!(x,y) for easier
# reading. 

function velocity(L::CachedLenseFlow, f₀::Field)
    function v!(v::Field, t::Real, f::Field)
        Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf,  L.memŁvf
        p = L.p[τ(t)]
        
        @! Ðf  = Ð(f)
        @! Ð∇f = ∇ᵢ * Ðf
        @! Ł∇f = Ł(Ð∇f)
        @! v   = p' * Ł∇f
    end
    return (v!, L.memŁf .= Ł(f₀))
end

function velocityᴴ(L::CachedLenseFlow, f₀::Field)
    function v!(v::Field, t::Real, f::Field)
        Łf, Łf_p, Ð_Łf_p = L.memŁf, L.memŁvf, L.memÐvf
        p = L.p[τ(t)]
        
        @! Łf = Ł(f)
        @! Łf_p = p * Łf
        @! Ð_Łf_p = Ð(Łf_p)
        @! v = -∇ᵢ' * Ð_Łf_p
    end
    return (v!, L.memÐf .= Ð(f₀))
end

function negδvelocityᴴ(L::CachedLenseFlow, (f₀, δf₀)::FieldTuple)
    
    function v!((df_dt, dδf_dt, dδϕ_dt)::FieldTuple, t::Real, (f, δf, δϕ)::FieldTuple)
    
        p   = L.p[τ(t)]
        M⁻¹ = L.M⁻¹[τ(t)]
        
        # dδf/dt
        Łδf, Łδf_p, Ð_Łδf_p = L.memŁf, L.memŁvf, L.memÐvf
        @! Łδf     = Ł(δf)
        @! Łδf_p   = p * Łδf
        @! Ð_Łδf_p = Ð(Łδf_p)
        @! dδf_dt  = -∇ᵢ' * Ð_Łδf_p
        
        # df/dt
        Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf,  L.memŁvf
        @! Ðf     = Ð(f)
        @! Ð∇f    = ∇ᵢ * Ðf
        @! Ł∇f    = Ł(Ð∇f)
        @! df_dt  = p' * Ł∇f

        # dδϕ/dt
        δfᵀ_∇f, M⁻¹_δfᵀ_∇f, Ð_M⁻¹_δfᵀ_∇f = L.memŁvϕ, L.memŁvϕ, L.memÐvϕ
        @! δfᵀ_∇f       = spin_adjoint(Łδf) * Ł∇f
        @! M⁻¹_δfᵀ_∇f   = M⁻¹ * δfᵀ_∇f
        @! Ð_M⁻¹_δfᵀ_∇f = Ð(M⁻¹_δfᵀ_∇f)
        @! dδϕ_dt       = -∇ⁱ' * Ð_M⁻¹_δfᵀ_∇f
        memÐϕ = L.memÐϕ
        for i=1:2, j=1:2
            dδϕ_dt .+= (@! memÐϕ = ∇ⁱ[i]' * (@! memÐϕ = ∇ᵢ[j]' * (@! memÐϕ = Ð(@. L.memŁϕ = t * p[j].diag * M⁻¹_δfᵀ_∇f[i]))))
        end
        
        FieldTuple(df_dt, dδf_dt, dδϕ_dt)
    
    end
    
    return (v!, FieldTuple(L.memŁf .= Ł(f₀), L.memÐf .= Ð(δf₀), L.memÐϕ .= Ð(zero(getϕ(L)))))
    
end

# adapting storage
adapt_structure(storage, Lϕ::LenseFlow) = LenseFlow(adapt(storage, Lϕ.ϕ), Lϕ.odesolve, Lϕ.t₀, Lϕ.t₁)
function adapt_structure(storage, Lϕ::CachedLenseFlow)
    # Adapt may reuse storage (e.g. Array -> Array), but distinct caches must
    # not share mutable fields: recomputing one would corrupt the other.
    function _adapt(f::BaseField)
        f′ = adapt(storage, f)
        Base.mightalias(f.arr, f′.arr) ? copy(f′) : f′
    end
    _adapt(f::FieldTuple) = FieldTuple(map(_adapt, f.fs))
    _adapt(f::Field) = copy(adapt(storage, f))
    _adapt(d::Diagonal) = Diagonal(_adapt(d.diag))
    CachedLenseFlow(
        Ref{Any}(_adapt(Lϕ.ϕ[])), Ref(Lϕ.needs_precompute[]),
        Lϕ.odesolve, Lϕ.t₀, Lϕ.t₁,
        Dict(τ => _adapt.(x) for (τ,x) in Lϕ.p), 
        Dict(τ => _adapt.(x) for (τ,x) in Lϕ.M⁻¹),
        _adapt(Lϕ.memŁf), _adapt(Lϕ.memÐf), _adapt.(Lϕ.memŁvf), _adapt.(Lϕ.memÐvf),
        _adapt(Lϕ.memŁϕ), _adapt(Lϕ.memÐϕ), _adapt.(Lϕ.memŁvϕ), _adapt.(Lϕ.memÐvϕ)
    )
end



"""
Returns αmax such that 𝕀 + ∇∇(ϕ + α * η) has non-zero discriminant
(pixel-by-pixel) for all α values in [0, αmax]. 

This mean ϕ + αmax * η is the maximum step in the η direction which
can be added to ϕ and still yield a lensing potential in the
weak-lensing regime. This is important because it guarantees the
potential can be paseed to LenseFlow, which cannot handle the
strong-lensing / "shell-crossing" regime.
"""
function get_max_lensing_step(ϕ, η)

    ϕ₁₁, ϕ₁₂, ϕ₂₁, ϕ₂₂ = Map.(gradhess(ϕ)[2])
    η₁₁, η₁₂, η₂₁, η₂₂ = Map.(gradhess(η)[2])

    a = @. η₁₁*η₂₂ - η₁₂^2
    b = @. η₁₁*(1+ϕ₂₂) + η₂₂*(1+ϕ₁₁) - 2*η₁₂*ϕ₁₂
    c = @. (1+ϕ₁₁) * (1+ϕ₂₂) - ϕ₁₂^2

    α₁ = cpu(@. (-b + sqrt(b^2 - 4*a*c))/(2*a))
    α₂ = cpu(@. (-b - sqrt(b^2 - 4*a*c))/(2*a))

    αmax = min(minimum(α₁[α₁.>0]), minimum(α₂[α₂.>0]))

end
