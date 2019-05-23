
export LenseFlow, CachedLenseFlow

abstract type ODESolver end
# only one single ODE solver implemented for now, a simple custom RK4:
abstract type jrk4{nsteps} <: ODESolver  end

abstract type LenseFlowOp{I<:ODESolver,t₀,t₁} <: LenseOp end


# `L = LenseFlow(ϕ)` just creates a wrapper holding ϕ. Then when you do `L*f` or
# `cache(L,f)` we create a CachedLenseFlow object which holds all the
# precomputed quantities and preallocated memory needed to do the lense.

struct LenseFlow{I<:ODESolver,t₀,t₁,Φ<:Field{<:Any,<:S0}} <: LenseFlowOp{I,t₀,t₁}
    ϕ::Φ
end

struct CachedLenseFlow{N,t₀,t₁,ŁΦ<:Field,ÐΦ<:Field,ŁF<:Field,ÐF<:Field} <: LenseFlowOp{jrk4{N},t₀,t₁}
    # p and M⁻¹ quantities precomputed at every time step
    p   :: Dict{Float16,FieldVector{ŁΦ}}
    M⁻¹ :: Dict{Float16,FieldMatrix{ŁΦ}}
    
    # f type memory 
    memŁf  :: ŁF
    memÐf  :: ÐF
    memŁvf :: FieldVector{ŁF}
    memÐvf :: FieldVector{ÐF}
    
    # ϕ type memory
    memŁϕ  :: ŁΦ
    memÐϕ  :: ÐΦ
    memŁvϕ :: FieldVector{ŁΦ}
    memÐvϕ :: FieldVector{ÐΦ}
end

# constructors
LenseFlow(ϕ,n=7) = LenseFlow{jrk4{n}}(ϕ)
LenseFlow{I}(ϕ) where {I<:ODESolver} = LenseFlow{I,0,1}(ϕ)
LenseFlow{I,t₀,t₁}(ϕ) where {I,t₀,t₁} = LenseFlow{I,float(t₀),float(t₁),typeof(ϕ)}(ϕ)

zero(L::LenseFlow) = zero(L.ϕ)
zero(L::CachedLenseFlow) = zero(L.memŁϕ)

# todo, remove this `→` crap, maybe
@∷ _getindex(L::LenseFlow{I,∷,∷,F}, ::→{t₀,t₁}) where {I,t₀,t₁,F} = LenseFlow{I,t₀,t₁,F}(L.ϕ)

# Define integrations for L*f, L'*f, L\f, and L'\f
*(L::        LenseFlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = (cL=cache(L,f);  I((v,t,f)->velocity!( v,cL,f,t), Ł(f), t₀, t₁))
*(L::AdjOp{<:LenseFlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = (cL=cache(L',f); I((v,t,f)->velocityᴴ!(v,cL,f,t), Ð(f), t₁, t₀))
\(L::        LenseFlowOp{I,t₀,t₁},  f::Field) where {I,t₀,t₁} = (cL=cache(L,f);  I((v,t,f)->velocity!( v,cL,f,t), Ł(f), t₁, t₀))
\(L::AdjOp{<:LenseFlowOp{I,t₀,t₁}}, f::Field) where {I,t₀,t₁} = (cL=cache(L',f); I((v,t,f)->velocityᴴ!(v,cL,f,t), Ð(f), t₀, t₁))

# Define integrations for Jacobians
*(J::AdjOp{<:δfϕₛ_δfϕₜ{s,t,<:LenseFlowOp{I}}}, (δf,δϕ)::FΦTuple) where {s,t,I} =
    (cL=cache(J'.L,δf); FieldTuple(I((v,t,y)->negδvelocityᴴ!(v,cL,y,t),FieldTuple(Ł(J'.fₛ),Ð(δf),Ð(δϕ)),s,t)[2:3]...))


τ(t) = Float16(t)
cache(L::LenseFlow, f) = cache!(alloc_cache(L,f),L,f)
cache(L::CachedLenseFlow, f) = L
function cache!(cL::CachedLenseFlow{N,t₀,t₁}, L::LenseFlow{jrk4{N},t₀,t₁}, f) where {N,t₀,t₁}
    ts = linspace(t₀,t₁,2N+1)
    ∇ϕ,Hϕ = Map.(gradhess(L.ϕ))
    T = eltype(L.ϕ)
    for (t,τ) in zip(ts,τ.(ts))
        @! cL.M⁻¹[τ] = inv(T(1)*I + T(t)*Hϕ)
        @! cL.p[τ] = permutedims(cL.M⁻¹[τ]) * ∇ϕ
    end
    cL
end
function alloc_cache(L::LenseFlow{jrk4{N},t₀,t₁}, f) where {N,t₀,t₁}
    ts = linspace(t₀,t₁,2N+1)
    p, M⁻¹ = Dict(), Dict()
    Łf,Ðf = Ł(f),  Ð(f)
    Łϕ,Ðϕ = Ł(L.ϕ),Ð(L.ϕ)
    for (t,τ) in zip(ts,τ.(ts))
        M⁻¹[τ] = similar.(@SMatrix[Łϕ Łϕ; Łϕ Łϕ])
        p[τ]   = similar.(@SVector[Łϕ,Łϕ])
    end
    CachedLenseFlow{N,t₀,t₁,typeof(Łϕ),typeof(Ðϕ),typeof(Łf),typeof(Ðf)}(
        p, M⁻¹, 
        similar(Łf), similar(Ðf), similar.(@SVector[Łf,Łf]), similar.(@SVector[Ðf,Ðf]),
        similar(Łϕ), similar(Ðϕ), similar.(@SVector[Łϕ,Łϕ]), similar.(@SVector[Ðϕ,Ðϕ]),
    )
end

# the way these velocities work is that they unpack the preallocated fields
# stored in L.mem* into variables with more meaningful names, which are then
# used in a bunch of in-place (eg mul!, Ł!, etc...) functions. note the use of
# the @! macro, which just rewrites @! x = f(y) to x = f!(x,y) for easier
# reading. 

function velocity!(v::Field, L::CachedLenseFlow, f::Field, t::Real)
    Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf,  L.memŁvf
    p = L.p[τ(t)]
    
    @! Ðf  = Ð(f)
    @! Ð∇f = ∇ᵢ*Ðf
    @! Ł∇f = Ł(Ð∇f)
    @. v  = p' ⨳ Ł∇f
end

function velocityᴴ!(v::Field, L::CachedLenseFlow, f::Field, t::Real)
    Łf, Łf_p, Ð_Łf_p = L.memŁf, L.memŁvf, L.memÐvf
    p = L.p[τ(t)]
    
    @! Łf = Ł(f)
    @! Łf_p = Łf * p
    @! Ð_Łf_p = Ð(Łf_p)
    @! v = ∇ᵢ' * Ð_Łf_p
end

function negδvelocityᴴ!((df_dt, dδf_dt, dδϕ_dt)::FieldTuple, L::CachedLenseFlow, (f, δf, δϕ)::FieldTuple, t::Real)
    
    p   = L.p[τ(t)]
    M⁻¹ = L.M⁻¹[τ(t)]
    
    # dδf/dt
    Łδf, Łδf_p, Ð_Łδf_p = L.memŁf, L.memŁvf, L.memÐvf
    @! Łδf     = Ł(δf)
    @! Łδf_p   = Łδf * p
    @! Ð_Łδf_p = Ð(Łδf_p)
    @! dδf_dt  = ∇ᵢ' * Ð_Łδf_p
    
    # df/dt
    Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf,  L.memŁvf
    @! Ðf     = Ð(f)
    @! Ð∇f    = ∇ᵢ * Ðf
    @! Ł∇f    = Ł(Ð∇f)
    @. df_dt  = p' ⨳ Ł∇f

    # dδϕ/dt
    δfᵀ_∇f, M⁻¹_δfᵀ_∇f, Ð_M⁻¹_δfᵀ_∇f = L.memŁvϕ, L.memŁvϕ, L.memÐvϕ
    @! δfᵀ_∇f       = Łδf' * Ł∇f
    @! M⁻¹_δfᵀ_∇f   = M⁻¹ * δfᵀ_∇f
    @! Ð_M⁻¹_δfᵀ_∇f = Ð(M⁻¹_δfᵀ_∇f)
    @! dδϕ_dt       = ∇ⁱ' * Ð_M⁻¹_δfᵀ_∇f
    memÐϕ = L.memÐϕ
    for i=1:2, j=1:2
        dδϕ_dt .+= (@! memÐϕ = ∇ⁱ[i]' * (@! memÐϕ = ∇ᵢ[j]' * (@! memÐϕ = Ð(@. L.memŁϕ = t * p[j] * M⁻¹_δfᵀ_∇f[i]))))
    end
    
    FieldTuple(df_dt, dδf_dt, dδϕ_dt)
    
end

# can swap integration points without recaching, although not arbitrarily change them
_getindex(L::CachedLenseFlow{N,t₀,t₁,ŁΦ,ÐΦ,ŁF,ÐF}, ::→{t₀,t₁}) where {N,t₀,t₁,ŁΦ,ÐΦ,ŁF,ÐF} = L
_getindex(L::CachedLenseFlow{N,t₁,t₀,ŁΦ,ÐΦ,ŁF,ÐF}, ::→{t₀,t₁}) where {N,t₀,t₁,ŁΦ,ÐΦ,ŁF,ÐF} = CachedLenseFlow{N,t₀,t₁,ŁΦ,ÐΦ,ŁF,ÐF}(fieldvalues(L)...)

# # ud_grading lenseflow ud_grades the ϕ map
# ud_grade(L::LenseFlow{I,t₀,t₁}, args...; kwargs...) where {I,t₀,t₁} = LenseFlow{I,t₀,t₁}(ud_grade(L.ϕ,args...;kwargs...))
# ud_grade(L::CachedLenseFlow, args...; kwargs...)  = cache(ud_grade(L.L,args...;kwargs...))

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
jrk4{N}(F!,y₀,t₀,t₁) where {N} = jrk4(F!,y₀,t₀,t₁,N)
