
abstract type LenseFlowOp{I<:ODESolver,t₀,t₁,Φ} <: FlowOpWithAdjoint{I,t₀,t₁} end

# `L = LenseFlow(ϕ)` just creates a wrapper holding ϕ. Then when you do `L*f` or
# `cache(L,f)` we create a CachedLenseFlow object which holds all the
# precomputed quantities and preallocated memory needed to do the lense.

struct LenseFlow{I<:ODESolver,t₀,t₁,Φ<:Field{<:Any,<:S0}} <: LenseFlowOp{I,t₀,t₁,Φ}
    ϕ::Φ
end

struct CachedLenseFlow{N,t₀,t₁,Φ<:Field,ŁΦ<:Field,ÐΦ<:Field,ŁF<:Field,ÐF<:Field,T} <: LenseFlowOp{RK4Solver{N},t₀,t₁,Φ}
    
    # save ϕ to know when to trigger recaching
    ϕ :: Ref{Any}
    
    # p and M⁻¹ quantities precomputed at every time step
    p   :: Dict{Float16,FieldOrOpVector{Diagonal{T,ŁΦ}}}
    M⁻¹ :: Dict{Float16,FieldOrOpMatrix{Diagonal{T,ŁΦ}}}
    
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

### constructors
LenseFlow(ϕ,n=7) = LenseFlow{RK4Solver{n}}(ϕ)
LenseFlow{I}(ϕ) where {I<:ODESolver} = LenseFlow{I,0,1}(ϕ)
LenseFlow{I,t₀,t₁}(ϕ) where {I,t₀,t₁} = LenseFlow{I,float(t₀),float(t₁),typeof(ϕ)}(ϕ)


### printing
show(io::IO, ::L) where {I,t₀,t₁,Φ,L<:LenseFlow{I,t₀,t₁,Φ}} = print(io, "$(L.name.name){$t₀→$t₁, $I}(ϕ::$Φ)")
show(io::IO, ::L) where {N,t₀,t₁,Φ,ŁF,L<:CachedLenseFlow{N,t₀,t₁,Φ,<:Any,<:Any,ŁF}} = print(io, "$(L.name.name){$t₀→$t₁, $(RK4Solver{N})}(ϕ::$Φ, Łf::$ŁF)")
string(::Type{RK4Solver{N}}) where {N} = "$N-step RK4"

# convenience for getting the actual ϕ map
getϕ(L::LenseFlow) = L.ϕ
getϕ(L::CachedLenseFlow) = L.ϕ[]


### caching
τ(t) = Float16(t)
cache(L::LenseFlow, f) = cache!(alloc_cache(L,f),L,f)
cache(cL::CachedLenseFlow, f) = cL
cache!(cL::CachedLenseFlow{N,t₀,t₁}, ϕ) where {N,t₀,t₁} = (cL.ϕ[]===ϕ) ? cL : cache!(cL,LenseFlow{RK4Solver{N},t₀,t₁}(ϕ),cL.memŁf)
(cL::CachedLenseFlow)(ϕ) = cache!(cL,ϕ)
function cache!(cL::CachedLenseFlow{N,t₀,t₁}, L::LenseFlow{RK4Solver{N},t₀,t₁}, f) where {N,t₀,t₁}
    ts = range(t₀,t₁,length=2N+1)
    ∇ϕ,Hϕ = map(Ł, gradhess(L.ϕ))
    T = eltype(L.ϕ)
    for (t,τ) in zip(ts,τ.(ts))
        @! cL.M⁻¹[τ] = pinv(Diagonal.(I + T(t)*Hϕ))
        @! cL.p[τ] = cL.M⁻¹[τ]' * Diagonal.(∇ϕ)
    end
    cL.ϕ[] = L.ϕ
    cL
end
function alloc_cache(L::LenseFlow{RK4Solver{N},t₀,t₁}, f) where {N,t₀,t₁}
    ts = range(t₀,t₁,length=2N+1)
    p, M⁻¹ = Dict(), Dict()
    Łf,Ðf = Ł(f),  Ð(f)
    Łϕ,Ðϕ = Ł(L.ϕ),Ð(L.ϕ)
    for (t,τ) in zip(ts,τ.(ts))
        M⁻¹[τ] = Diagonal.(similar.(@SMatrix[Łϕ Łϕ; Łϕ Łϕ]))
        p[τ]   = Diagonal.(similar.(@SVector[Łϕ,Łϕ]))
    end
    CachedLenseFlow{N,t₀,t₁,typeof(L.ϕ),typeof(Łϕ),typeof(Ðϕ),typeof(Łf),typeof(Ðf),eltype(Łϕ)}(
        Ref{Any}(L.ϕ), p, M⁻¹, 
        similar(Łf), similar(Ðf), similar.(@SVector[Łf,Łf]), similar.(@SVector[Ðf,Ðf]),
        similar(Łϕ), similar(Ðϕ), similar.(@SVector[Łϕ,Łϕ]), similar.(@SVector[Ðϕ,Ðϕ]),
    )
end


# the way these velocities work is that they unpack the preallocated fields
# stored in L.mem* into variables with more meaningful names, which are then
# used in a bunch of in-place (eg mul!, Ł!, etc...) functions. note the use of
# the @! macro, which just rewrites @! x = f(y) to x = f!(x,y) for easier
# reading. 

function velocity(L::LenseFlowOp{<:RK4Solver}, f₀::Field)
    function v!(v::Field, t::Real, f::Field)
        Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf,  L.memŁvf
        p = L.p[τ(t)]
        
        @! Ðf  = Ð(f)
        @! Ð∇f = ∇ᵢ * Ðf
        @! Ł∇f = Ł(Ð∇f)
        @! v   = p' * Ł∇f
    end
    return (v!, batch_promote!(L.memŁf,Ł(f₀)))
end

function velocityᴴ(L::LenseFlowOp{<:RK4Solver}, f₀::Field)
    function v!(v::Field, t::Real, f::Field)
        Łf, Łf_p, Ð_Łf_p = L.memŁf, L.memŁvf, L.memÐvf
        p = L.p[τ(t)]
        
        @! Łf = Ł(f)
        @! Łf_p = p * Łf
        @! Ð_Łf_p = Ð(Łf_p)
        @! v = -∇ᵢ' * Ð_Łf_p
    end
    return (v!, batch_promote!(L.memÐf,Ð(f₀)))
end

function negδvelocityᴴ(L::LenseFlowOp{<:RK4Solver}, (f₀, δf₀)::FieldTuple)
    
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
        @! δfᵀ_∇f       = tuple_adjoint(Łδf) * Ł∇f
        @! M⁻¹_δfᵀ_∇f   = M⁻¹ * δfᵀ_∇f
        @! Ð_M⁻¹_δfᵀ_∇f = Ð(M⁻¹_δfᵀ_∇f)
        @! dδϕ_dt       = -∇ⁱ' * Ð_M⁻¹_δfᵀ_∇f
        memÐϕ = L.memÐϕ
        for i=1:2, j=1:2
            dδϕ_dt .+= (@! memÐϕ = ∇ⁱ[i]' * (@! memÐϕ = ∇ᵢ[j]' * (@! memÐϕ = Ð(@. L.memŁϕ = t * p[j].diag * M⁻¹_δfᵀ_∇f[i]))))
        end
        
        FieldTuple(df_dt, dδf_dt, dδϕ_dt)
    
    end
    
    return (v!, FieldTuple(Ł(f₀), Ð(δf₀), batch_promote!(L.memÐϕ,Ð(zero(getϕ(L))))))
    
end

# adapting storage
adapt_structure(storage, Lϕ::LenseFlow{I,t₀,t₁}) where {I<:ODESolver,t₀,t₁} = LenseFlow{I,t₀,t₁}(adapt(storage,Lϕ.ϕ))
adapt_structure(storage, Lϕ::CachedLenseFlow{N,t₀,t₁}) where {N,t₀,t₁} = cache(LenseFlow{RK4Solver{N},t₀,t₁}(adapt(storage,Lϕ.ϕ[])), adapt(storage,Lϕ.memŁf))
