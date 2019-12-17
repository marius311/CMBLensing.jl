
struct QuasiLenseFlow{I<:ODESolver,t₀,t₁,Φ<:Field{<:Any,<:S0},TMÐ,TMŁ} <: LenseFlowOp{I,t₀,t₁,Φ}
    ϕ  :: Φ
    MÐ :: TMÐ
    MŁ :: TMŁ
end

struct CachedQuasiLenseFlow{N,t₀,t₁,Φ<:Field,ŁΦ<:Field,ÐΦ<:Field,ŁF<:Field,ÐF<:Field,T,TMÐ,TMŁ} <: LenseFlowOp{RK4Solver{N},t₀,t₁,Φ}
    
    # save ϕ to know when to trigger recaching
    ϕ  :: Ref{Φ}
    MÐ :: TMÐ
    MŁ :: TMŁ
    
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
QuasiLenseFlow(ϕ,MÐ,MŁ,n=7) = QuasiLenseFlow{RK4Solver{n}}(ϕ,MÐ,MŁ)
QuasiLenseFlow{I}(ϕ,MÐ,MŁ) where {I<:ODESolver} = QuasiLenseFlow{I,0,1}(ϕ,MÐ,MŁ)
QuasiLenseFlow{I,t₀,t₁}(ϕ,MÐ,MŁ) where {I,t₀,t₁} = QuasiLenseFlow{I,float(t₀),float(t₁),typeof(ϕ),typeof(MÐ),typeof(MŁ)}(ϕ,MÐ,MŁ)


### printing
show(io::IO, ::L) where {I,t₀,t₁,Φ,L<:QuasiLenseFlow{I,t₀,t₁,Φ}} = print(io, "$(L.name.name){$t₀→$t₁, $I}(ϕ::$Φ)")
show(io::IO, ::L) where {N,t₀,t₁,Φ,ŁF,L<:CachedQuasiLenseFlow{N,t₀,t₁,Φ,<:Any,<:Any,ŁF}} = print(io, "$(L.name.name){$t₀→$t₁, $(RK4Solver{N})}(ϕ::$Φ, Łf::$ŁF)")
string(::Type{RK4Solver{N}}) where {N} = "$N-step RK4"

# convenience for getting the actual ϕ map
getϕ(L::QuasiLenseFlow) = L.ϕ
getϕ(L::CachedQuasiLenseFlow) = L.ϕ[]

# todo, remove this `→` crap, maybe
_getindex(L::QuasiLenseFlow{I,<:Any,<:Any,F}, ::→{t₀,t₁}) where {I,t₀,t₁,F} = QuasiLenseFlow{I,t₀,t₁,F}(L.ϕ,L.MÐ,L.MŁ)


τ(t) = Float16(t)
cache(L::QuasiLenseFlow, f) = cache!(alloc_cache(L,f),L,f)
cache(cL::CachedQuasiLenseFlow, f) = cL
cache!(cL::CachedQuasiLenseFlow{N,t₀,t₁}, ϕ) where {N,t₀,t₁} = (cL.ϕ[]===ϕ) ? cL : cache!(cL,QuasiLenseFlow{RK4Solver{N},t₀,t₁}(ϕ,cL.MÐ,cL.MŁ),cL.memŁf)
(cL::CachedQuasiLenseFlow)(ϕ) = cache!(cL,ϕ)
function cache!(cL::CachedQuasiLenseFlow{N,t₀,t₁}, L::QuasiLenseFlow{RK4Solver{N},t₀,t₁}, f) where {N,t₀,t₁}
    @assert cL.MÐ==L.MÐ && cL.MŁ==L.MŁ
    ts = range(t₀,t₁,length=2N+1)
    ∇ϕ,Hϕ = map(Ł, gradhess(L.ϕ))
    T = eltype(L.ϕ)
    for (t,τ) in zip(ts,τ.(ts))
        @! cL.M⁻¹[τ] = pinv(Diagonal.(I + T(t) * Hϕ))
        for m in cL.M⁻¹[τ] 
            m.diag .*= L.MŁ.diag
        end
        @! cL.p[τ] = cL.M⁻¹[τ]' * Diagonal.(∇ϕ)
    end
    cL.ϕ[] = L.ϕ
    cL
end
function alloc_cache(L::QuasiLenseFlow{RK4Solver{N},t₀,t₁}, f) where {N,t₀,t₁}
    ts = range(t₀,t₁,length=2N+1)
    p, M⁻¹ = Dict(), Dict()
    Łf,Ðf = Ł(f),  Ð(f)
    Łϕ,Ðϕ = Ł(L.ϕ),Ð(L.ϕ)
    for (t,τ) in zip(ts,τ.(ts))
        M⁻¹[τ] = Diagonal.(similar.(@SMatrix[Łϕ Łϕ; Łϕ Łϕ]))
        p[τ]   = Diagonal.(similar.(@SVector[Łϕ,Łϕ]))
    end
    CachedQuasiLenseFlow{N,t₀,t₁,typeof(L.ϕ),typeof(Łϕ),typeof(Ðϕ),typeof(Łf),typeof(Ðf),eltype(Łϕ),typeof(L.MÐ),typeof(L.MŁ)}(
        Ref(L.ϕ), L.MÐ, L.MŁ, p, M⁻¹,
        similar(Łf), similar(Ðf), similar.(@SVector[Łf,Łf]), similar.(@SVector[Ðf,Ðf]),
        similar(Łϕ), similar(Ðϕ), similar.(@SVector[Łϕ,Łϕ]), similar.(@SVector[Ðϕ,Ðϕ]),
    )
end


# the way these velocities work is that they unpack the preallocated fields
# stored in L.mem* into variables with more meaningful names, which are then
# used in a bunch of in-place (eg mul!, Ł!, etc...) functions. note the use of
# the @! macro, which just rewrites @! x = f(y) to x = f!(x,y) for easier
# reading. 

function velocity!(v::Field, L::CachedQuasiLenseFlow, f::Field, t::Real)
    Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf,  L.memŁvf
    p = L.p[τ(t)]
    
    @! Ðf  = Ð(f)
    @! Ð∇f = L.MÐ * ∇ᵢ * Ðf
    @! Ł∇f = Ł(Ð∇f)
    @! v   = p' * Ł∇f
end

function velocityᴴ!(v::Field, L::CachedQuasiLenseFlow, f::Field, t::Real)
    Łf, Łf_p, Ð_Łf_p = L.memŁf, L.memŁvf, L.memÐvf
    p = L.p[τ(t)]
    
    @! Łf = Ł(f)
    @! Łf_p = p * Łf
    @! Ð_Łf_p = Ð(Łf_p)
    @! v = -∇ᵢ' * L.MÐ' * Ð_Łf_p
end

function negδvelocityᴴ!((df_dt, dδf_dt, dδϕ_dt)::FieldTuple, L::CachedQuasiLenseFlow, (f, δf, δϕ)::FieldTuple, t::Real)
    
    p   = L.p[τ(t)]
    M⁻¹ = L.M⁻¹[τ(t)]
    
    # dδf/dt
    Łδf, Łδf_p, Ð_Łδf_p = L.memŁf, L.memŁvf, L.memÐvf
    @! Łδf     = Ł(δf)
    @! Łδf_p   = p * Łδf
    @! Ð_Łδf_p = Ð(Łδf_p)
    @! dδf_dt  = -∇ᵢ' * L.MÐ' * Ð_Łδf_p
    
    # df/dt
    Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf,  L.memŁvf
    @! Ðf     = Ð(f)
    @! Ð∇f    = L.MÐ * ∇ᵢ * Ðf
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

# can swap integration points without recaching, although not arbitrarily change them
_getindex(L::CachedQuasiLenseFlow{N,t₀,t₁,Φ,ŁΦ,ÐΦ,ŁF,ÐF,T}, ::→{t₀,t₁}) where {N,t₀,t₁,Φ,ŁΦ,ÐΦ,ŁF,ÐF,T} = L
_getindex(L::CachedQuasiLenseFlow{N,t₁,t₀,Φ,ŁΦ,ÐΦ,ŁF,ÐF,T}, ::→{t₀,t₁}) where {N,t₀,t₁,Φ,ŁΦ,ÐΦ,ŁF,ÐF,T} = CachedQuasiLenseFlow{N,t₀,t₁,Φ,ŁΦ,ÐΦ,ŁF,ÐF,T}(fieldvalues(L)...)

# adapting storage
adapt_structure(storage, Lϕ::QuasiLenseFlow{I,t₀,t₁}) where {I<:ODESolver,t₀,t₁} = QuasiLenseFlow{I,t₀,t₁}(adapt(storage,Lϕ.ϕ))
adapt_structure(storage, Lϕ::CachedQuasiLenseFlow{N,t₀,t₁}) where {N,t₀,t₁} = cache(QuasiLenseFlow{RK4Solver{N},t₀,t₁}(adapt(storage,Lϕ.ϕ[])), adapt(storage,Lϕ.memŁf))

# # ud_grading QuasiLenseFlow ud_grades the ϕ map
# ud_grade(L::QuasiLenseFlow{I,t₀,t₁}, args...; kwargs...) where {I,t₀,t₁} = QuasiLenseFlow{I,t₀,t₁}(ud_grade(L.ϕ,args...;kwargs...))
# ud_grade(L::CachedQuasiLenseFlow, args...; kwargs...)  = cache(ud_grade(L.L,args...;kwargs...))
