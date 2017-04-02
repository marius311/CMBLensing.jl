
export LenseFlowOp, LenseBasis, δlenseflow

abstract type ODESolver end

struct LenseFlowOp{I<:ODESolver,t1,t2,F<:Field} <: LenseOp
    ϕ::F
    ∇ϕ::SVector{2,F}
    Jϕ::SMatrix{2,2,F,4}
end


@∷ function LenseFlowOp(ϕ::Field{∷,<:S0}, ::Type{I}=ode4{10}, t1=0., t2=1.) where {I<:ODESolver}
    ∇ϕ = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlowOp{I,t1,t2,typeof(ϕ)}(ϕ, ∇ϕ, ∇⨳(∇ϕ'))
end

# the ODE solvers
abstract type ode45{reltol,abstol,maxsteps,debug} <: ODESolver  end
abstract type ode4{nsteps} <: ODESolver  end
kwargs{ϵr,ϵa,N,dbg}(::Type{ode45{ϵr,ϵa,N,dbg}}) = Dict(:norm=>pixstd, :reltol=>ϵr, :abstol=>ϵa, :minstep=>1/N, :points=>((dbg[1] || dbg[2]) ? :all : :specified))
kwargs(::Type{<:ode4}) = Dict()
run_ode(::Type{<:ode45}) = ODE.ode45
run_ode(::Type{<:ode4}) = ODE.ode4
dbg(::Type{ode45{ϵr,ϵa,N,d}}) where {ϵr,ϵa,N,d} = d
dbg(::Type{<:ode4}) = (false,false)
tts(::Type{ode4{N}},ts) where {N} = linspace(ts...,N)
tts(::Type{<:ode45}) = ts


# the LenseFlow algorithm 
velocity(L::LenseFlowOp, f::Field, t::Real) = @⨳ L.∇ϕ' ⨳ inv(𝕀 + t*L.Jϕ) ⨳ $Ł(∇*f)

function lenseflow(L::LenseFlowOp{I}, f::F, ts) where {I,F<:Field}
    ys = run_ode(I)((t,y)->F(velocity(L,y,t)), f, tts(I,ts); kwargs(I)...)
    dbg(I)[1] && info("lenseflow: ode45 took $(length(ys[2])) steps")
    dbg(I)[2] ? ys : ys[2][end]::F # <-- ODE.jl not type stable
end


@∷ _getindex(L::LenseFlowOp{I,∷,∷,F}, ::→{t1,t2}) where {I,t1,t2,F} = LenseFlowOp{I,t1,t2,F}(L.ϕ,L.∇ϕ,L.Jϕ)
@∷ *(L::LenseFlowOp{∷,t1,t2}, f::Field) where {t1,t2} = lenseflow(L,Ð(f),Float32[t1,t2])
@∷ \(L::LenseFlowOp{∷,t1,t2}, f::Field) where {t1,t2} = lenseflow(L,Ð(f),Float32[t2,t1])


## transpose lenseflow

*(δP_δfₛ::Field, J::δfₛ_δfₜϕ{s,t,<:LenseFlowOp}) where {s,t} = δfₛ_δfₜϕ(J.L,Ł(J.fₛ),Ł(δP_δfₛ),s,t)

""" Compute [(δf̃(f)/δf)ᵀ * δP/δf̃, (δf̃(f)/δϕ)ᵀ * δP/δf̃] """
@∷ function δfₛ_δfₜϕ(L::LenseFlowOp{I,∷,∷,F}, fₛ::Ff, δP_δfₛ::Fδf, s::Real, t::Real, δP_δϕ::Fδϕ=Ð(zero(F))) where {I,F,Ff<:Field,Fδf<:Field,Fδϕ<:Field}
    
    # this specifies the basis in which we do the ODE, which is taken to be the
    # basis in which the fields come into this function
    Fy = Field3Tuple{Ff,Fδf,Fδϕ}
    # now run negative transpose perturbed lense flow backwards
    ys = run_ode(I)(
        (t,y)->Fy(FieldTuple(δvelocityᵀ(L,y...,t)...)), 
        FieldTuple(fₛ,δP_δfₛ,δP_δϕ), tts(I,Float32[s,t]); 
        kwargs(I)...)
        
    dbg(I)[1] && info("δf̃_δfϕᵀ: ode45 took $(length(ys[2])) steps")
    dbg(I)[2] ? ys : ys[2][end][2:3] :: Tuple{Fδf,Fδϕ}
end


function δvelocityᵀ(L::LenseFlowOp, f::Field, δPδf̃::Field, δPδϕ::Field, t::Real)
    
    ŁδPδf̃       = Ł(δPδf̃)
    iM          = Ł(inv(𝕀 + t*L.Jϕ))
    ∇f          = Ł(∇*f)
    iM_δPδf̃ᵀ_∇f = Ł(iM ⨳ (ŁδPδf̃'*∇f))
    iM_∇ϕ       = Ł(iM ⨳ L.∇ϕ)
    
    f′    = @⨳ L.∇ϕ' ⨳ iM ⨳ ∇f
    δPδf̃′ = @⨳ ∇ᵀ ⨳ $Ð(ŁδPδf̃*iM_∇ϕ)
    δPδϕ′ = @⨳ ∇ᵀ ⨳ $Ð(iM_δPδf̃ᵀ_∇f) + t*(∇ᵀ ⨳ ((∇ᵀ ⨳ $Ð(iM_∇ϕ ⨳ iM_δPδf̃ᵀ_∇f'))'))
    
    (f′, δPδf̃′, δPδϕ′)

end
