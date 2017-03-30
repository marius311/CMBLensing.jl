
export LenseFlowOp, LenseBasis, δlenseflow

abstract type ODESolver end

struct LenseFlowOp{I<:ODESolver,F<:Field} <: LenseOp
    ϕ::F
    ∇ϕ::SVector{2,F}
    Jϕ::SMatrix{2,2,F,4}
end

function LenseFlowOp{I<:ODESolver}(ϕ::Field{<:Pix,<:S0,<:Basis}, ::Type{I}=ode4{4})
    ∇ϕ = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlowOp{I,typeof(ϕ)}(ϕ, ∇ϕ, ∇⨳(∇ϕ'))
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


# the LenseFlow algorithm 
velocity(L::LenseFlowOp, f::Field, t::Real) = @⨳ L.∇ϕ' ⨳ inv(𝕀 + t*L.Jϕ) ⨳ $Ł(∇*f)

function lenseflow(L::LenseFlowOp{I}, f::F, ts) where {I,F<:Field}
    ys = run_ode(I)((t,y)->F(velocity(L,y,t)), f, ts; kwargs(I)...)
    dbg(I)[1] && info("lenseflow: ode45 took $(length(ys[2])) steps")
    dbg(I)[2] ? ys : ys[2][end]::F # <-- ODE.jl not type stable
end

# function lenseflow(L::LenseFlowOp{ode4{N}}, f::F, ts) where {N,F<:Field}
#     ODE.ode4((t,y)->F(velocity(L,y,t)), f, Float32.(linspace(ts...,N)))[2][end]::F
# end


*(L::LenseFlowOp, f::Field) = lenseflow(L,Ð(f),Float32[0,1])
\(L::LenseFlowOp, f::Field) = lenseflow(L,Ð(f),Float32[1,0])


## transpose lenseflow

*(J::δf̃_δfϕᵀ{<:LenseFlowOp}, δPδf̃::Field) = δf̃_δfϕᵀ(J.L,Ł(J.f),Ł(δPδf̃))

""" Compute [(δf̃(f)/δf)ᵀ * δP/δf̃, (δf̃(f)/δϕ)ᵀ * δP/δf̃] """
function δf̃_δfϕᵀ(L::LenseFlowOp{I,F}, f::Ff, δPδf̃::Fδf̃, δLδϕ::Fδϕ=Ð(zero(F))) where {I,F,Ff<:Field,Fδf̃<:Field,Fδϕ<:Field}
    
    # first get lensed field at t=1
    f̃ = Ff(L*f)
    # this specifies the basis in which we do the ODE, which is taken to be the
    # basis in which the fields come into this function
    Fy = Field3Tuple{Ff,Fδf̃,Fδϕ}
    # now run negative transpose perturbed lense flow backwards
    ys = run_ode(I)(
        (t,y)->Fy(FieldTuple(δvelocityᵀ(L,y...,t)...)), 
        FieldTuple(f̃,δPδf̃,δLδϕ), Float32[1,0]; 
        kwargs(I)...)
        
    dbg(I)[1] && info("δf̃_δfϕᵀ: ode45 took $(length(ys[2])) steps")
    dbg(I)[2] ? ys : ys[2][end][2:3] :: Tuple{Fδf̃,Fδϕ}
end

# function dLdf_dfdf̃ϕ{reltol,abstol,maxsteps,F}(L::LenseFlowOp{ode45{reltol,abstol,maxsteps},F}, f::Field, dLdf::Field, δPδϕ::F=zero(F); debug=false)
#     
#     # now run negative transpose perturbed lense flow forwards
#     ys = ODE.ode45(
#         (t,y)->δvelocityᵀ(L,y[~(f,dLdf,δPδϕ)]...,t)[:], 
#         [f,dLdf,δPδϕ][:], [0.,1]; 
#         reltol=reltol, abstol=abstol, points=:all, minstep=1/maxsteps)
#         
#     if debug
#         info("dLdf_dfdf̃ϕ: ode45 took $(length(ys)) steps")
#         ys
#     else:
#         ys[2][end][~(f,dLdf,δPδϕ)][2:3]
#     end
#     
# end

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
