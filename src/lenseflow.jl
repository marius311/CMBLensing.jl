using ODE

export LenseFlowOp, LenseBasis, δlenseflow

abstract type ODESolver end
abstract type ode45{reltol,abstol,maxsteps,debug} <: ODESolver  end
abstract type ode4{nsteps} <: ODESolver  end


struct LenseFlowOp{I<:ODESolver,F<:Field} <: LenseOp
    ϕ::F
    ∇ϕ::SVector{2,F}
    Jϕ::SMatrix{2,2,F,4}
end

function LenseFlowOp{I<:ODESolver}(ϕ::Field{<:Pix,<:S0,<:Basis}, ::Type{I}=ode45{1e-3,1e-3,100,false})
    ∇ϕ = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlowOp{I,typeof(ϕ)}(ϕ, ∇ϕ, ∇⨳(∇ϕ'))
end

# the LenseFlow algorithm 
velocity(L::LenseFlowOp, f::Field, t::Real) = @⨳ L.∇ϕ' ⨳ inv(𝕀 + t*L.Jϕ) ⨳ $Ł(∇*f)

function lenseflow(L::LenseFlowOp{ode45{ϵr,ϵa,N,dbg}}, f::F, ts) where {ϵr,ϵa,N,dbg,F<:Field}
    ys = ODE.ode45(
        (t,y)->velocity(L,y[~f],t)[:], f[:], ts;
        reltol=ϵr, abstol=ϵa, minstep=1/N, points=:all)
    if dbg
        info("lenseflow: ode45 took $(length(ys[2])) steps")
        ys
    else
        ys[2][end][~f]::F
    end
end

function lenseflow(L::LenseFlowOp{ode4{N}}, f::Field, ts) where {N}
    ODE.ode4((t,y)->velocity(L,y[~f],t)[:], f[:], linspace(ts...,N))[2][end][~f]
end


*(L::LenseFlowOp, f::Field) = lenseflow(L,Ł(f),[0.,1])
\(L::LenseFlowOp, f::Field) = lenseflow(L,Ł(f),[1.,0])


# transpose lenseflow

*(J::δf̃_δfϕᵀ{<:LenseFlowOp}, δLδf̃::Field) = δf̃_δfϕᵀ(J.L,J.f,δLδf̃)

""" Compute [(δf̃(f)/δf)ᵀ * δP/δf̃, (δf̃(f)/δϕ)ᵀ * δP/δf̃] """
function δf̃_δfϕᵀ(L::LenseFlowOp{ode45{ϵr,ϵa,N},F}, f::Field, δLδf̃::Field, δLδϕ::F=zero(F)) where {ϵr,ϵa,N,F}
    
    # first get lensed field at t=1
    f̃ = L*f
    
    # now run negative transpose perturbed lense flow backwards
    ys = ODE.ode45(
        (t,y)->δvelocityᵀ(L,y[~(f̃,δLδf̃,δLδϕ)]...,t)[:], 
        [f̃,δLδf̃,δLδϕ][:], [1.,0]; 
        reltol=ϵr, abstol=ϵa, points=:all, minstep=1/N)[2]
        
    info("δf̃_δfϕᵀ: ode45 took $(length(ys)) steps")
    
    ys[end][~(f̃,δLδf̃,δLδϕ)][2:3]
    
end

# function dLdf_dfdf̃ϕ{reltol,abstol,maxsteps,F}(L::LenseFlowOp{ode45{reltol,abstol,maxsteps},F}, f::Field, dLdf::Field, dLdϕ::F=zero(F); debug=false)
#     
#     # now run negative transpose perturbed lense flow forwards
#     ys = ODE.ode45(
#         (t,y)->δvelocityᵀ(L,y[~(f,dLdf,dLdϕ)]...,t)[:], 
#         [f,dLdf,dLdϕ][:], [0.,1]; 
#         reltol=reltol, abstol=abstol, points=:all, minstep=1/maxsteps)
#         
#     if debug
#         info("dLdf_dfdf̃ϕ: ode45 took $(length(ys)) steps")
#         ys
#     else:
#         ys[2][end][~(f,dLdf,dLdϕ)][2:3]
#     end
#     
# end

function δvelocityᵀ(L::LenseFlowOp, f::Field, dLdf̃::Field, dLdϕ::Field, t::Real)
    ⨳
    iM          = Ł(inv(𝕀 + t*L.Jϕ))
    ∇f          = Ł(∇*f)
    iM_dLdf̃ᵀ_∇f = Ł(dLdf̃)' * (iM ⨳ ∇f)
    iM_∇ϕ       = Ł(iM ⨳ L.∇ϕ) 
    
    f′    = Ł(L.∇ϕ' ⨳ iM ⨳ ∇f)
    dLdf̃′ = Ł(∇ᵀ ⨳ Ð(dLdf̃'*iM_∇ϕ))
    dLdϕ′ = Ł(∇ᵀ ⨳ Ð(iM_dLdf̃ᵀ_∇f) + t*(∇ᵀ ⨳ ((∇ᵀ ⨳ Ð(iM_∇ϕ ⨳ iM_dLdf̃ᵀ_∇f'))')))
    
    [f′, dLdf̃′, dLdϕ′]

end
