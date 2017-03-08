using Sundials
using ODE

export LenseFlowOp, LenseBasis, δlenseflow

abstract ODESolver
abstract ode45{reltol,abstol,maxsteps} <: ODESolver 
abstract ode4{nsteps} <: ODESolver 


immutable LenseFlowOp{I<:ODESolver,F<:Field} <: LinearOp
    ϕ::F
    ∇ϕ::Vector{F}
    Jϕ::Matrix{F}
end

function LenseFlowOp{I<:ODESolver}(ϕ::Field, ::Type{I}=ode45{1e-3,1e-3,100})
    ∇ϕ = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlowOp{I,typeof(ϕ)}(ϕ, ∇ϕ, ∇*∇ϕ')
end

# the LenseFlow algorithm 

velocity(L::LenseFlowOp, f::Field, t::Real) = L.∇ϕ'*inv(eye(2)+t*L.Jϕ)*Ł(∇*f)

function lenseflow{reltol,abstol,maxsteps}(L::LenseFlowOp{ode45{reltol,abstol,maxsteps}}, f::Field, ts)
    ys = ODE.ode45(
        (t,y)->velocity(L,y[~f],t)[:], f[:], ts;
        reltol=reltol, abstol=abstol, minstep=1/maxsteps, points=:all)[2]
    info("lenseflow: ode45 took $(length(ys)) steps")
    ys[end][~f]
end

function lenseflow{nsteps}(L::LenseFlowOp{ode4{nsteps}}, f::Field, ts)
    ODE.ode4((t,y)->velocity(L,y[~f],t)[:], f[:], linspace(ts...,nsteps))[2][end][~f]
end

*(L::LenseFlowOp, f::Field) = lenseflow(L,LenseBasis(f),[0.,1])
\(L::LenseFlowOp, f::Field) = lenseflow(L,LenseBasis(f),[1.,0])


# transpose lenseflow

function dLdf̃_df̃dfϕ{reltol,abstol,F}(L::LenseFlowOp{ode45{reltol,abstol},F}, f::Field, dLdf̃::Field, dLdϕ::F=zero(F))
    
    # first get lensed field at t=1
    f̃ = lenseflow(L,f,[0.,1])
    
    # now run negative transpose perturbed lense flow backwards
    ys = ODE.ode45(
        (t,y)->δvelocityᵀ(L,y[~(f̃,dLdf̃,dLdϕ)]...,t)[:], 
        [f̃,dLdf̃,dLdϕ][:], [1.,0]; 
        reltol=reltol, abstol=abstol, points=:all, minstep=1e-1)[2]
        
    info("dLdf̃_df̃dfϕ: ode45 took $(length(ys)) steps")
    
    ys[end][~(f̃,dLdf̃,dLdϕ)][2:3]
    
end

function dLdf_dfdf̃ϕ{reltol,abstol,F}(L::LenseFlowOp{ode45{reltol,abstol},F}, f::Field, dLdf::Field, dLdϕ::F=zero(F))
    
    # now run negative transpose perturbed lense flow forwards
    ys = ODE.ode45(
        (t,y)->δvelocityᵀ(L,y[~(f,dLdf,dLdϕ)]...,t)[:], 
        [f,dLdf,dLdϕ][:], [0.,1]; 
        reltol=reltol*10, abstol=abstol*10, points=:all, minstep=1e-1)[2]
        
    info("dLdf_dfdf̃ϕ: ode45 took $(length(ys)) steps")
    
    ys[end][~(f,dLdf,dLdϕ)][2:3]
    
end


function δvelocityᵀ(L::LenseFlowOp, f::Field, dLdf̃::Field, dLdϕ::Field, t::Real)
    
    iM          = inv(eye(2)+t*L.Jϕ) |> Ł
    ∇f          = ∇*f                |> Ł
    iM_dLdf̃ᵀ_∇f = iM*(dLdf̃'*∇f)      |> Ł
    iM_∇ϕ       = iM*L.∇ϕ            |> Ł
    
    f′    = Ł(L.∇ϕ'*(iM*∇f))
    dLdf̃′ = Ł(∇ᵀ*(dLdf̃'*iM_∇ϕ))
    dLdϕ′ = Ł(∇ᵀ*iM_dLdf̃ᵀ_∇f + t*(∇ᵀ*((∇ᵀ*(iM_∇ϕ*iM_dLdf̃ᵀ_∇f'))')))
    
    [f′, dLdf̃′, dLdϕ′]

end
