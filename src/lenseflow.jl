using Sundials
using ODE

export LenseFlowOp, ode45, Euler, CVODE

abstract ODESolver
abstract Euler{nsteps} <: ODESolver 
abstract CVODE{reltol,abstol} <: ODESolver  # doesn't work very well...
abstract ode45{reltol,abstol} <: ODESolver 


immutable LenseFlowOp{F<:Field,I<:ODESolver} <: LinearFieldOp
    ϕ::F
    d::Vector{F}
    Jac::Matrix{F}
end

function LenseFlowOp{I<:ODESolver}(ϕ::Field, ::Type{I}=ode45{1e-3,1e-3})
    d = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlowOp{typeof(ϕ),I}(ϕ, d, ∇*d')
end

velocity(L::LenseFlowOp, f::Field, t::Real) = L.d'*inv(eye(2)+t*L.Jac)*Map(∇*f)


function lense_flow{F,nsteps}(L::LenseFlowOp{F,Euler{nsteps}}, f::Field, ts)
    Δt = 1/nsteps * (ts[2]-ts[1])
    t = ts[1]
    f = Map(f)
    for i=1:nsteps
        f = f + Δt * velocity(L,f,t)
        t += Δt
    end
    f
end

function lense_flow{F,reltol,abstol}(L::LenseFlowOp{F,CVODE{reltol,abstol}}, f::Field, ts)
    Sundials.cvode((t,y,ẏ)->(ẏ .= velocity(L,y[~f],t)[:]), f[:], ts; reltol=reltol, abstol=abstol)[2,:][~f]
end

function lense_flow{F,reltol,abstol}(L::LenseFlowOp{F,ode45{reltol,abstol}}, f::Field, ts)
    ODE.ode45((t,y)->velocity(L,y[~f],t)[:], f[:], ts; reltol=reltol, abstol=abstol, points=:specified)[2][end][~f]
end

*(L::LenseFlowOp, f::Field) = lense_flow(L,f,[0.,1])
\(L::LenseFlowOp, f::Field) = lense_flow(L,f,[1.,0])
