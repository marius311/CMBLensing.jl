using Sundials
using ODE

export LenseFlowOp, ode45, Euler, CVODE, LenseBasis, δlense_flow

abstract ODESolver
abstract Euler{nsteps} <: ODESolver 
abstract CVODE{reltol,abstol} <: ODESolver  # doesn't work very well...
abstract ode45{reltol,abstol} <: ODESolver 

immutable LenseFlowOp{I<:ODESolver,F<:Field} <: LinearOp
    ϕ::F
    ∇ϕ::Vector{F}
    Jϕ::Matrix{F}
end

function LenseFlowOp{I<:ODESolver}(ϕ::Field, ::Type{I}=ode45{1e-3,1e-3})
    ∇ϕ = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlowOp{I,typeof(ϕ)}(ϕ, ∇ϕ, ∇*∇ϕ')
end

# For each Field type, LenseFlow needs to know the basis in which lensing is a
# remapping. E.g. for FlatS0 and FlatS2 this is Map and QUMap, respectively.
# Fields should implement their own LenseBasis(::Type{F}) to specify. 
LenseBasis{F<:Field}(f::F) = LenseBasis(F)(f)
LenseBasis{F<:Field}(::Type{F}) = error("""To lense a field of type $(typeof(f)), LenseBasis(f::$(typeof(f))) needs to be implemented.""")
LenseBasis{F<:Field}(x::AbstractArray{F}) = map(LenseBasis,x)


# the LenseFlow algorithm 
velocity(L::LenseFlowOp, f::Field, t::Real) = L.∇ϕ'*inv(eye(2)+t*L.Jϕ)*LenseBasis(∇*f)

# the velocity [df/dt dδf/dt] to propagate a small perturbation (δf, δϕ)
# (note: dδϕ/dt = 0 so its not returned)
function δvelocity(L::LenseFlowOp, f::Field, δf::Field, δϕ::Field, t::Real)
    
    ∇δϕ = ∇*δϕ                |> LenseBasis
    Jδϕ = ∇*∇δϕ'              |> LenseBasis
    M   = inv(eye(2)+t*L.Jϕ)  |> LenseBasis
    ∇f  = ∇*f                 |> LenseBasis
    ∇δf = ∇*δf                |> LenseBasis
    
    ḟ = L.∇ϕ'*M*∇f
    δḟ = (∇δϕ'-L.∇ϕ'*M*t*Jδϕ)*M*∇f + L.∇ϕ'*M*∇δf
    
    (ḟ,δḟ)
    
end


function δlense_flow{reltol,abstol}(L::LenseFlowOp{ode45{reltol,abstol}}, f0::Field, δf0::Field, δϕ0::Field, ts)
    
    y0 = vcat(f0[:],δf0[:])
    n = length(y0)÷2
    
    y = ODE.ode45((t,y) -> vcat(map(f->f[:],δvelocity(L,y[1:n][~f0],y[n+1:end][~δf0],δϕ0,t))...), y0, ts; reltol=reltol, abstol=abstol, points=:specified)[2][end]
    
    (y[1:n][~f0], y[n+1:end][~δf0])
end




# three different ODE solvers... 

function lense_flow{nsteps}(L::LenseFlowOp{Euler{nsteps}}, f::Field, ts)
    Δt = 1/nsteps * (ts[2]-ts[1])
    t = ts[1]
    for i=1:nsteps
        f = f + Δt * velocity(L,f,t)
        t += Δt
    end
    f
end

function lense_flow{reltol,abstol}(L::LenseFlowOp{CVODE{reltol,abstol}}, f::Field, ts)
    Sundials.cvode((t,y,ẏ)->(ẏ .= velocity(L,y[~f],t)[:]), f[:], ts; reltol=reltol, abstol=abstol)[2,:][~f]
end

function lense_flow{reltol,abstol}(L::LenseFlowOp{ode45{reltol,abstol}}, f::Field, ts)
    ODE.ode45((t,y)->velocity(L,y[~f],t)[:], f[:], ts; reltol=reltol, abstol=abstol, points=:specified)[2][end][~f]
end

*(L::LenseFlowOp, f::Field) = lense_flow(L,LenseBasis(f),[0.,1])
\(L::LenseFlowOp, f::Field) = lense_flow(L,LenseBasis(f),[1.,0])
