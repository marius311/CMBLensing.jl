using Sundials
using ODE

export LenseFlowOp, ode45, Euler, CVODE, LenseBasis, δlenseflow

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

velocity(L::LenseFlowOp, f::Field, t::Real) = L.∇ϕ'*inv(eye(2)+t*L.Jϕ)*Ł(∇*f)

function lenseflow{nsteps}(L::LenseFlowOp{Euler{nsteps}}, f::Field, ts)
    Δt = 1/nsteps * (ts[2]-ts[1])
    t = ts[1]
    for i=1:nsteps
        f = f + Δt * velocity(L,f,t)
        t += Δt
    end
    f
end

function lenseflow{reltol,abstol}(L::LenseFlowOp{CVODE{reltol,abstol}}, f::Field, ts)
    Sundials.cvode((t,y,ẏ)->(ẏ .= velocity(L,y[~f],t)[:]), f[:], ts; reltol=reltol, abstol=abstol)[2,:][~f]
end

function lenseflow{reltol,abstol}(L::LenseFlowOp{ode45{reltol,abstol}}, f::Field, ts)
    ODE.ode45((t,y)->velocity(L,y[~f],t)[:], f[:], ts; reltol=reltol, abstol=abstol, points=:specified)[2][end][~f]
end

*(L::LenseFlowOp, f::Field) = lenseflow(L,LenseBasis(f),[0.,1])
\(L::LenseFlowOp, f::Field) = lenseflow(L,LenseBasis(f),[1.,0])


# transpose lenseflow

function dLdf̃_df̃dfϕ{reltol,abstol,F}(L::LenseFlowOp{ode45{reltol,abstol},F}, f::Field, dLdf̃::Field, dLdϕ::F=zero(F))
    
    # first get lensed field at t=1
    f̃ = lenseflow(L,f,[0.,1])
    
    # now run transpose perturbed lense flow backwards
    y = ODE.ode45(
        (t,y)->δvelocityᵀ(L,y[~(f̃,dLdf̃,dLdϕ)]...,t)[:], 
        [f̃,dLdf̃,dLdϕ][:], [1.,0]; 
        reltol=reltol, abstol=abstol, points=:specified)[2][end]
    
    y[~(f̃,dLdf̃,dLdϕ)][2:3]
    
end

function δvelocityᵀ(L::LenseFlowOp, f::Field, dLdf̃::Field, dLdϕ::Field, t::Real)
    
    iM  = inv(eye(2)+t*L.Jϕ) |> Ł
    ∇f  = ∇*f                |> Ł
    iM_∇f = iM*∇f            |> Ł
    iM_∇ϕ = iM*L.∇ϕ          |> Ł
    
    f′ = L.∇ϕ'*iM_∇f
    dLdf̃′ = ∇ᵀ*(dLdf̃*iM_∇ϕ)
    dLdϕ′ = ∇ᵀ*(dLdf̃*iM_∇f) + t*(∇ᵀ*((∇ᵀ*(iM_∇ϕ*iM_∇f'))'))
    
    [f′, dLdf̃′, dLdϕ′]

end
