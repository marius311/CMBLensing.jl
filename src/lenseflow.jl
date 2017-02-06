
immutable LenseFlowOp{F<:Field} <: LinearFieldOp
    ϕ::F
    steps::Int
    d::Vector{F}
    Jac::Matrix{F}
end

LenseFlowOp{F<:Field}(ϕ::F, steps::Int) = (d = ∇*ϕ; LenseFlowOp{F}(ϕ, steps, d, ∇*d'))
    
function lense_flow(L::LenseFlowOp, f::Field, forward=true)
    Δt = 1/L.steps * (forward ? 1 : -1)
    t = forward ? 0 : 1
    f = Map(f)
    for i=1:L.steps
        f = f + Δt * velocity(L,f,t)
        t += Δt
    end
    f
end

velocity(L::LenseFlowOp, f::Field, t::Real) = L.d'*inv(eye(2)+t*L.Jac)*Map(∇*f)

*(L::LenseFlowOp, f::Field) = lense_flow(L,f,true)
\(L::LenseFlowOp, f::Field) = lense_flow(L,f,false)
