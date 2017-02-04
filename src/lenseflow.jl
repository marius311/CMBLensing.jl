
immutable LenseFlowOp{P,F<:Field} <: LinearFieldOp
    ϕ::Field{P,S0}
    steps::Int
    d::Vector{F}
    Jac::Matrix{F}
end

LenseFlowOp(ϕ::Field, steps::Int) = (d = ∇*ϕ; LenseFlowOp(ϕ, steps, d, ∇*d'))
    
function lense_flow(L::LenseFlowOp, f::Field, forward=true)
    Δt = 1/L.steps * (forward ? 1 : -1)
    t = forward ? 0 : 1
    for i=1:L.steps
        f = f + Δt * d'*inv(1+t*L.Jac)*(∇*f)
        t += Δt
    end
    f
end

*(L::LenseFlowOp, f::Field) = lense_flow(L,f,true)
\(L::LenseFlowOp, f::Field) = lense_flow(L,f,false)
