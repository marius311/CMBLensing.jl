
# interface with MPMEstimate.jl

using .MPMEstimate: AbstractMPMProblem
import .MPMEstimate: ∇θ_logP, sample_x_z, ẑ_at_θ, mpm

export CMBLensingMPMProblem

struct CMBLensingMPMProblem <: AbstractMPMProblem
    ds :: DataSet
    parameterization
    MAP_joint_kwargs
end

function CMBLensingMPMProblem(ds; parameterization=0, MAP_joint_kwargs=(;))
    CMBLensingMPMProblem(ds, parameterization, MAP_joint_kwargs)
end


function ∇θ_logP(prob::CMBLensingMPMProblem, d, θ, (f,ϕ)) 
    @unpack ds = prob
    gradient(θ -> lnP(prob.parameterization, f, ϕ, θ, @set(ds.d=d)), θ)[1]
end

function sample_x_z(prob::CMBLensingMPMProblem, rng::AbstractRNG, θ) 
    @unpack ds = prob
    @unpack d,f,ϕ = resimulate(ds(θ); rng)
    (x=d, z=(f,ϕ))
end

function ẑ_at_θ(prob::CMBLensingMPMProblem, d, θ, (f₀,ϕ₀))
    @unpack ds = prob
    (fJ,ϕJ) = MAP_joint(θ, @set(ds.d=d); fstart=f₀, ϕstart=ϕ₀, prob.MAP_joint_kwargs...)
    (fJ,ϕJ)
end

function mpm(prob::CMBLensingMPMProblem, θ₀; kwargs...)
    mpm(prob, prob.ds.d, θ₀; kwargs...)
end

function mpm(ds::DataSet, θ₀; parameterization=0, MAP_joint_kwargs=(;), kwargs...)
    mpm(CMBLensingMPMProblem(ds; parameterization, MAP_joint_kwargs), θ₀; kwargs...)
end