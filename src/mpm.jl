
# interface with MPMEstimate.jl

using .MPMEstimate: AbstractMPMProblem
import .MPMEstimate: ∇θ_logLike, sample_x_z, ẑ_at_θ, mpm

export CMBLensingMPMProblem

struct CMBLensingMPMProblem{DS<:DataSet} <: AbstractMPMProblem
    ds :: DS
    parameterization
    MAP_joint_kwargs
end

function CMBLensingMPMProblem(ds; parameterization=0, MAP_joint_kwargs=(;))
    CMBLensingMPMProblem(ds, parameterization, MAP_joint_kwargs)
end


function ∇θ_logLike(prob::CMBLensingMPMProblem, d, θ, z) 
    @unpack ds, parameterization = prob
    @set! ds.d = d
    if parameterization == 0
        gradient(θ -> lnP(parameterization, z..., θ, ds), θ)[1]
    elseif parameterization == 1
        (f, ϕ) = z
        f̃ = ds.L(ϕ)*f
        gradient(θ -> lnP(parameterization, f̃, ϕ, θ, ds), θ)[1]
    elseif parameterization == :mix
        f°, ϕ° = mix(z..., θ, ds)
        gradient(θ -> lnP(parameterization, f°, ϕ°, θ, ds), θ)[1]
    end
end

function sample_x_z(prob::CMBLensingMPMProblem, rng::AbstractRNG, θ) 
    @unpack d,f,ϕ = resimulate(prob.ds(θ); rng)
    (x=d, z=(f,ϕ))
end

function sample_x_z(prob::CMBLensingMPMProblem{<:NoLensingDataSet}, rng::AbstractRNG, θ) 
    @unpack d,f = resimulate(prob.ds(θ); rng)
    (x=d, z=(f,))
end

function ẑ_at_θ(prob::CMBLensingMPMProblem, d, θ, (f₀,ϕ₀); ∇z_logLike_atol=nothing)
    @unpack ds = prob
    MAP_joint(θ, @set(ds.d=d); fstart=f₀, ϕstart=ϕ₀, prob.MAP_joint_kwargs...)[1:2]
end

function ẑ_at_θ(prob::CMBLensingMPMProblem{<:NoLensingDataSet}, d, θ, (f₀,); ∇z_logLike_atol=nothing)
    @unpack ds = prob
    (argmaxf_lnP(I, θ, @set(ds.d=d); fstart=f₀, prob.MAP_joint_kwargs...),)
end

function mpm(prob::CMBLensingMPMProblem, θ₀; kwargs...)
    mpm(prob, prob.ds.d, θ₀; kwargs...)
end

function mpm(ds::DataSet, θ₀; parameterization=0, MAP_joint_kwargs=(;), kwargs...)
    mpm(CMBLensingMPMProblem(ds; parameterization, MAP_joint_kwargs), θ₀; kwargs...)
end