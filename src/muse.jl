
# interface with MuseEstimate.jl

using .MuseEstimate: AbstractMuseProblem, MuseResult
import .MuseEstimate: ∇θ_logLike, sample_x_z, ẑ_at_θ, muse!

export CMBLensingMuseProblem

struct CMBLensingMuseProblem{DS<:DataSet} <: AbstractMuseProblem
    ds :: DS
    parameterization
    MAP_joint_kwargs
end

function CMBLensingMuseProblem(ds; parameterization=0, MAP_joint_kwargs=(;))
    CMBLensingMuseProblem(ds, parameterization, MAP_joint_kwargs)
end


function ∇θ_logLike(prob::CMBLensingMuseProblem, d, θ, z) 
    @unpack ds, parameterization = prob
    @set! ds.d = d
    if parameterization == 0
        gradient(θ -> logpdf(ds; z..., θ), θ)[1]
    elseif parameterization == 1
        (f, ϕ) = z
        f̃ = ds.L(ϕ)*f
        gradient(θ -> logpdf(ds; f̃, ϕ, θ), θ)[1]
    elseif parameterization == :mix
        f°, ϕ° = mix(ds; z..., θ)
        gradient(θ -> logpdf(Mixed(ds); f°, ϕ°, θ), θ)[1]
    end
end

function sample_x_z(prob::CMBLensingMuseProblem, rng::AbstractRNG, θ) 
    @unpack d,f,ϕ = simulate(rng, prob.ds(θ))
    (x=d, z=(;f,ϕ))
end

function sample_x_z(prob::CMBLensingMuseProblem{<:NoLensingDataSet}, rng::AbstractRNG, θ) 
    @unpack d,f = simulate(rng, prob.ds(θ))
    (x=d, z=(;f))
end

function ẑ_at_θ(prob::CMBLensingMuseProblem, d, θ, (f₀,ϕ₀); ∇z_logLike_atol=nothing)
    @unpack ds = prob
    (f, ϕ, history) = MAP_joint(θ, @set(ds.d=d); fstart=f₀, ϕstart=ϕ₀, prob.MAP_joint_kwargs...)
    (;f, ϕ), history
end

function ẑ_at_θ(prob::CMBLensingMuseProblem{<:NoLensingDataSet}, d, θ, (f₀,); ∇z_logLike_atol=nothing)
    @unpack ds = prob
    (argmaxf_logpdf(I, θ, @set(ds.d=d); fstart=f₀, prob.MAP_joint_kwargs...),)
end

function muse!(result::MuseResult, prob::CMBLensingMuseProblem, θ₀=nothing; kwargs...)
    muse!(result, prob, prob.ds.d, θ₀; kwargs...)
end

function muse!(result::MuseResult, ds::DataSet, θ₀=nothing; parameterization=0, MAP_joint_kwargs=(;), kwargs...)
    muse!(result, CMBLensingMuseProblem(ds; parameterization, MAP_joint_kwargs), θ₀; kwargs...)
end