
# interface with MuseInference.jl

using .MuseInference: AbstractMuseProblem, MuseResult
import .MuseInference: ∇θ_logLike, sample_x_z, ẑ_at_θ, muse!

export CMBLensingMuseProblem

struct CMBLensingMuseProblem{DS<:DataSet} <: AbstractMuseProblem
    ds :: DS
    parameterization
    MAP_joint_kwargs
end

function CMBLensingMuseProblem(ds; parameterization=0, MAP_joint_kwargs=(;))
    CMBLensingMuseProblem(ds, parameterization, MAP_joint_kwargs)
end

function ∇θ_logLike(prob::CMBLensingMuseProblem, d, z, θ) 
    @unpack ds, parameterization = prob
    @set! ds.d = d
    if parameterization == 0
        (f, ϕ) = z
        gradient(θ -> logpdf(ds; f, ϕ, θ), θ)[1]
    elseif parameterization == 1
        (f, ϕ) = z
        f̃ = ds.L(ϕ)*f
        gradient(θ -> logpdf(ds; f̃, ϕ, θ), θ)[1]
    elseif parameterization == :mix
        (f, ϕ) = z
        (f°, ϕ°) = mix(ds; f, ϕ, θ)
        gradient(θ -> logpdf(Mixed(ds); f°, ϕ°, θ), θ)[1]
    end
end

function sample_x_z(prob::CMBLensingMuseProblem, rng::AbstractRNG, θ) 
    @unpack d,f,ϕ = simulate(rng, prob.ds(θ))
    (x=d, z=FieldTuple(;f,ϕ))
end

function sample_x_z(prob::CMBLensingMuseProblem{<:NoLensingDataSet}, rng::AbstractRNG, θ) 
    @unpack d,f = simulate(rng, prob.ds(θ))
    (x=d, z=FieldTuple(;f))
end

function ẑ_at_θ(prob::CMBLensingMuseProblem, d, (f₀,ϕ₀), θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    (f, ϕ, history) = MAP_joint(θ, @set(ds.d=d); fstart=f₀, ϕstart=ϕ₀, prob.MAP_joint_kwargs...)
    FieldTuple(;f, ϕ), history
end

function ẑ_at_θ(prob::CMBLensingMuseProblem{<:NoLensingDataSet}, d, (f₀,), θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    FieldTuple(f=argmaxf_logpdf(I, θ, @set(ds.d=d); fstart=f₀, prob.MAP_joint_kwargs...))
end

function muse!(result::MuseResult, prob::CMBLensingMuseProblem, θ₀=nothing; kwargs...)
    muse!(result, prob, prob.ds.d, θ₀; kwargs...)
end

function muse!(result::MuseResult, ds::DataSet, θ₀=nothing; parameterization=0, MAP_joint_kwargs=(;), kwargs...)
    muse!(result, CMBLensingMuseProblem(ds; parameterization, MAP_joint_kwargs), θ₀; kwargs...)
end