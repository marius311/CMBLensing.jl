
# interface with MuseInference.jl

using .MuseInference: AbstractMuseProblem, MuseResult
import .MuseInference: ∇θ_logLike, sample_x_z, ẑ_at_θ, muse!, standardizeθ

export CMBLensingMuseProblem

struct CMBLensingMuseProblem{DS<:DataSet,DS_SIM<:DataSet} <: AbstractMuseProblem
    ds :: DS
    ds_for_sims :: DS_SIM
    parameterization
    MAP_joint_kwargs
    θ_fixed
    x
    latent_vars
end

function CMBLensingMuseProblem(ds, ds_for_sims=ds; parameterization=0, MAP_joint_kwargs=(;), θ_fixed=(;), latent_vars=nothing)
    CMBLensingMuseProblem(ds, ds_for_sims, parameterization, MAP_joint_kwargs, θ_fixed, ds.d, latent_vars)
end

mergeθ(prob::CMBLensingMuseProblem, θ) = isempty(prob.θ_fixed) ? θ : (;prob.θ_fixed..., θ...)

function standardizeθ(prob, θ)
    θ isa Union{NamedTuple,ComponentVector} || error("θ should be a NamedTuple or ComponentVector")
    1f0 * ComponentVector(θ) # ensure component vector and float
end

function ∇θ_logLike(prob::CMBLensingMuseProblem, d, z, θ) 
    @unpack ds, parameterization = prob
    @set! ds.d = d
    if parameterization == 0
        gradient(θ -> logpdf(ds; z..., θ = mergeθ(prob, θ)), θ)[1]
    elseif parameterization == :mix
        z° = mix(ds; z..., θ = mergeθ(prob, θ))
        gradient(θ -> logpdf(Mixed(ds); z°..., θ = mergeθ(prob, θ)), θ)[1]
    else
        error("parameterization should be 0 or :mix")
    end
end

function sample_x_z(prob::CMBLensingMuseProblem, rng::AbstractRNG, θ) 
    sim = simulate(rng, prob.ds_for_sims, θ = mergeθ(prob, θ))
    if prob.latent_vars == nothing
        # this is a guess which might not work for everything necessarily
        z = FieldTuple(delete(sim, (:f̃, :d, :μ))) 
    else
        z = FieldTuple(select(sim, prob.latent_vars))
    end
    x = sim.d
    (;x, z)
end

function ẑ_at_θ(prob::CMBLensingMuseProblem, d, zguess, θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    Ωstart = delete(NamedTuple(zguess), :f)
    MAP = MAP_joint(mergeθ(prob, θ), @set(ds.d=d), Ωstart; fstart=zguess.f, prob.MAP_joint_kwargs...)
    FieldTuple(;delete(MAP, :history)...), MAP.history
end

function ẑ_at_θ(prob::CMBLensingMuseProblem{<:NoLensingDataSet}, d, (f₀,), θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    FieldTuple(f=argmaxf_logpdf(I, mergeθ(prob, θ), @set(ds.d=d); fstart=f₀, prob.MAP_joint_kwargs...)), nothing
end

function muse!(result::MuseResult, ds::DataSet, θ₀=nothing; parameterization=0, MAP_joint_kwargs=(;), kwargs...)
    muse!(result, CMBLensingMuseProblem(ds; parameterization, MAP_joint_kwargs), θ₀; kwargs...)
end