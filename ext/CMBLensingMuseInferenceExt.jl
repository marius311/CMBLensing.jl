
module CMBLensingMuseInferenceExt

using CMBLensing

if isdefined(Base, :get_extension)
    using MuseInference
    using MuseInference: AbstractMuseProblem, MuseResult, Transformedθ, UnTransformedθ
    import AbstractDifferentiation
    import AbstractDifferentiation: pushforward_function, gradient, jacobian, hessian, value_and_gradient, value_and_gradient
else
    using ..MuseInference
    using ..MuseInference: AbstractMuseProblem, MuseResult, Transformedθ, UnTransformedθ
    import ..AbstractDifferentiation
    import ..AbstractDifferentiation: pushforward_function, gradient, jacobian, hessian, value_and_gradient, value_and_gradient
end

const AD = AbstractDifferentiation

using Base: @kwdef
using ComponentArrays
using NamedTupleTools
using Random
using Requires
using Setfield
using ForwardDiff

# we're going to make our own backend
struct ForwardDiffNoTagBackend{CS} <: AD.AbstractForwardMode end
chunk(::ForwardDiffNoTagBackend{Nothing}, x) = ForwardDiff.Chunk(x)
chunk(::ForwardDiffNoTagBackend{N}, _) where {N} = ForwardDiff.Chunk{N}()

function pushforward_function(ba::ForwardDiffNoTagBackend{CS}, f, xs...) where CS
    pushforward_function(AD.ForwardDiffBackend{CS}(), f, xs...)
end

function AD.gradient(ba::ForwardDiffNoTagBackend, f, x::AbstractArray)
    cfg = ForwardDiff.GradientConfig(nothing, x, chunk(ba, x))
    return (ForwardDiff.gradient(f, x, cfg),)
end

function AD.jacobian(ba::ForwardDiffNoTagBackend, f, x::AbstractArray)
    cfg = ForwardDiff.JacobianConfig(nothing, x, chunk(ba, x))
    return (ForwardDiff.jacobian(AD.asarray ∘ f, x, cfg),)
end

function AD.jacobian(ba::ForwardDiffNoTagBackend, f, x::R) where {R <: Number}
    T = typeof(ForwardDiff.Tag(nothing, R))
    return (ForwardDiff.extract_derivative(T, f(ForwardDiff.Dual{T}(x, one(x)))),)
end

function AD.hessian(ba::ForwardDiffNoTagBackend, f, x::AbstractArray)
    cfg = ForwardDiff.HessianConfig(nothing, x, chunk(ba, x))
    return (ForwardDiff.hessian(f, x, cfg),)
end

function AD.value_and_gradient(ba::ForwardDiffNoTagBackend, f, x::AbstractArray)
    result = DiffResults.GradientResult(x)
    cfg = ForwardDiff.GradientConfig(nothing, x, chunk(ba, x))
    ForwardDiff.gradient!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function AD.value_and_hessian(ba::ForwardDiffNoTagBackend, f, x)
    result = DiffResults.HessianResult(x)
    cfg = ForwardDiff.HessianConfig(nothing, result, x, chunk(ba, x))
    ForwardDiff.hessian!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end


@kwdef struct CMBLensingMuseProblem{DS<:DataSet,DS_SIM<:DataSet} <: AbstractMuseProblem
    ds :: DS
    ds_for_sims :: DS_SIM = ds
    parameterization = 0
    MAP_joint_kwargs = (;)
    θ_fixed = (;)
    x = ds.d
    latent_vars = nothing
    autodiff = AD.HigherOrderBackend((ForwardDiffNoTagBackend(), AD.ZygoteBackend()))
    transform_θ = identity
    inv_transform_θ = identity
end
CMBLensing.CMBLensingMuseProblem(ds, ds_for_sims=ds; kwargs...) = CMBLensingMuseProblem(;ds, ds_for_sims, kwargs...)

mergeθ(prob::CMBLensingMuseProblem, θ) = isempty(prob.θ_fixed) ? θ : (;prob.θ_fixed..., θ...)

function MuseInference.standardizeθ(prob::CMBLensingMuseProblem, θ)
    θ isa Union{NamedTuple,ComponentVector} || error("θ should be a NamedTuple or ComponentVector")
    1f0 * ComponentVector(θ) # ensure component vector and float
end

MuseInference.transform_θ(prob::CMBLensingMuseProblem, θ) = prob.transform_θ(θ)
MuseInference.inv_transform_θ(prob::CMBLensingMuseProblem, θ) = prob.inv_transform_θ(θ)

function MuseInference.logLike(prob::CMBLensingMuseProblem, d, z, θ, ::UnTransformedθ) 
    logpdf(prob.ds; z..., θ = mergeθ(prob, θ), d)
end
function MuseInference.logLike(prob::CMBLensingMuseProblem, d, z, θ, ::Transformedθ) 
    MuseInference.logLike(prob, d, z, MuseInference.inv_transform_θ(prob, θ), UnTransformedθ())
end

function MuseInference.∇θ_logLike(prob::CMBLensingMuseProblem, d, z, θ, θ_space)
    AD.gradient(prob.autodiff, θ -> MuseInference.logLike(prob, d, z, θ, θ_space), θ)[1]
end

function MuseInference.sample_x_z(prob::CMBLensingMuseProblem, rng::AbstractRNG, θ) 
    sim = simulate(rng, prob.ds_for_sims, θ = mergeθ(prob, θ))
    if prob.latent_vars == nothing
        # this is a guess which might not work for everything necessarily
        z = LenseBasis(FieldTuple(delete(sim, (:f̃, :d, :μ))) )
    else
        z = LenseBasis(FieldTuple(select(sim, prob.latent_vars)))
    end
    x = sim.d
    (;x, z)
end

function MuseInference.ẑ_at_θ(prob::CMBLensingMuseProblem, d, zguess, θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    Ωstart = delete(NamedTuple(zguess), :f)
    MAP = MAP_joint(mergeθ(prob, θ), @set(ds.d=d), Ωstart; fstart=zguess.f, prob.MAP_joint_kwargs...)
    LenseBasis(FieldTuple(;delete(MAP, :history)...)), MAP.history
end

function MuseInference.ẑ_at_θ(prob::CMBLensingMuseProblem{<:CMBLensing.Mixed}, d, zguess, θ; ∇z_logLike_atol=nothing)
    ds = prob.ds.ds
    zguess = CMBLensing.unmix(ds; θ, zguess...)
    Ωstart = delete(NamedTuple(zguess), :f)
    MAP = MAP_joint(Base.get_extension(CMBLensing,:CMBLensingMuseInferenceExt).mergeθ(prob, θ), @set(ds.d=d), Ωstart; fstart=zguess.f, prob.MAP_joint_kwargs...)
    MAP = CMBLensing.mix(ds; θ, MAP...)
    LenseBasis(FieldTuple(;delete(MAP, (:history, :θ))...)), MAP.history
end

function MuseInference.ẑ_at_θ(prob::CMBLensingMuseProblem{<:CMBLensing.NoLensingDataSet}, d, (f₀,), θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    f, hist = argmaxf_logpdf(@set(ds.d=d), (θ=mergeθ(prob, θ),); fstart=f₀, prob.MAP_joint_kwargs...)
    LenseBasis(FieldTuple(;f)), hist
end

function MuseInference.muse!(result::MuseResult, ds::DataSet, θ₀=nothing; parameterization=0, MAP_joint_kwargs=(;), kwargs...)
    muse!(result, CMBLensingMuseProblem(ds; parameterization, MAP_joint_kwargs), θ₀; kwargs...)
end

end
