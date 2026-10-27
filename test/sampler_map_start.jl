using CMBLensing
using Adapt
using LinearAlgebra
using Random
using Test

struct SamplerStartFloat32 end
Adapt.adapt_storage(::SamplerStartFloat32, x::Array{Float64}) = Float32.(x)
Adapt.adapt_storage(::SamplerStartFloat32, x::Array{ComplexF64}) = ComplexF32.(x)

@testset "MAP-based sampler starts" begin
    # The bundled spectra keep this a real one-iteration solve without CAMB or
    # a GPU. Converting the dataset exercises the sampler's storage option.
    (; ds) = load_sim(; Cℓ=CMBLensing._default_Cℓs(), pol=:I, T=Float64,
        θpix=4, Nside=8, storage=Array, L=LenseFlow(3), rng=MersenneTwister(7))
    ds.Nϕ *= 2
    evaluations = Ref(0)
    original_Cϕ = ds.Cϕ
    ds.Cϕ = ParamDependentOp([:Aϕ], function (; Aϕ=1, _...)
        evaluations[] += 1
        original_Cϕ((; Aϕ))
    end)
    # The posterior's prior also receives the actual initialized parameters.
    ds.logprior = (; θ, _...) -> zero(θ.Aϕ)
    storage = SamplerStartFloat32()
    θ = (Aϕ=1.3,)
    map_kwargs = (; nsteps=1, αmax=1, conjgrad_kwargs=(tol=1e-4, nsteps=100))

    # Drawing θ from a fixed range makes θstart itself :prior; the optimizer
    # must instead receive the initialized θ and this dataset's Nϕ.
    expected = MAP_joint(θ, adapt(storage, ds(θ)); progress=false,
        quasi_sample=false, map_kwargs...).ϕ
    evaluations[] = 0
    chains = sample_joint(ds; nsamps_per_chain=1, nchains=1, pmap=map,
        resume=false, storage, ϕstart=:best_fit, θstart=:prior,
        θrange=(Aϕ=range(1.3, 1.3; length=2),), MAP_kwargs=map_kwargs,
        progress=:verbose)
    initial = only(chains[1])
    @test evaluations[] == 1
    @test initial[:θ] == θ
    @test eltype(initial[:ϕ]) == Float32
    @test all(isfinite, initial[:ϕ][:])
    @test norm(initial[:ϕ] - expected) ≈ 0 atol=1e-6 * norm(expected)

    # Reset the RNG around a direct optimizer call and the actual initializer
    # to verify that :quasi_sample selects a quasi-sample, not a best fit.
    Random.seed!(18)
    expected_quasi = MAP_joint(θ, adapt(storage, ds(θ)); progress=false,
        quasi_sample=true, map_kwargs...).ϕ
    state = Dict{Symbol,Any}(:ϕstart=>:quasi_sample, :θ=>θ, :Ω=>(;),
        :nchains=>1, :Nbatch=>(), :storage=>storage, :progress=>false,
        :MAP_kwargs=>map_kwargs)
    Random.seed!(18)
    evaluations[] = 0
    CMBLensing.gibbs_initialize_ϕ!(state, ds)
    @test evaluations[] == 1
    @test state[:Ω].ϕ === state[:ϕ]
    @test eltype(state[:ϕ]) == Float32
    @test all(isfinite, state[:ϕ][:])
    @test norm(state[:ϕ] - expected_quasi) ≈ 0 atol=1e-6 * norm(expected_quasi)
    @test norm(expected_quasi - expected) > 0

    # A resumed state already has a ϕ and does not need MAP-only options.
    resumed = Dict{Symbol,Any}(:ϕstart=>:best_fit, :θ=>θ, :Ω=>(;),
        :nchains=>1, :Nbatch=>(), :ϕ=>expected)
    CMBLensing.gibbs_initialize_ϕ!(resumed, ds)
    @test resumed[:ϕ] === expected && resumed[:Ω].ϕ === expected
end
