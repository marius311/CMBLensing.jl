using CMBLensing, MuseInference, ComponentArrays, LinearAlgebra, Random, Test

@testset "MUSE mixed simulation and default latents" begin
    proj = ProjLambert(; Ny=8, Nx=8, θpix=60*180/π, T=Float64)
    d = FlatMap(reshape(sin.(collect(1.0:64.0) ./ 7), 8, 8), proj)
    C = Diagonal(one(d))
    Cf = ParamDependentOp((; signal_scale=1.0, _...)->signal_scale*C)
    D = ParamDependentOp((; field_scale=1.0, _...)->field_scale*C)
    G = ParamDependentOp((; potential_scale=1.0, _...)->potential_scale*C)
    ds = BaseDataSet(; d, Cf, Cn=C, Cϕ=C, D, G, L=(ϕ->I), Nϕ=C)
    mixed_ds = CMBLensing.Mixed(ds)

    for (fixed, supplied, scale) in (
        ((;), (field_scale=2.5, potential_scale=1.7, signal_scale=4.0), 2.5),
        ((;), ComponentVector(field_scale=2.5, potential_scale=1.7, signal_scale=4.0), 2.5),
        ((field_scale=2.5, potential_scale=1.7), (signal_scale=4.0,), 2.5),
        ((field_scale=2.5, potential_scale=1.7), ComponentVector(signal_scale=4.0), 2.5),
        ((field_scale=2.5, potential_scale=1.7), (field_scale=3.0, signal_scale=4.0), 3.0),
    )
        θ = (field_scale=scale, potential_scale=1.7, signal_scale=4.0)
        base = simulate(MersenneTwister(163), ds; θ)
        prob = CMBLensingMuseProblem(mixed_ds; θ_fixed=fixed,
            MAP_joint_kwargs=(nsteps=0, progress=false))
        sim = MuseInference.sample_x_z(prob, MersenneTwister(163), supplied)
        @test keys(sim.z.fs) == (:f°, :ϕ°)
        @test Array(Map(sim.x).arr) == Array(Map(base.d).arr)
        @test Array(Map(sim.z.f°).arr) ≈ scale .* Array(Map(base.f).arr) rtol=1e-12 atol=1e-12
        @test Array(Map(sim.z.ϕ°).arr) ≈ 1.7 .* Array(Map(base.ϕ).arr) rtol=1e-12 atol=1e-12
        zhat, history = MuseInference.ẑ_at_θ(prob, sim.x, sim.z, supplied)
        @test isempty(history)
        @test Array(Map(zhat.f°).arr) ≈ Array(Map(sim.z.f°).arr) rtol=1e-12 atol=1e-12
        @test Array(Map(zhat.ϕ°).arr) ≈ Array(Map(sim.z.ϕ°).arr) rtol=1e-12 atol=1e-12
    end

    θ = (field_scale=2.5, potential_scale=1.7, signal_scale=4.0)
    base = simulate(MersenneTwister(163), ds; θ)
    explicit = CMBLensingMuseProblem(mixed_ds; latent_vars=(:f°,))
    selected = MuseInference.sample_x_z(explicit, MersenneTwister(163), θ)
    @test keys(selected.z.fs) == (:f°,)
    @test Array(Map(selected.z.f°).arr) ≈ 2.5 .* Array(Map(base.f).arr) rtol=1e-12 atol=1e-12

    regular = MuseInference.sample_x_z(CMBLensingMuseProblem(ds), MersenneTwister(163), θ)
    @test keys(regular.z.fs) == (:f, :ϕ)
    @test Array(Map(regular.x).arr) == Array(Map(base.d).arr)
    @test Array(Map(regular.z.f).arr) ≈ Array(Map(base.f).arr) rtol=1e-12 atol=1e-12
    @test Array(Map(regular.z.ϕ).arr) ≈ Array(Map(base.ϕ).arr) rtol=1e-12 atol=1e-12
end
