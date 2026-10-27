using CMBLensing, LinearAlgebra, Random, Test

# Custom simulations need not accept θ and may supply it in their result.
struct MixedSimulatorWithoutTheta{D,G,L,F} <: CMBLensing.DataSet
    D::D
    G::G
    L::L
    f::F
    ϕ::F
end
CMBLensing.simulate(::AbstractRNG, ds::MixedSimulatorWithoutTheta) =
    (; ds.f, ds.ϕ, θ=(field_scale=2.5, potential_scale=1.7))

struct MixedSimulatorWithTheta{D,G,L,F} <: CMBLensing.DataSet
    D::D
    G::G
    L::L
    f::F
    ϕ::F
end
CMBLensing.simulate(::AbstractRNG, ds::MixedSimulatorWithTheta; θ) =
    (; ds.f, ds.ϕ, θ=(field_scale=1.0, potential_scale=1.0), seenθ=θ)

@testset "Mixed simulation parameters" begin
    proj = ProjLambert(; Ny=8, Nx=8, θpix=60*180/π, T=Float64)
    d = FlatMap(reshape(sin.(collect(1.0:64.0) ./ 7), 8, 8), proj)
    C = Diagonal(one(d))
    Cf = ParamDependentOp((; signal_scale=1.0, _...)->signal_scale*C)
    D = ParamDependentOp((; field_scale=1.0, _...)->field_scale*C)
    G = ParamDependentOp((; potential_scale=1.0, _...)->potential_scale*C)
    ds = BaseDataSet(; d, Cf, Cn=C, Cϕ=C, D, G, L=(ϕ->I), Nϕ=C)

    for (θ, field_scale, potential_scale) in (
        ((;), 1.0, 1.0),
        ((field_scale=2.5, potential_scale=1.7, signal_scale=4.0), 2.5, 1.7),
    )
        base = simulate(MersenneTwister(163), ds; θ)
        mixed = simulate(MersenneTwister(163), CMBLensing.Mixed(ds); θ)
        @test mixed.θ == θ
        @test Array(Map(mixed.d).arr) == Array(Map(base.d).arr)
        @test Array(Map(mixed.f°).arr) ≈ field_scale .* Array(Map(base.f).arr) rtol=1e-12 atol=1e-12
        @test Array(Map(mixed.ϕ°).arr) ≈ potential_scale .* Array(Map(base.ϕ).arr) rtol=1e-12 atol=1e-12
    end

    default_sim = simulate(MersenneTwister(163), CMBLensing.Mixed(ds))
    empty_sim = simulate(MersenneTwister(163), CMBLensing.Mixed(ds); θ=(;))
    @test default_sim.θ == (;)
    @test Array(Map(default_sim.f°).arr) == Array(Map(empty_sim.f°).arr)
    @test Array(Map(default_sim.ϕ°).arr) == Array(Map(empty_sim.ϕ°).arr)

    θ = (field_scale=2.5, potential_scale=1.7, signal_scale=4.0)
    fixed_f = 0.3*d
    base = simulate(MersenneTwister(271), ds; θ, f=fixed_f)
    mixed = simulate(MersenneTwister(271), CMBLensing.Mixed(ds); θ, f=fixed_f)
    @test Array(Map(mixed.d).arr) == Array(Map(base.d).arr)
    @test Array(Map(mixed.f°).arr) ≈ 2.5 .* Array(Map(fixed_f).arr) rtol=1e-12 atol=1e-12

    custom = MixedSimulatorWithoutTheta(D, G, ϕ->I, d, 0.2*d)
    custom_mixed = simulate(MersenneTwister(163), CMBLensing.Mixed(custom))
    @test custom_mixed.θ == (field_scale=2.5, potential_scale=1.7)
    @test Array(Map(custom_mixed.f°).arr) ≈ 2.5 .* Array(Map(d).arr) rtol=1e-12 atol=1e-12
    @test Array(Map(custom_mixed.ϕ°).arr) ≈ 1.7 .* Array(Map(0.2*d).arr) rtol=1e-12 atol=1e-12

    custom_with_θ = MixedSimulatorWithTheta(D, G, ϕ->I, d, 0.2*d)
    custom_mixed = simulate(MersenneTwister(163), CMBLensing.Mixed(custom_with_θ); θ)
    @test custom_mixed.seenθ == θ
    @test custom_mixed.θ == θ
    @test Array(Map(custom_mixed.f°).arr) ≈ 2.5 .* Array(Map(d).arr) rtol=1e-12 atol=1e-12
    @test Array(Map(custom_mixed.ϕ°).arr) ≈ 1.7 .* Array(Map(0.2*d).arr) rtol=1e-12 atol=1e-12
end
