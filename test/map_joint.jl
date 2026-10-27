using CMBLensing
using LinearAlgebra
using StableRNGs
using Test

@testset "MAP_joint line search" begin
    # Use the bundled spectra and a fixed CPU simulation, so this does not
    # require Python, CAMB, or a GPU. Both precisions exercise Optim's Brent
    # line search through the actual mixed-posterior objective.
    for T in (Float32, Float64)
        @testset "$T" begin
            (; ds) = load_sim(; Cℓ=CMBLensing._default_Cℓs(), pol=:I, T,
                θpix=4, Nside=8, storage=Array, L=LenseFlow(3), rng=StableRNG(7))
            result = MAP_joint(ds; nsteps=1, progress=false, αmax=1,
                conjgrad_kwargs=(tol=1e-4, nsteps=100),
                history_keys=(:f°, :ϕ°, :ΔΩ°, :α, :αmax, :total_logpdf))
            @test length(result.history) == 1
            h = only(result.history)

            # MAP_joint sets G=I. Compare the posterior at the chosen step to
            # both ends of the interval, holding the Wiener-filtered f° fixed.
            dsθ = ds()
            dsθ.G = I
            mixed = Mixed(dsθ)
            start = logpdf(mixed; f°=h.f°, ϕ°=zero(h.ϕ°))
            finish = logpdf(mixed; f°=h.f°, ϕ°=h.ϕ°)
            endpoint = logpdf(mixed; f°=h.f°, ϕ°=h.ϕ° + (h.αmax-h.α)*h.ΔΩ°.ϕ°)

            @test h.α isa T
            @test 0 < h.α < h.αmax
            @test norm(h.ΔΩ°) > 0
            @test all(isfinite, result.f[:]) && all(isfinite, result.ϕ[:])
            @test all(isfinite, (start, finish, endpoint))
            @test finish > start + T(1e-3)
            @test finish >= endpoint - T(2e-3)
            @test isapprox(h.total_logpdf, finish; atol=T(1e-3), rtol=T(1e-6))
        end
    end
end
