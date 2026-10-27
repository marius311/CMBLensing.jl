using CMBLensing, LinearAlgebra, Random, Test, Zygote

@testset "Lambert polarization with odd widths" begin
    rng = MersenneTwister(541)
    for T in (Float32, Float64), ny in (2, 4, 6), nx in 1:9
        q = randn(rng, T, ny, nx)
        u = randn(rng, T, ny, nx)
        field = FlatQUMap(q, u; θpix=3)
        eb = EBFourier(field)
        restored = QUMap(eb)

        # Q/U -> E/B is an orthogonal change of basis. Compare directly
        # with the original maps as well as checking the field norm.
        @test restored.Qx ≈ q rtol=100eps(T) atol=100eps(T)
        @test restored.Ux ≈ u rtol=100eps(T) atol=100eps(T)
        @test norm(eb) ≈ sqrt(sum(abs2, q) + sum(abs2, u)) rtol=100eps(T) atol=100eps(T)
    end

    for T in (Float32, Float64), nx in (3, 5, 7)
        q, u, dq, du = (randn(rng, T, 6, nx) for _ in 1:4)
        field = FlatQUMap(q, u; θpix=3)
        direction = FlatQUMap(dq, du; θpix=3)
        gradient_field, = gradient(f -> norm(EBFourier(f))^2, field)
        expected = T(2) * (sum(q .* dq) + sum(u .* du))
        @test isapprox(dot(gradient_field, direction), expected;
            rtol=200eps(T), atol=200eps(T))
    end
end
