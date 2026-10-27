@testset "SVector times Field gradients" begin
    # All expected inner products below are computed directly in pixel space.
    # In Fourier space, the stored coefficients are complex and use a weighted
    # half-plane representation, so summing them is not the field inner product.
    pixels = [1.0 2.0 -1.0 3.0; -3.0 4.0 0.5 -2.0;
              2.5 -1.5 3.5 1.0; 0.25 -4.0 2.0 5.0]
    weights1 = [3.0 -2.0 1.5 0.5; 1.0 0.5 -2.0 3.0;
                -1.0 2.0 0.25 -3.0; 4.0 1.0 -0.5 2.0]
    weights2 = [-1.0 4.0 0.5 -2.5; 2.0 -3.0 1.0 0.25;
                3.0 -0.5 2.0 1.0; -2.0 3.5 -1.0 0.5]
    direction = [0.5 -0.25 1.0 0.75; -1.5 0.5 2.0 -1.0;
                 0.25 1.0 -0.5 2.0; -0.75 0.5 1.5 -2.0]
    f_map, w1_map, w2_map, h_map = map(x -> maybegpu(FlatMap(x)),
                                       (pixels, weights1, weights2, direction))
    v = @SVector [1.5, -2.0]
    dv = @SVector [0.3, -0.7]
    expected_v = @SVector [sum(pixels .* weights1), sum(pixels .* weights2)]
    expected_f = v[1] * w1_map + v[2] * w2_map
    ε = 1e-5

    for B in (Map, Fourier)
        f, w1, w2, h = B.((f_map, w1_map, w2_map, h_map))
        objective(v, f) = begin
            y = v * f
            dot(w1, y[1]) + dot(w2, y[2])
        end
        v̄, f̄ = gradient(objective, v, f)
        @test v̄ ≈ expected_v
        @test basis(f̄) == basis(f)
        @test f̄ ≈ B(expected_f)
        @test dot(v̄, dv) ≈ (objective(v + ε*dv, f) - objective(v - ε*dv, f)) / (2ε) rtol=1e-7
        @test dot(f̄, h) ≈ (objective(v, f + ε*h) - objective(v, f - ε*h)) / (2ε) rtol=1e-7

        for i in 1:2
            wi = (w1, w2)[i]
            single(v, f) = dot(wi, (v * f)[i])
            v̄, f̄ = gradient(single, v, f)
            expected_single = i == 1 ? (@SVector [expected_v[1], 0.0]) : (@SVector [0.0, expected_v[2]])
            @test v̄ ≈ expected_single
            @test basis(f̄) == basis(f)
            @test f̄ ≈ v[i] * wi
            @test dot(v̄, dv) ≈ (single(v + ε*dv, f) - single(v - ε*dv, f)) / (2ε) rtol=1e-7
            @test dot(f̄, h) ≈ (single(v, f + ε*h) - single(v, f - ε*h)) / (2ε) rtol=1e-7
        end

        # Also exercise explicit missing/zero components and a tangent in a
        # different basis, as returned by callers of the pullback.
        _, back = Zygote.pullback(*, v, f)
        for zero_tangent in (nothing, CMBLensing.ChainRulesCore.ZeroTangent())
            seed = CMBLensing.SVector{2,Any}((w1_map, zero_tangent))
            v̄, f̄ = back(seed)
            @test v̄ ≈ (@SVector [expected_v[1], 0.0])
            @test basis(f̄) == basis(f)
            @test f̄ ≈ B(v[1] * w1_map)
        end
    end
end
