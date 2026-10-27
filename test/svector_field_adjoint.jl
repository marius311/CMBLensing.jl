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
        @test back(CMBLensing.SVector{2,Any}((nothing, CMBLensing.ChainRulesCore.ZeroTangent()))) == (nothing, nothing)
    end
end

@testset "SVector times batched Field pullback" begin
    pixels = [1.0 2.0 -1.0 3.0; -3.0 4.0 0.5 -2.0;
              2.5 -1.5 3.5 1.0; 0.25 -4.0 2.0 5.0]
    weights1 = [3.0 -2.0 1.5 0.5; 1.0 0.5 -2.0 3.0;
                -1.0 2.0 0.25 -3.0; 4.0 1.0 -0.5 2.0]
    weights2 = [-1.0 4.0 0.5 -2.5; 2.0 -3.0 1.0 0.25;
                3.0 -0.5 2.0 1.0; -2.0 3.5 -1.0 0.5]
    f_map, w1_map, w2_map = map(x -> maybegpu(FlatMap(x)), (pixels, weights1, weights2))
    fb_map = batch(f_map, 2f_map)
    w1b_map = batch(w1_map, w2_map)
    w2b_map = batch(2w2_map, -w1_map)
    v = @SVector [1.5, -2.0]
    vb = @SVector [batch([1.5, -0.5]), batch([-2.0, 3.0])]
    vb_single = @SVector [batch([1.5]), batch([-2.0])]
    shared1 = [sum(pixels .* weights1), sum(pixels .* weights2)]
    shared2 = [2sum(pixels .* weights2), -sum(pixels .* weights1)]
    perbatch1 = [shared1[1], 2shared1[2]]
    perbatch2 = [shared2[1], 2shared2[2]]

    for B in (Map, Fourier)
        B == Fourier && !has_batched_fft && continue
        f, fb, w1b, w2b = B.((f_map, fb_map, w1b_map, w2b_map))
        seed = @SVector [w1b, w2b]

        # Plain coefficients are shared by both batches and sum contributions.
        _, back = Zygote.pullback(*, v, fb)
        v̄, f̄ = back(seed)
        @test v̄ ≈ (@SVector [sum(perbatch1), sum(perbatch2)])
        @test f̄ ≈ v[1] * w1b + v[2] * w2b
        @test size(f̄.arr) == size(fb.arr)

        # Batched coefficients retain separate contributions for each batch.
        _, back = Zygote.pullback(*, vb, fb)
        v̄, f̄ = back(seed)
        @test unbatch(v̄[1]) ≈ perbatch1
        @test unbatch(v̄[2]) ≈ perbatch2
        @test f̄ ≈ vb[1] * w1b + vb[2] * w2b

        # A length-one BatchedReal broadcasts and receives one summed tangent.
        _, back = Zygote.pullback(*, vb_single, fb)
        v̄, _ = back(seed)
        @test unbatch(v̄[1]) ≈ [sum(perbatch1)]
        @test unbatch(v̄[2]) ≈ [sum(perbatch2)]

        # A shared input field likewise receives the sum over the batches.
        _, back = Zygote.pullback(*, vb, f)
        v̄, f̄ = back(seed)
        expected_shared = 1.5w1_map - 0.5w2_map - 4w2_map - 3w1_map
        @test unbatch(v̄[1]) ≈ shared1
        @test unbatch(v̄[2]) ≈ shared2
        @test basis(f̄) == basis(f)
        @test size(f̄.arr) == size(f.arr)
        @test f̄ ≈ B(expected_shared)

        # The same reduction applies independently to FieldTuple components.
        ft = FieldTuple(; first=f, second=f)
        w1t = FieldTuple(; first=w1b, second=w2b)
        w2t = FieldTuple(; first=w2b, second=w1b)
        _, back = Zygote.pullback(*, vb, ft)
        _, ft̄ = back(@SVector [w1t, w2t])
        @test size(ft̄.first.arr) == size(f.arr)
        @test size(ft̄.second.arr) == size(f.arr)
        @test ft̄.first ≈ B(expected_shared)
        @test ft̄.second ≈ B(-1.5w1_map + 6w2_map)

        # The two coefficients may have different batch behavior.
        vmixed = CMBLensing.SVector{2,Real}((1.5, batch([-2.0, 3.0])))
        _, back = Zygote.pullback(*, vmixed, fb)
        v̄, _ = back(seed)
        @test v̄[1] ≈ sum(perbatch1)
        @test unbatch(v̄[2]) ≈ perbatch2

        # The cotangent's storage precision is projected back to the input.
        f32 = B(maybegpu(FlatMap(Float32.(pixels))))
        v32 = @SVector [1.5f0, -2.0f0]
        _, back = Zygote.pullback(*, v32, f32)
        v̄, f̄ = back(@SVector [w1_map, w2_map])
        @test eltype(v̄) == Float32
        @test eltype(f̄) == eltype(f32)
        @test Map(f̄).arr ≈ Float32.(1.5weights1 - 2weights2)
    end
end
