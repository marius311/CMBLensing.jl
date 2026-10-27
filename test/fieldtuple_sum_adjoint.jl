@testset "FieldTuple sum gradients" begin
    for T in (Float32, Float64)
        zeros_map = zeros(T, 4, 4)
        ones_map = ones(T, 4, 4)
        impulse = zeros(T, 4, 4)
        impulse[2,1] = 1
        ft = FieldTuple(; map=FlatMap(zeros_map), fourier=Fourier(FlatMap(zeros_map)),
                         scalar=T(2), vector=T[3,4])
        real_dir = FieldTuple(; map=FlatMap(ones_map), fourier=Fourier(FlatMap(ones_map)),
                               scalar=T(0.2), vector=T[0.1,0.3])
        imag_dir = FieldTuple(; map=FlatMap(zeros_map), fourier=Fourier(FlatMap(impulse)),
                               scalar=T(0), vector=T[0,0])

        greal = gradient(x -> real(sum(x)), ft)[1]
        gimag = gradient(x -> imag(sum(x)), ft)[1]
        gcomplex = gradient(x -> real(complex(T(0.7), T(1.3))*sum(x)), ft)[1]
        pair(g, d) = dot(g.map, d.map) + dot(g.fourier, d.fourier) +
                     g.scalar*d.scalar + dot(g.vector, d.vector)
        @test greal isa FieldTuple
        @test gimag isa FieldTuple
        @test pair(greal, real_dir) ≈ T(32.6)
        @test pair(gimag, imag_dir) ≈ T(-4)
        @test pair(gcomplex, real_dir) ≈ T(22.82)
        @test pair(gcomplex, imag_dir) ≈ T(5.2)
        @test greal.scalar ≈ one(T)
        @test greal.vector ≈ ones(T,2)
        @test iszero(gimag.scalar)
        @test all(iszero, gimag.vector)
        @test all(iszero, Map(gimag.map).arr)
        @test eltype(greal.map) === T
        @test eltype(greal.fourier) === Complex{T}
        @test greal.scalar isa T
        @test eltype(greal.vector) === T

        # Compare the mixed-component tangent with a scalar objective's
        # independent finite difference, including a complex output weight.
        ε = T === Float32 ? T(1e-3) : T(1e-5)
        tol = T === Float32 ? 1e-2 : 1e-6
        for direction in (real_dir, imag_dir)
            objective(a) = real(complex(T(0.7), T(1.3)) * sum(FieldTuple(;
                map=ft.map + a*direction.map,
                fourier=ft.fourier + a*direction.fourier,
                scalar=ft.scalar + a*direction.scalar,
                vector=ft.vector + a*direction.vector)))
            finite = (objective(ε) - objective(-ε)) / (2ε)
            @test isapprox(pair(gcomplex, direction), finite; rtol=tol, atol=tol)
        end

        nested = FieldTuple(; inner=ft)
        nested_gradient = gradient(x -> real(sum(x)), nested)[1]
        @test nested_gradient.inner.scalar ≈ one(T)
        @test pair(nested_gradient.inner, real_dir) ≈ T(32.6)
        _, sum_back = Zygote.pullback(sum, ft)
        for zero_seed in (nothing, CMBLensing.ChainRulesCore.ZeroTangent())
            zero_back = sum_back(zero_seed)
            @test isnothing(zero_back) ||
                  (length(zero_back) == 1 &&
                   (isnothing(zero_back[1]) || zero_back[1] isa CMBLensing.ChainRulesCore.AbstractZero))
        end
    end

    metadata64 = FlatMap(zeros(Float64, 4, 4)).metadata
    narrow_fourier = CMBLensing.BaseField{Fourier}(zeros(ComplexF32, 3, 4), metadata64)
    mixed = FieldTuple(; field=narrow_fourier, scalar=1.0)
    for objective in (x -> real(sum(x)), x -> imag(sum(x)))
        result = gradient(objective, mixed)[1]
        @test eltype(result.field) === ComplexF32
    end
end
