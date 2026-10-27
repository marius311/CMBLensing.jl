@testset "Fourier field sum gradients" begin
    # With unnormalized forward FFTs, a constant 4×4 map changes the
    # sum of the half-plane coefficients by 16. A unit pixel at (2,1)
    # changes their imaginary sum by -4.
    for T in (Float32, Float64), polarized in (false, true)
        zeros_map = zeros(T, 4, 4)
        constant = ones(T, 4, 4)
        impulse = zeros(T, 4, 4)
        impulse[2,1] = 1
        makefield(a) = polarized ? FlatQUMap(a, zeros_map) : FlatMap(a)
        f, real_dir, imag_dir = Fourier.((makefield(zeros_map), makefield(constant), makefield(impulse)))

        greal = gradient(x -> real(sum(x)), f)[1]
        gimag = gradient(x -> imag(sum(x)), f)[1]
        @test basis(greal) == basis(f)
        @test basis(gimag) == basis(f)
        @test eltype(greal) == eltype(f)
        @test eltype(gimag) == eltype(f)
        @test dot(greal, real_dir) ≈ T(16)
        @test dot(gimag, imag_dir) ≈ T(-4)

        gcomplex = gradient(x -> real(complex(T(0.7), T(1.3)) * sum(x)), f)[1]
        @test dot(gcomplex, real_dir) ≈ T(11.2)
        @test dot(gcomplex, imag_dir) ≈ T(5.2)

        _, sum_back = Zygote.pullback(sum, f)
        for zero_seed in (nothing, CMBLensing.ChainRulesCore.ZeroTangent())
            zero_back = sum_back(zero_seed)
            @test isnothing(zero_back) ||
                  (length(zero_back) == 1 &&
                   (isnothing(zero_back[1]) || zero_back[1] isa CMBLensing.ChainRulesCore.AbstractZero))
        end
    end

    # Metadata may be wider than the stored coefficients. The cotangent
    # must keep the input storage precision rather than promote to Float64.
    metadata64 = FlatMap(zeros(Float64, 4, 4)).metadata
    f32 = CMBLensing.BaseField{Fourier}(zeros(ComplexF32, 3, 4), metadata64)
    for objective in (x -> real(sum(x)), x -> imag(sum(x)))
        gradient32 = gradient(objective, f32)[1]
        @test eltype(gradient32) === ComplexF32
        @test size(gradient32.arr) == size(f32.arr)
    end
end
