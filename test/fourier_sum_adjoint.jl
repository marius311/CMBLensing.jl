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
    end
end
