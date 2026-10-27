using CMBLensing, FFTW, Random, Test

@testset "EquiRect polarization Fourier conversion" begin
    rng = MersenneTwister(713)
    for T in (Float32, Float64), nx in 1:8
        proj = ProjEquiRect(; Ny=4, Nx=nx, θspan=(0.4, 1.1),
            φspan=(0.0, 1.4), T)
        q = randn(rng, T, 4, nx)
        u = randn(rng, T, 4, nx)
        field = EquiRectQUMap(q, u, proj)
        transformed = QUAzFourier(field)

        # Direct FFTs of Q+iU and Q-iU independently describe the two
        # packed halves; this also checks the odd-width highest frequency.
        polarization = complex.(q, u)
        expected = vcat(fft(polarization, 2), fft(conj.(polarization), 2))[:, 1:nx÷2+1] ./ T(sqrt(nx))
        @test size(transformed.arr) == (8, nx÷2+1)
        @test transformed.arr ≈ expected rtol=20eps(T) atol=20eps(T)

        restored = QUMap(transformed)
        @test restored.Qx ≈ q rtol=20eps(T) atol=20eps(T)
        @test restored.Ux ≈ u rtol=20eps(T) atol=20eps(T)
    end
end
