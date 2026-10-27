using CMBLensing, LinearAlgebra, Random, Test

@testset "PowerLens anti-lensing" begin
    for T in (Float32, Float64)
        Ny, Nx = 6, 8
        proj = ProjLambert(; Ny, Nx, θpix=60*180/π, T)
        # The two deflection components differ, so swapping or duplicating
        # either component changes the resulting lensing operator.
        potential = FlatMap(T[0.08sin(0.7i + 1.3j) + 0.05cos(1.1i - 0.4j)
                              for i in 1:Ny, j in 1:Nx], proj)
        rng = MersenneTwister(30271)
        scalar() = FlatMap(randn(rng, T, Ny, Nx), proj)
        q, u = scalar(), scalar()
        polarization = FlatQUMap(q.Ix, u.Ix, proj)
        fields = (scalar(), Fourier(scalar()), polarization, QUFourier(polarization))
        tol = T === Float32 ? 3e-4 : 2e-11

        for order in (0, 1, 2, 3)
            lens = PowerLens(potential, order)
            anti = CMBLensing.antilensing(lens)
            reference = PowerLens(-potential, order)
            restored = CMBLensing.antilensing(anti)
            @test anti.order == order
            for f in fields
                # Compare both the operator and its adjoint to independently
                # constructed lenses, and apply anti-lensing twice.
                for (got, expected) in ((anti*f, reference*f),
                                        (anti'*f, reference'*f),
                                        (restored*f, lens*f),
                                        (restored'*f, lens'*f))
                    @test isapprox(Map(got).arr, Map(expected).arr; rtol=tol, atol=tol)
                end
            end
        end
    end
end
