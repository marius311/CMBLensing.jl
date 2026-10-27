@testset "EquiRect scalar AzFourier array gradients" begin
    for T in (Float32, Float64), nx in (4, 5), nbatch in (1, 2)
        # MKL's CPU FFT backend does not support this batched transform.
        nbatch == 2 && FFTW.fftw_provider == "mkl" && continue
        proj = ProjEquiRect(; Ny=4, Nx=nx, T, θspan=(0,π), φspan=(0,2π), storage=Array)
        vals = reshape(T.(sin.(1:4nx*nbatch)), 4, nx, 1, nbatch)
        dir = reshape(T.(cos.(1:4nx*nbatch)), 4, nx, 1, nbatch)
        f = CMBLensing.EquiRectMap(vals, proj)
        h = CMBLensing.EquiRectMap(dir, proj)
        az = CMBLensing.AzFourier(f)
        n = length(az.arr)
        weights = reshape(complex.(T.(sin.(1:n)), T.(cos.(1:n))), size(az.arr))
        raw_direction = reshape(complex.(T.(cos.(1:n)), T.(sin.(1:n))), size(az.arr))
        ε = T === Float32 ? T(1e-3) : T(1e-5)
        tol = T === Float32 ? 4e-3 : 2e-6

        # Arbitrary complex seeds exercise the distinct rFFT weights on the
        # DC/Nyquist bins and on the interior frequencies.
        for B in (Map, CMBLensing.AzFourier)
            x, d = B(f), B(h)
            objective(x) = sum(real.(conj.(weights) .* CMBLensing.AzFourier(x).arr))
            g = gradient(objective, x)[1]
            directional = dot(Map(g).arr, Map(d).arr)
            finite = (objective(x + ε*d) - objective(x - ε*d)) / (2ε)
            @test isapprox(directional, finite; atol=tol, rtol=tol)
        end

        # Constructor followed by array access is exactly the identity,
        # including for independent complex coefficients at the endpoints.
        raw = copy(az.arr)
        constructor_array(x) = CMBLensing.EquiRectAzFourier(x, proj).arr
        objective_raw(x) = sum(real.(conj.(weights) .* constructor_array(x)))
        raw_gradient = gradient(objective_raw, raw)[1]
        @test isapprox(raw_gradient, weights; atol=tol, rtol=tol)

        # Also check the constructor followed by the inverse transform against
        # an independent directional finite difference in the raw array.
        map_weights = reshape(T.(cos.(1:4nx*nbatch)), size(vals))
        objective_map(x) = sum(map_weights .* Map(CMBLensing.EquiRectAzFourier(x, proj)).arr)
        map_gradient = gradient(objective_map, raw)[1]
        directional = sum(real.(conj.(map_gradient) .* raw_direction))
        finite = (objective_map(raw + ε*raw_direction) - objective_map(raw - ε*raw_direction)) / (2ε)
        @test isapprox(directional, finite; atol=tol, rtol=tol)
    end
end
