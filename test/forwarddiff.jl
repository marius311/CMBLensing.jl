@testset "ForwardDiff bilinear sampling" begin
    for T in (Float32, Float64)
        img = T[1 2; 3 4]
        sample(x) = CMBLensing._bilinear_sample(img, x[1], x[2])
        for (xy, expected) in (
            (T[1, 1.5], T[0, 1]),
            (T[1.5, 1], T[2, 0]),
            (T[1, 1], T[0, 0]),
            (T[1.25, 1.75], T[2, 1]),
            (T[0.5, 1.25], T[1.25, 0.5]),
        )
            @test ForwardDiff.gradient(sample, xy) ≈ expected
            reverse_gradient = Zygote.gradient(sample, xy)[1]
            @test isnothing(reverse_gradient) ? all(iszero, expected) : reverse_gradient ≈ expected
        end

        # A pixel with zero weight must not contaminate the value or either
        # coordinate derivative, even when that pixel contains a NaN.
        for (img_nan, xy, expected) in (
            (T[1 2; NaN NaN], T[1, 1.5], T[0, 1]),
            (T[1 NaN; 3 NaN], T[1.5, 1], T[2, 0]),
        )
            sample_nan(x) = CMBLensing._bilinear_sample(img_nan, x[1], x[2])
            @test isfinite(sample_nan(xy))
            @test ForwardDiff.gradient(sample_nan, xy) ≈ expected
            @test Zygote.gradient(sample_nan, xy)[1] ≈ expected
        end

        img_nan = T[1 2; NaN NaN]
        sample_image(pixels) = CMBLensing._bilinear_sample(reshape(pixels, 2, 2), T(1), T(1.5))
        image_gradient = T[0.5, 0, 0.5, 0]
        @test ForwardDiff.gradient(sample_image, vec(img_nan)) ≈ image_gradient
        @test Zygote.gradient(sample_image, vec(img_nan))[1] ≈ image_gradient

        # Nested dual numbers must select the same exact-grid branch.
        sample_row(r) = CMBLensing._bilinear_sample(T[1 2; NaN NaN], r, T(1.5))
        @test ForwardDiff.derivative(sample_row, T(1)) == 0
        @test ForwardDiff.derivative(x -> ForwardDiff.derivative(sample_row, x), T(1)) == 0
    end
end

@testset "ForwardDiff zero-potential bilinear lens" begin
    for T in (Float32, Float64)
        proj = ProjLambert(Ny=4, Nx=4, θpix=4, T=T)
        field = FlatMap(reshape(T.(1:16) ./ T(16), 4, 4), proj)
        direction = FlatMap(T(1e-8) .* reshape(sin.(T.(1:16)), 4, 4), proj)
        lens_power = a -> sum(abs2, Map(BilinearLens(a * direction) * field).Ix)
        @test lens_power(T(0)) ≈ sum(abs2, field.Ix)
        @test ForwardDiff.derivative(lens_power, T(0)) == 0
    end
end

@testset "ForwardDiff FFT plans" begin
    # Compare Jacobians with independent applications of each linear plan.
    asreal(x) = vcat(vec(real.(x)), vec(imag.(x)))
    for T in (Float32, Float64)
        a = T[0.1, 0.7, -0.3, 0.4]
        b = T[0.3, -0.2, 0.1, 0.9]
        p = T[0.4, -0.7]
        rplan = plan_rfft(a)
        fftplan = plan_fft(complex.(a, b))
        c, d = complex.(a, b), complex.(b, -a)
        for (f, expected) in (
            (x -> asreal(rplan * (x[1]*a + x[2]*b)), hcat(asreal(rplan*a), asreal(rplan*b))),
            (x -> asreal(rfft(x[1]*a + x[2]*b)), hcat(asreal(rplan*a), asreal(rplan*b))),
            (x -> asreal(fftplan * (x[1]*c + x[2]*d)), hcat(asreal(fftplan*c), asreal(fftplan*d))),
            (x -> asreal(fft(x[1]*c + x[2]*d)), hcat(asreal(fftplan*c), asreal(fftplan*d))),
            (x -> asreal(fftplan \ (x[1]*c + x[2]*d)), hcat(asreal(fftplan\c), asreal(fftplan\d))),
            (x -> asreal(inv(rplan) * (x[1]*(rplan*a) + x[2]*(rplan*b))), hcat(asreal(a), asreal(b))),
        )
            @test ForwardDiff.jacobian(f, p) ≈ expected
        end

        function fmul(x)
            input = x[1]*a + x[2]*b
            output = Vector{Complex{eltype(input)}}(undef, length(rplan*a))
            mul!(output, rplan, input)
            asreal(output)
        end
        @test ForwardDiff.jacobian(fmul, p) ≈ hcat(asreal(rplan*a), asreal(rplan*b))

        function fmulinv(x)
            input = x[1]*(rplan*a) + x[2]*(rplan*b)
            output = Vector{typeof(real(first(input)))}(undef, length(a))
            mul!(output, inv(rplan), input)
            output
        end
        @test ForwardDiff.jacobian(fmulinv, p) ≈ hcat(a, b)
    end

    # NFFT and its adjoint have separate dual-number dispatches.
    nodes = [-0.31 0.01 0.26; 0.2 -0.3 0.14]
    plan = NFFT.plan_nfft(Array, nodes, (4, 4))
    a = reshape(complex.(collect(1.0:16.0), collect(16.0:-1.0:1.0)), 4, 4)
    b = (0.2 + 0.4im) .* reverse(a, dims=1)
    p = [0.4, -0.7]
    f = x -> asreal(plan * (x[1]*a + x[2]*b))
    @test ForwardDiff.jacobian(f, p) ≈ hcat(asreal(plan*a), asreal(plan*b))
    c, d = plan*a, plan*b
    g = x -> asreal(adjoint(plan) * (x[1]*c + x[2]*d))
    @test ForwardDiff.jacobian(g, p) ≈ hcat(asreal(adjoint(plan)*c), asreal(adjoint(plan)*d))
end

@testset "ForwardDiff Lambert projection" begin
    proj = ProjLambert(Ny=8, Nx=8, θpix=4, T=Float64, rotator=(0, 30, 0))
    coordinates = [1.1, 1.7]
    project_coordinates = x -> collect(CMBLensing.θϕ_to_ij(proj, x[1], x[2]))
    h = 1e-6
    finite_jacobian = hcat(((project_coordinates(coordinates + h * (1:2 .== i)) -
                             project_coordinates(coordinates - h * (1:2 .== i))) / (2h) for i in 1:2)...)
    @test ForwardDiff.jacobian(project_coordinates, coordinates) ≈ finite_jacobian rtol=1e-6
    @test isfinite(CMBLensing.get_ψpol(proj, coordinates...))
end
