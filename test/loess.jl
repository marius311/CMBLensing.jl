@testset "Loess interpolation" begin
    xs = collect(1:20)
    ys = Float64.(2 .* xs .+ 3)

    # Mixed input types should be promoted and options forwarded to Loess.
    model = CMBLensing.cmblensing_loess(xs, ys; span=1.0, degree=1)
    @test size(model.bs, 2) == 2
    @test CMBLensing.Loess.predict(model, 10.0) ≈ 23.0

    # The generic matrix-input method must forward options as keywords too.
    matrix_model = CMBLensing.cmblensing_loess(reshape(Float64.(xs), :, 1), ys; span=1.0, degree=1)
    @test size(matrix_model.bs, 2) == 2
    @test CMBLensing.Loess.predict(matrix_model, 10.0) ≈ 23.0

    # Exercise the two package operations that use the helper.
    spectrum = CMBLensing.Cℓs(xs, ys)
    smoothed = CMBLensing.smooth(spectrum; smoothing=1.0)
    @test smoothed.Cℓ ≈ ys

    samples, logpdf, _ = CMBLensing.grid_and_sample(x -> -x^2 / 2, range(-3, 3; length=41); nsamples=0, span=0.75)
    @test isempty(samples)
    @test isfinite(logpdf(0.0))
end
