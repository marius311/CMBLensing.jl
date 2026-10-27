@testset "Loess interpolation" begin
    xs = collect(1:20)
    ys = Float64.(2 .* xs .+ 3)

    # Mixed input types should be promoted and options forwarded to Loess.
    model = CMBLensing.cmblensing_loess(xs, ys; span=1.0, degree=1)
    @test size(model.bs, 2) == 2
    @test CMBLensing.Loess.predict(model, 10.0) ≈ 23.0
    abstract_model = CMBLensing.cmblensing_loess(Any[xs...], Any[ys...]; span=1.0, degree=1)
    @test CMBLensing.Loess.predict(abstract_model, 10.0) ≈ 23.0
    offset_x = OffsetArray(Float64.(xs), -5:14)
    offset_y = OffsetArray(ys, -5:14)
    offset_model = CMBLensing.cmblensing_loess(offset_x, offset_y; span=1.0, degree=1)
    @test CMBLensing.Loess.predict(offset_model, 10.0) ≈ 23.0

    # The generic matrix-input method must forward options as keywords too.
    matrix_model = CMBLensing.cmblensing_loess(reshape(Float64.(xs), :, 1), ys; span=1.0, degree=1)
    @test size(matrix_model.bs, 2) == 2
    @test CMBLensing.Loess.predict(matrix_model, 10.0) ≈ 23.0
    offset_matrix = OffsetArray(reshape(Float64.(xs), :, 1), -5:14, 2:2)
    offset_matrix_model = CMBLensing.cmblensing_loess(offset_matrix, offset_y; span=1.0, degree=1)
    @test CMBLensing.Loess.predict(offset_matrix_model, 10.0) ≈ 23.0

    # Exercise the two package operations that use the helper.
    spectrum = CMBLensing.Cℓs(xs, ys)
    smoothed = CMBLensing.smooth(spectrum; smoothing=1.0)
    @test smoothed.Cℓ ≈ ys
    spectrum32 = CMBLensing.Cℓs(xs, Float32.(ys))
    @test CMBLensing.smooth(spectrum32; smoothing=1.0).Cℓ ≈ Float32.(ys)

    samples, logpdf, _ = CMBLensing.grid_and_sample(x -> -x^2 / 2, range(-3, 3; length=41); nsamples=0, span=0.75)
    @test isempty(samples)
    @test isfinite(logpdf(0.0))
    @test logpdf(0) ≈ logpdf(0.0)
    _, abstract_logpdf, _ = CMBLensing.grid_and_sample(Any[-x^2 / 2 for x in xs], Any[Float64(x) for x in xs]; nsamples=0, span=0.75)
    @test isfinite(abstract_logpdf(10))
end
