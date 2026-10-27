using CMBLensing, MuseInference, ComponentArrays, LinearAlgebra, Test

@testset "Mixed MUSE fixed parameters" begin
    # For an identity forward model with unit signal/noise covariance, the
    # conditional maximum has f=d/2. The mixing operator is scale*I.
    proj = ProjLambert(; Ny=8, Nx=8, θpix=60*180/π, T=Float64)
    d = FlatMap(reshape(sin.(collect(1.0:64.0) ./ 7),8,8),proj)
    f0, ϕ0 = 0.3*d, 0.1*d
    C = Diagonal(one(d))
    D = ParamDependentOp((; fixed_scale=1.0, _...)->fixed_scale*C)
    ds = BaseDataSet(; d, Cf=C, Cn=C, Cϕ=C, D, G=I, L=(ϕ->I), Nϕ=C)
    mixed_ds = CMBLensing.Mixed(ds)
    alternate_data = zero(d)
    @test CMBLensingMuseProblem(mixed_ds).x === d
    @test CMBLensingMuseProblem(mixed_ds; x=alternate_data).x === alternate_data
    @test CMBLensingMuseProblem(ds).x === d
    @test CMBLensingMuseProblem(ds; x=alternate_data).x === alternate_data

    for (fixed, supplied, scale) in (
        ((;), (fixed_scale=2.5, unused=0.7), 2.5),
        ((fixed_scale=2.5,), (unused=0.7,), 2.5),
        ((fixed_scale=2.5,), ComponentVector(unused=0.7), 2.5),
        ((fixed_scale=2.5,), (fixed_scale=3.0, unused=0.7), 3.0),
    )
        full = (fixed_scale=scale, unused=0.7)
        initial = CMBLensing.mix(ds; f=f0, ϕ=ϕ0, θ=full)
        guess = FieldTuple(; f°=initial.f°, ϕ°=initial.ϕ°)
        for nsteps in (0,1)
            options = (; nsteps, progress=false, αmax=1.0,
                conjgrad_kwargs=(tol=1e-8,nsteps=100))
            prob = CMBLensingMuseProblem(mixed_ds; θ_fixed=fixed, MAP_joint_kwargs=options)
            result, history = MuseInference.ẑ_at_θ(prob,d,guess,supplied)
            expected_f = nsteps == 0 ? scale*0.3 : scale*0.5
            @test Array(Map(result.f°).arr) ≈ expected_f .* Array(d.arr) rtol=1e-8 atol=1e-10
            @test length(history) == nsteps
        end
    end

    # A starting guess may contain stale parameter metadata. Check which
    # parameters the real unmix/MAP/mix path uses: a round trip alone would
    # hide the error when both transformations use the same stale value.
    seen_scales = Float64[]
    traced_D = ParamDependentOp((; fixed_scale=1.0, _...)->
        (push!(seen_scales, fixed_scale); fixed_scale*C))
    traced_ds = BaseDataSet(; d, Cf=C, Cn=C, Cϕ=C, D=traced_D, G=I, L=(ϕ->I), Nϕ=C)
    full = (fixed_scale=2.5, unused=0.7)
    initial = CMBLensing.mix(traced_ds; f=f0, ϕ=ϕ0, θ=full)
    guess = (; f°=initial.f°, ϕ°=initial.ϕ°, θ=(fixed_scale=9.0, unused=0.7))
    prob = CMBLensingMuseProblem(CMBLensing.Mixed(traced_ds); θ_fixed=(fixed_scale=2.5,),
        MAP_joint_kwargs=(nsteps=0, progress=false))
    empty!(seen_scales)
    result, history = MuseInference.ẑ_at_θ(prob, d, guess, (unused=0.7,))
    @test !isempty(seen_scales)
    @test all(==(2.5), seen_scales)
    @test Array(Map(result.f°).arr) ≈ Array(Map(initial.f°).arr) rtol=1e-12 atol=1e-12
    @test !haskey(result, :θ)
    @test isempty(history)
end
