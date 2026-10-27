using CMBLensing, Adapt, LinearAlgebra, Test, Serialization, FileIO, FFTW
using CMBLensing: @SVector

@testset "LenseFlow workspaces and fused diagonal rows" begin
    N = 8
    θpix = 60 * 180 / π
    ϕ = FlatMap([0.02sin(2π*(i+j)/N) for i=1:N, j=1:N]; θpix)
    f = FlatMap([cos(2π*(i-2j)/N) for i=1:N, j=1:N]; θpix)

    @test fieldnames(RK4Solver) == (:nsteps,)
    @test :graphs ∉ fieldnames(CachedLenseFlow)
    L = precompute!!(LenseFlow(3; gpu_graphs=true)(ϕ), f)
    expected = LenseFlow(ϕ, 3) * f
    @test L * f ≈ expected
    @test isnothing(L.odesolve.graphs.state) # CPU fallback never creates device state.
    @test adapt(Array, L).odesolve isa CMBLensing._GraphRK4Solver

    # Workspace sizing is part of compatibility, including when only the
    # batch/spin changes and all real element types remain the same.
    if FFTW.fftw_provider != "mkl"
        fb = batch(f, 0.7f)
        Lb = precompute!!(L, fb)
        @test Lb !== L
        @test (Lb * fb).arr ≈ (LenseFlow(ϕ, 3) * fb).arr
        @test precompute!!(Lb, f) !== Lb
    end
    fp = FlatQUMap(f.arr, 0.7f.arr; θpix)
    Lp = precompute!!(L, fp)
    @test Lp !== L
    @test Lp * fp ≈ LenseFlow(ϕ, 3) * fp

    # The opaque state can contain nonserializable GPU resources; only its
    # empty marker may survive copies, adaptation, or either checkpoint API.
    L.odesolve.graphs.state = Ref(:device_resource)
    for copied in (adapt(Array, L), deepcopy(L))
        @test copied.odesolve.graphs !== L.odesolve.graphs
        @test isnothing(copied.odesolve.graphs.state)
        @test copied * f ≈ expected
    end
    io = IOBuffer()
    serialize(io, L)
    seekstart(io)
    restored = deserialize(io)
    @test isnothing(restored.odesolve.graphs.state)
    @test restored.odesolve isa CMBLensing._GraphRK4Solver
    @test restored * f ≈ expected
    mktempdir() do dir
        path = joinpath(dir, "lenseflow.jld2")
        save(path, "L", L)
        restored = load(path, "L")
        @test isnothing(restored.odesolve.graphs.state)
        @test restored * f ≈ expected
    end
    @test L.odesolve.graphs.state[] === :device_resource

    @testset "complex diagonal conjugation" begin
        z = Fourier(f)
        v = @SVector [Diagonal((1+2im)*z), Diagonal((2-im)*z)]
        w = @SVector [z, (0.5+im)*z]
        x = Diagonal((0.3-im)*z)
        before = copy.(w)
        dst = similar(z)
        @test mul!(dst, v', w) ≈ v' * w
        @test mul!(dst, v', x, w) ≈ x * (v' * w)
        @test all(before .== w)
    end
end
