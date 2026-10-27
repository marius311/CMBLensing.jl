using CMBLensing, ForwardDiff, LinearAlgebra, Random, Test, Zygote

@testset "BilinearLens inverse with zero field" begin
    for T in (Float32, Float64)
        rng = MersenneTwister(72619)
        Ny, Nx = 6, 8
        proj = ProjLambert(; Ny, Nx, θpix=60*180/π, T)
        ϕ = maybegpu(FlatMap(T(0.1)*randn(rng,T,Ny,Nx),proj))
        f = maybegpu(FlatMap(randn(rng,T,Ny,Nx),proj))
        z = maybegpu(FlatMap(zeros(T,Ny,Nx),proj))
        qu = maybegpu(FlatQUMap(zeros(T,Ny,Nx),Array(f.Ix),proj))
        mixed_batch = batch(z,f)
        L = BilinearLens(ϕ)
        original_f = Array(f.arr)
        original_qu = Array(qu.arr)
        tol = T === Float32 ? 2e-3 : 2e-9
        for adj in (false,true)
            apply(x) = adj ? L'\x : L\x
            scalar = apply(f)
            zero_scalar = apply(z)
            mixed_qu = apply(qu)
            mixed = apply(mixed_batch)
            @test all(iszero,Array(zero_scalar.arr))
            @test eltype(zero_scalar.arr) === T
            @test all(iszero,Array(mixed_qu.Qx))
            @test isapprox(Array(mixed_qu.Ux),Array(scalar.Ix);atol=tol,rtol=tol)
            @test all(iszero,Array(mixed.arr[:,:,1,1]))
            @test isapprox(Array(mixed.arr[:,:,1,2]),Array(scalar.Ix);atol=tol,rtol=tol)
        end
        @test Array(f.arr) == original_f
        @test Array(qu.arr) == original_qu
    end
end

@testset "GMRES zero result type and differentiation" begin
    # Test the solver's promoted output type, including real/complex mixed
    # precision, using plain CPU arrays independently of the field storage.
    for (TA,TB) in ((Float32,Float64),(Float64,Float32),
                    (ComplexF32,Float64),(Float64,ComplexF32))
        A = TA.(diagm(0=>range(1.0,1.1;length=8)))
        x = CMBLensing.gmres(A,zeros(TB,8);maxiter=5)
        @test eltype(x) === promote_type(TA,TB)
        @test all(iszero,x)
    end

    # A nonzero, well-conditioned system is solved to a known dense reference.
    A = diagm(0=>range(1.0,1.1;length=8))
    b = collect(range(-1.0,2.0;length=8))
    x = CMBLensing.gmres(A,b;maxiter=5)
    @test isapprox(x,A\b;atol=1e-8,rtol=1e-8)

    # At zero RHS the fixed-step solver has direction-dependent, generally
    # nonadditive field slopes. A dual seed must not silently become zero.
    @test_throws ArgumentError ForwardDiff.derivative(
        t -> sum(CMBLensing.gmres(A,t*b;maxiter=5)), 0.0)
    Ac = complex.(A)
    bc = complex.(b,reverse(b))
    @test_throws ArgumentError ForwardDiff.derivative(
        t -> real(sum(CMBLensing.gmres(Ac,t*bc;maxiter=5))), 0.0)
    @test_throws Exception Zygote.gradient(
        v -> sum(CMBLensing.gmres(A,v;maxiter=5)), zero(b))

    # Reverse-mode inverse gradients are still unsupported. A nonzero field
    # retains its pre-existing ForwardDiff behavior; zero potential remains
    # the exact identity branch for field differentiation.
    rng = MersenneTwister(41773)
    proj = ProjLambert(;Ny=6,Nx=8,θpix=60*180/π,T=Float64)
    field() = FlatMap(randn(rng,6,8),proj)
    ϕ, f, d, g = 0.08*field(), field(), field(), field()
    L = BilinearLens(ϕ)
    z = zero(f)
    for adj in (false,true)
        apply(x) = adj ? L'\x : L\x
        @test_throws ArgumentError ForwardDiff.derivative(t -> dot(g,apply(t*d)), 0.0)
        @test_throws Exception Zygote.gradient(x -> dot(g,apply(x)), z)
        objective(t) = dot(g,apply(f+t*d))
        fd = (objective(1e-4)-objective(-1e-4))/(2e-4)
        @test isapprox(ForwardDiff.derivative(objective,0.0),fd;atol=1e-6,rtol=1e-6)
        L0 = BilinearLens(zero(ϕ))
        identity(t) = dot(g,adj ? L0'\(t*d) : L0\(t*d))
        @test isapprox(ForwardDiff.derivative(identity,0.0),dot(g,d);atol=1e-11,rtol=1e-11)
    end
end
