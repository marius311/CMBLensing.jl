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
        mixed_tuple = FieldTuple(; scalar=z, polar=qu)
        otherproj = ProjLambert(; Ny=Ny+1, Nx, θpix=60*180/π, T)
        wrongzero = maybegpu(FlatMap(zeros(T,Ny+1,Nx),otherproj))
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
            tuple = apply(mixed_tuple)
            @test all(iszero,Array(zero_scalar.arr))
            @test eltype(zero_scalar.arr) === T
            @test all(iszero,Array(mixed_qu.Qx))
            @test isapprox(Array(mixed_qu.Ux),Array(scalar.Ix);atol=tol,rtol=tol)
            @test all(iszero,Array(mixed.arr[:,:,1,1]))
            @test isapprox(Array(mixed.arr[:,:,1,2]),Array(scalar.Ix);atol=tol,rtol=tol)
            @test all(iszero,Array(tuple.scalar.arr))
            @test isapprox(Array(tuple.polar.Ux),Array(scalar.Ix);atol=tol,rtol=tol)
            @test_throws DimensionMismatch apply(wrongzero)
        end
        @test Array(f.arr) == original_f
        @test Array(qu.arr) == original_qu
    end
end

@testset "BilinearLens zero field precision and differentiation" begin
    # Even when their action on the RHS is defined, rectangular matrices do
    # not fit the existing solver's square, RHS-sized output buffers.
    @test_throws DimensionMismatch CMBLensing._bilinear_inverse_gmres(ones(7,8),zeros(8),I)
    @test_throws DimensionMismatch CMBLensing._bilinear_inverse_gmres(ones(8,7),zeros(8),ones(7,8))
    @test_throws DimensionMismatch CMBLensing._bilinear_inverse_gmres(ones(8,8),zeros(8),ones(7,8))
    @test_throws DimensionMismatch ForwardDiff.derivative(
        t -> sum(CMBLensing._bilinear_inverse_gmres(ones(7,8),t*ones(8),I)), 0.0)

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
        for T in (Float32,Float64,ComplexF32,ComplexF64)
            mixed_zero = FlatMap(zeros(T,6,8),proj)
            output = apply(mixed_zero)
            @test eltype(output.arr) === T
            @test all(iszero,output.arr)
        end
        # Zero fields can have nonzero ForwardDiff partials. They do not have
        # a general linear derivative for this truncated solver.
        @test_throws ArgumentError ForwardDiff.derivative(t -> dot(g,apply(t*d)), 0.0)
        cd = FlatMap(complex.(d.Ix,reverse(d.Ix;dims=1)),proj)
        @test_throws ArgumentError ForwardDiff.derivative(t -> real(dot(g,apply(t*cd))), 0.0)
        @test_throws Exception Zygote.gradient(x -> dot(g,apply(x)), z)
        objective(t) = dot(g,apply(f+t*d))
        fd = (objective(1e-4)-objective(-1e-4))/(2e-4)
        @test isapprox(ForwardDiff.derivative(objective,0.0),fd;atol=1e-6,rtol=1e-6)
        L0 = BilinearLens(zero(ϕ))
        identity(t) = dot(g,adj ? L0'\(t*d) : L0\(t*d))
        @test isapprox(ForwardDiff.derivative(identity,0.0),dot(g,d);atol=1e-11,rtol=1e-11)
    end

    # A zero field must not mask an invalid lens or preconditioner. Their
    # unmodified solve produces a nonfinite result or a factorization error.
    for adj in (false,true), which in (:lens,:preconditioner)
        bad = BilinearLens(ϕ)
        A = which === :lens ? bad.sparse_repr : CMBLensing.get_anti_lensing_sparse_repr!(bad)
        A.nzval[1] = NaN
        output = try
            adj ? bad'\z : bad\z
        catch
            nothing
        end
        @test output === nothing || any(!isfinite,output.arr)
    end
end
