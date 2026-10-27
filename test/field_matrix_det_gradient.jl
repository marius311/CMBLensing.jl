using CMBLensing, FFTW, LinearAlgebra, Random, Test, Zygote
using CMBLensing: @SMatrix, SMatrix

@testset "2×2 field-operator determinant gradients" begin
    for T in (Float32, Float64), shape in ((4,4), (4,6)), B in (Map, Fourier)
        rng = MersenneTwister(8133)
        proj = ProjLambert(; Ny=shape[1], Nx=shape[2], T, θpix=3)
        field() = B(FlatMap(randn(rng,T,shape...), proj))
        a, b, c, d = field(), field(), field(), field()
        da, db, dc, dd = field(), field(), field(), field()
        matrix(a,b,c,d) = @SMatrix [Diagonal(a) Diagonal(b); Diagonal(c) Diagonal(d)]
        objective(a,b,c,d) = sum(Map(det(matrix(a,b,c,d)).diag))
        objective_t(t) = objective(a+t*da, b+t*db, c+t*dc, d+t*dd)
        h = T === Float32 ? T(1e-3) : T(1e-5)
        tol = T === Float32 ? 3e-3 : 3e-7
        g = gradient(objective, a, b, c, d)
        reverse = sum(dot(x,y) for (x,y) in zip(g, (da,db,dc,dd)))
        finite = (objective_t(h)-objective_t(-h))/(2h)
        @test isapprox(reverse, finite; rtol=tol, atol=tol)
        @test isapprox(gradient(objective_t, T(0))[1], finite; rtol=tol, atol=tol)
    end
    # Mixed element precision needs an explicit common field-operator type.
    for B in (Map, Fourier)
        rng = MersenneTwister(20581)
        p32 = ProjLambert(; Ny=4, Nx=6, θpix=60*180/π, T=Float32)
        p64 = ProjLambert(; Ny=4, Nx=6, θpix=60*180/π, T=Float64)
        field32() = B(FlatMap(randn(rng, Float32, 4, 6), p32))
        field64() = B(FlatMap(randn(rng, Float64, 4, 6), p64))
        a,b,c,d = field32(),field64(),field64(),field32()
        da,db,dc,dd = field32(),field64(),field64(),field32()
        weight = field64()
        matrix(a,b,c,d) = SMatrix{2,2,CMBLensing.DiagOp}((Diagonal(a),Diagonal(c),Diagonal(b),Diagonal(d)))
        objective(a,b,c,d) = dot(weight,diag(det(matrix(a,b,c,d))))
        gradients = gradient(objective,a,b,c,d)
        reverse = sum(dot(x,y) for (x,y) in zip(gradients,(da,db,dc,dd)))
        analytic = dot(weight,da .* d .+ a .* dd .- db .* c .- b .* dc)
        @test isapprox(reverse,analytic;atol=2e-5,rtol=2e-5)
        @test map(eltype,gradients) == map(eltype,(a,b,c,d))
    end
    # Check arbitrary complex coefficients and heterogeneous component fields,
    # using a nonuniform weight so cancellation cannot hide incorrect gradients.
    let
        rng = MersenneTwister(617033)
        proj = ProjLambert(; Ny=6, Nx=8, θpix=60*180/π, T=Float64)
        scalar() = FlatMap(randn(rng, 6, 8), proj)
        complexfourier() = FlatFourier(randn(rng, ComplexF64, 4, 8), proj)
        tuplefield() = FieldTuple(; map=scalar(), fourier=complexfourier())
        matrix(a,b,c,d) = @SMatrix [Diagonal(a) Diagonal(b); Diagonal(c) Diagonal(d)]
        for field in (complexfourier, tuplefield)
            a,b,c,d = field(),field(),field(),field()
            da,db,dc,dd = field(),field(),field(),field()
            weight = field()
            objective(a,b,c,d) = dot(weight, diag(det(matrix(a,b,c,d))))
            objective_t(t) = objective(a+t*da,b+t*db,c+t*dc,d+t*dd)
            gradients = gradient(objective,a,b,c,d)
            reverse = sum(dot(x,y) for (x,y) in zip(gradients,(da,db,dc,dd)))
            h = 1e-5
            finite = (objective_t(h)-objective_t(-h))/(2h)
            @test isapprox(reverse,finite;atol=3e-7,rtol=3e-7)
            @test isapprox(gradient(objective_t,0.0)[1],finite;atol=3e-7,rtol=3e-7)
            @test diag(det(matrix(a,b,c,d))) ≈ diag(Diagonal(a)*Diagonal(d)-Diagonal(b)*Diagonal(c))
        end
    end
    # Different bases represent operator compositions, not pointwise products.
    let
        rng = MersenneTwister(967)
        mapfield() = FlatMap(randn(rng, 4, 6))
        fourierfield() = Fourier(mapfield())
        a,b,c,d = Diagonal(mapfield()),Diagonal(mapfield()),Diagonal(fourierfield()),Diagonal(fourierfield())
        # Keep the abstract element type specific enough to select the
        # field-operator determinant method for heterogeneous bases.
        matrix = SMatrix{2,2,CMBLensing.DiagOp}((a,c,b,d))
        f = mapfield()
        result = det(matrix)
        @test result isa CMBLensing.LazyBinaryOp
        @test result*f ≈ (a*d-b*c)*f
    end
    # Batched fields retain per-batch sensitivities in parts of the existing
    # AD system. Reduce them once before comparing a scalar directional FD.
    for B in (Map, Fourier)
        B === Fourier && FFTW.fftw_provider == "mkl" && continue
        rng = MersenneTwister(28893)
        field() = B(FlatMap(randn(rng,4,6,2)))
        a,b,c,d = field(),field(),field(),field()
        da,db,dc,dd = field(),field(),field(),field()
        matrix(a,b,c,d) = @SMatrix [Diagonal(a) Diagonal(b); Diagonal(c) Diagonal(d)]
        reduce_batches(x) = x isa CMBLensing.BatchedReal ? sum(unbatch(x)) : x
        objective(a,b,c,d) = reduce_batches(sum(Map(det(matrix(a,b,c,d)).diag)))
        objective_t(t) = objective(a+t*da,b+t*db,c+t*dc,d+t*dd)
        gradients = gradient(objective,a,b,c,d)
        reverse = sum(reduce_batches(dot(x,y)) for (x,y) in zip(gradients,(da,db,dc,dd)))
        h=1e-5
        finite = (objective_t(h)-objective_t(-h))/(2h)
        @test isapprox(reverse,finite;atol=3e-7,rtol=3e-7)
        @test isapprox(reduce_batches(gradient(objective_t,0.0)[1]),finite;atol=3e-7,rtol=3e-7)
    end
    pixels=FlatMap(ones(4,4))
    matrix_constant(a,b,c,d)=@SMatrix [Diagonal(a*pixels) Diagonal(b*pixels); Diagonal(c*pixels) Diagonal(d*pixels)]
    f(a,b,c,d)=sum(det(matrix_constant(a,b,c,d)).diag)
    @test gradient(f,2.0,3.0,1.0,4.0) == (64.0,-16.0,-48.0,32.0)
end
