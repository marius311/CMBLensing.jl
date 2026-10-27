using CMBLensing, LinearAlgebra, Test, Zygote

@testset "EquiRect block logabsdet phase gradients" begin
    # The cotangent is an adjoint wrapper; extract its stored block matrices.
    blocks(Δ) = permutedims(conj.(parent(Δ).blocks), (2,1,3))
    direction(Δ, D) = real(dot(vec(blocks(Δ)), vec(D)))

    function fixture(T, nx, complex_blocks)
        proj = ProjEquiRect(; Ny=2, Nx=nx, T, θspan=(0.4,2.3), φspan=(0,2π), storage=Array)
        E = complex_blocks ? Complex{T} : T
        A = zeros(E, 2,2,nx÷2+1)
        D = similar(A)
        a = E.([2 0.1; -0.03 1.4])
        d = E.([0.12 -0.08; 0.09 0.04])
        if complex_blocks
            a .+= im .* T[0.1 -0.07; 0.1 0.2]
            d .+= im .* T[0.31 0.14; -0.06 0.2]
        end
        for k in axes(A,3)
            A[:,:,k] .= a
            A[1,1,k] += T(0.03k)
            D[:,:,k] .= d
        end
        makeop(blocks) = BlockDiagEquiRect{AzFourier}(blocks, proj)
        A, D, makeop
    end

    for T in (Float32, Float64), nx in (5,6), complex_blocks in (false,true)
        A, D, makeop = fixture(T, nx, complex_blocks)
        op = makeop(A)
        _, sign = logabsdet(op)
        z = sum(tr(A[:,:,k] \ D[:,:,k]) for k in axes(A,3))
        h = T === Float32 ? T(0.005) : T(1e-5)
        tol = T === Float32 ? 8e-4 : 3e-8
        # The two seeds contribute independently; an imaginary magnitude seed
        # must be projected away, and a real block input has no phase tangent.
        for (Δl, Δs) in ((T(1.2), complex(T(0.7),T(-0.9))),
                         (nothing, complex(T(0),T(1))), (1.0+3.0im, 0.7-0.9im))
            _, back = Zygote.pullback(logabsdet, op)
            Δ = only(back((Δl, Δs)))
            @test Δ isa Adjoint
            @test size(parent(Δ).blocks) == size(A)
            @test eltype(parent(Δ).blocks) == eltype(A)
            α = Δl === nothing ? 0 : real(Δl)
            β = imag(conj(sign)*Δs)
            expected = α*real(z) + β*imag(z)
            seed_objective = m -> begin
                l,s = logabsdet(m)
                α*l + real(conj(Δs)*s)
            end
            fd = (seed_objective(makeop(A+h*D))-seed_objective(makeop(A-h*D)))/(2h)
            @test isapprox(direction(Δ,D), expected; rtol=tol, atol=tol)
            @test isapprox(direction(Δ,D), fd; rtol=tol, atol=tol)
        end
        for (objective, expected) in ((m->real(logdet(m)), real(z)),
                                      (m->imag(logdet(m)), imag(z)))
            Δ = only(Zygote.gradient(objective, op))
            fd = (objective(makeop(A+h*D))-objective(makeop(A-h*D)))/(2h)
            @test isapprox(direction(Δ,D), expected; rtol=tol, atol=tol)
            @test isapprox(direction(Δ,D), fd; rtol=tol, atol=tol)
        end
        # Exercise a real scalar parameter through actual operator algebra.
        dop = makeop(D)
        parameter_objective = x -> imag(logdet(op+x*dop))
        g = only(Zygote.gradient(parameter_objective, zero(T)))
        fd = (parameter_objective(h)-parameter_objective(-h))/(2h)
        @test isapprox(g, imag(z); rtol=tol, atol=tol)
        @test isapprox(g, fd; rtol=tol, atol=tol)
    end

    @testset "caches, zero and thunked seeds" begin
        for T in (Float32, Float64), complex_blocks in (false,true)
            A, _, makeop = fixture(T, 5, complex_blocks)
            op = makeop(A)
            _, back = Zygote.pullback(logabsdet, op)
            zero_seeds = (CMBLensing.ChainRulesCore.ZeroTangent(), nothing)
            @test all(iszero, blocks(only(back(zero_seeds))))
            @test !isassigned(op.blocks_pinv)
            @test back(nothing) === nothing
            sqrt(op); pinv(op); value = logabsdet(op)
            sqrt_cache, pinv_cache = op.blocks_sqrt[], op.blocks_pinv[]
            lseed, sseed = T(0.3), complex(T(0.2),T(0.7))
            reference = only(back((lseed, sseed)))
            lthunk = CMBLensing.ChainRulesCore.@thunk lseed
            sthunk = CMBLensing.ChainRulesCore.@thunk sseed
            tuplethunk = CMBLensing.ChainRulesCore.@thunk (lseed, sseed)
            @test blocks(only(back((lthunk, sthunk)))) ≈ blocks(reference)
            @test blocks(only(back(tuplethunk))) ≈ blocks(reference)
            @test op.blocks_sqrt[] === sqrt_cache
            @test op.blocks_pinv[] === pinv_cache
            @test op.logabsdet[] == value
            A[1,:,1] .= 0
            singular = makeop(A)
            value, back = Zygote.pullback(logabsdet, singular)
            @test value[1] == -Inf && iszero(value[2])
            @test all(iszero, blocks(only(back((zero(T),zero(eltype(A)))))))
            @test !isassigned(singular.blocks_pinv)
        end
    end

    @testset "higher-precision direct seeds" begin
        proj = ProjEquiRect(; Ny=2, Nx=2, T=Float32, θspan=(0.4,2.3), φspan=(0,2π), storage=Array)
        for (diagonal, seed) in ((1f-20,1e-50), (1f20,1e40)), complex_blocks in (false,true)
            E = complex_blocks ? ComplexF32 : Float32
            A = zeros(E,2,2,2)
            for k in axes(A,3), i in 1:2; A[i,i,k] = diagonal; end
            op = BlockDiagEquiRect{AzFourier}(A, proj)
            _, back = Zygote.pullback(logabsdet, op)
            for phase in (complex_blocks ? (false,true) : (false,))
                seed_pair = phase ? (0.0,seed*im) : (seed,0.0im)
                expected = phase ? E(im*seed/Float64(diagonal)) : E(seed/Float64(diagonal))
                raw = blocks(only(back(seed_pair)))
                @test eltype(raw) === E
                @test all(isfinite, raw)
                @test isapprox(raw[1,1,1], expected; rtol=5e-5, atol=0)
            end
        end
    end

    @testset "negative sign and log branch" begin
        proj = ProjEquiRect(; Ny=2, Nx=2, T=Float64, θspan=(0.4,2.3), φspan=(0,2π), storage=Array)
        A = repeat(reshape(ComplexF64[-1 0; 0 2], 2,2,1), 1,1,2)
        A[:,:,2] .= [1 0; 0 1]
        D = zeros(ComplexF64,size(A)); D[1,1,1] = im
        makeop(A) = BlockDiagEquiRect{AzFourier}(A,proj)
        phase(A) = imag(logdet(makeop(A)))
        h = 1e-6
        @test abs(phase(A+h*D)-phase(A-h*D)) > 6 # logdet is discontinuous here
        Δ = only(Zygote.gradient(m->imag(logabsdet(m)[2]), makeop(A)))
        @test isapprox(direction(Δ,D), 1.0; atol=1e-12)
        real_A = real.(A)
        real_D = zeros(size(real_A)); real_D[1,1,1] = 0.3
        real_phase = m -> imag(logdet(m))
        Δ = only(Zygote.gradient(real_phase, makeop(real_A)))
        @test eltype(parent(Δ).blocks) === Float64
        @test iszero(direction(Δ,real_D))
        @test iszero(real_phase(makeop(real_A+h*real_D))-real_phase(makeop(real_A-h*real_D)))
    end
end
