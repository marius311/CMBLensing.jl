using CMBLensing, FFTW, LinearAlgebra, Random, Test, Zygote

@testset "Lambert real/Fourier conversion adjoints" begin
    for T in (Float32, Float64), (Ny, Nx) in ((6,8), (7,8), (7,9))
        rng = MersenneTwister(26013 + Ny + Nx)
        proj = ProjLambert(; Ny, Nx, θpix=60*180/π, T)
        mapfield() = FlatMap(randn(rng, T, Ny, Nx), proj)
        x, η, q = mapfield(), mapfield(), mapfield()
        xf = Fourier(x)
        direction = zeros(Complex{T}, Ny÷2+1, Nx)
        direction[1,2], direction[1,Nx] = complex(T(0.4),T(0.7)), complex(T(-0.2),T(0.3))
        direction[end,2], direction[end,Nx] = complex(T(0.6),T(-0.5)), complex(T(-0.1),T(-0.8))
        δ = FlatFourier(direction, proj)
        weight = zeros(Complex{T}, Ny÷2+1, Nx)
        weight[1,1], weight[1,2], weight[1,Nx] = complex(zero(T),T(0.6)), complex(T(1),T(0.4)), complex(T(-0.3),T(0.8))
        weight[end,2], weight[end,Nx] = complex(T(0.2),T(-0.5)), complex(T(-0.4),T(-0.6))
        w = FlatFourier(weight, proj)
        tol = T === Float32 ? 3e-5 : 2e-12
        h = T === Float32 ? T(1e-3) : T(1e-5)
        ftol = T === Float32 ? 4e-3 : 2e-8

        # Real-map conversion of a non-Hermitian Fourier direction followed
        # by a spectral derivative: check the transpose against the actual
        # linear forward operation and a central difference.
        objective(z) = dot(q, Map(∇[2]*z))
        exact = dot(q, Map(∇[2]*δ))
        reverse = dot(gradient(objective, xf)[1], δ)
        finite = (objective(xf+h*δ)-objective(xf-h*δ))/(2h)
        @test isapprox(reverse, exact; atol=tol, rtol=tol)
        @test isapprox(finite, exact; atol=ftol, rtol=ftol)

        # The forward transform's adjoint must also work for an arbitrary
        # complex cotangent; dot with unlike bases has its own pullback.
        exact = dot(w, Fourier(η))
        for f in (z -> dot(w, Fourier(z)), z -> dot(w,z), z -> dot(z,w))
            reverse = dot(gradient(f,x)[1], η)
            finite = (f(x+h*η)-f(x-h*η))/(2h)
            @test isapprox(reverse, exact; atol=tol, rtol=tol)
            @test isapprox(finite, exact; atol=ftol, rtol=ftol)
        end
    end

    # Wider projection metadata and a mixed-precision field pair keep the
    # existing cotangent precision and still obey the directional identity.
    let
        rng = MersenneTwister(73910)
        proj = ProjLambert(; Ny=6, Nx=8, θpix=60*180/π, T=Float64)
        xarr = randn(rng,Float32,6,8)
        warr = randn(rng,ComplexF64,4,8)
        xbefore, wbefore = copy(xarr), copy(warr)
        x = FlatMap(view(xarr,:,:),proj)
        η = FlatMap(randn(rng,Float32,6,8),proj)
        w = FlatFourier(view(warr,:,:),proj)
        gx,gw = gradient((a,b)->dot(a,b),x,w)
        @test eltype(gx) === Float64
        @test eltype(gw) === ComplexF64
        @test isapprox(dot(gx,η),dot(w,Fourier(η));atol=3e-5,rtol=3e-5)
        @test xarr == xbefore
        @test warr == wbefore
    end
end
