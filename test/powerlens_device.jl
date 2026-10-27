using CMBLensing, Adapt, LinearAlgebra, Random, Test

@testset "PowerLens device application" begin
    for T in (Float32, Float64), polarized in (false, true), B in (Map, Fourier)
        Ny, Nx = 6, 8
        proj = ProjLambert(; Ny, Nx, θpix=60*180/π, T)
        potential = FlatMap(T[0.08sin(0.7i + 1.3j) + 0.05cos(1.1i - 0.4j)
                              for i in 1:Ny, j in 1:Nx], proj)
        rng = MersenneTwister(30271)
        q, u = randn(rng,T,Ny,Nx), randn(rng,T,Ny,Nx)
        field = B(polarized ? FlatQUMap(q,u,proj) : FlatMap(q,proj))
        lens = PowerLens(potential, 2)
        d1, d2 = Map(∇[1]*potential).arr, Map(∇[2]*potential).arr
        f0 = Map(field).arr
        f1, f2 = Map(∇[1]*field).arr, Map(∇[2]*field).arr
        f11 = Map(∇[1]*(∇[1]*field)).arr
        f12 = Map(∇[1]*(∇[2]*field)).arr
        f22 = Map(∇[2]*(∇[2]*field)).arr
        linear = d1 .* f1 .+ d2 .* f2
        quadratic = (d1.^2 .* f11) ./ 2 .+ d1 .* d2 .* f12 .+ (d2.^2 .* f22) ./ 2
        tol = T === Float32 ? 3e-4 : 2e-11
        gpu_lens = storage == Array ? nothing : PowerLens(adapt(storage,potential), 2)
        for sign in (-1, 1)
            op = sign == 1 ? lens : CMBLensing.antilensing(lens)
            reference = f0 .+ sign .* linear .+ quadratic
            @test isapprox(Map(op*field).arr, reference; rtol=tol, atol=tol)
            if storage != Array
                gpu_op = sign == 1 ? gpu_lens : CMBLensing.antilensing(gpu_lens)
                gpu_field = adapt(storage,field)
                @test isapprox(Array(Map(gpu_op*gpu_field).arr), reference; rtol=tol, atol=tol)
                @test isapprox(Array(Map(gpu_op'*gpu_field).arr), Map(op'*field).arr; rtol=tol, atol=tol)
            end
        end
    end
end
