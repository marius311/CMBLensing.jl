using CMBLensing, LinearAlgebra, Test, Zygote

@testset "EquiRect adjoint block dot and logdet gradient" begin
    for T in (Float32, Float64), complex_blocks in (false, true)
        E = complex_blocks ? Complex{T} : T
        proj = ProjEquiRect(; Ny=4, Nx=4, θspan=(0.2, 2.8), φspan=(0.0, 2.6), T)
        A = reshape(E.(1:48) ./ T(13), 4, 4, 3)
        B = reshape(E.(48:-1:1) ./ T(17), 4, 4, 3)
        if complex_blocks
            A .+= im .* reshape(T.(48:-1:1) ./ T(19), 4, 4, 3)
            B .+= im .* reshape(T.(1:48) ./ T(23), 4, 4, 3)
        end
        left = BlockDiagEquiRect{AzFourier}(A, proj)
        right = BlockDiagEquiRect{AzFourier}(B, proj)
        expected = sum(dot(adjoint(A[:, :, i]), B[:, :, i]) for i in axes(A, 3))
        @test dot(left', right) ≈ expected rtol=200eps(T)

        positive = repeat(reshape(E[3 0.3 0.2 0.1; 0.3 2 0.2 0;
                                    0.2 0.2 1.4 0.1; 0.1 0 0.1 0.7], 4, 4, 1), 1, 1, 3)
        if complex_blocks
            positive[1, 2, 2] += E(0, 0.1)
            positive[2, 1, 2] -= E(0, 0.1)
        end
        op = BlockDiagEquiRect{AzFourier}(positive, proj)
        objective(a) = logabsdet(a * op)[1]
        g = only(gradient(objective, one(T)))
        h = T === Float32 ? T(0.01) : T(1e-5)
        fd = (objective(one(T) + h) - objective(one(T) - h)) / (2h)
        @test g ≈ size(op, 1) rtol=200eps(T)
        @test g ≈ fd rtol=3e-3
    end
end
