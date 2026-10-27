using CMBLensing, Adapt, LinearAlgebra, Test

@testset "EquiRect block factorizations with array views" begin
    for T in (Float32, Float64)
        n, nx = 2, 4
        p = ProjEquiRect(; Ny=n, Nx=nx, T, θspan=(0.4,2.3), φspan=(0,2π))
        host = zeros(T,2n+1,2n+1,nx÷2+1)
        for k in axes(host,3), j in 1:n
            host[2j,2j,k] = T(j+k)
        end
        backing = adapt(storage, host)
        stepped = view(backing,2:2:2n,2:2:2n,:)
        nested = reshape(reshape(stepped,n*n,:),n,n,:)
        permuted = PermutedDimsArray(stepped,(2,1,3))
        @test stepped isa SubArray
        @test nested isa Base.ReshapedArray && parent(nested) isa SubArray
        @test permuted isa PermutedDimsArray && parent(permuted) isa SubArray
        layouts = (copy(stepped), stepped, permuted, nested)
        for blocks in layouts, operation in (pinv,sqrt)
            op = BlockDiagEquiRect{AzFourier}(blocks, adapt(storage,p))
            expected = cat((operation(Array(blocks)[:,:,k]) for k in axes(blocks,3))...; dims=3)
            first = operation(op)
            second = operation(op)
            @test Array(first.blocks) ≈ expected rtol=1e-5 atol=1e-6
            @test Array(second.blocks) ≈ expected rtol=1e-5 atol=1e-6
            if blocks === layouts[1]
                # Ordinary dense storage retains the cached factorization.
                @test first.blocks === second.blocks
            end
        end
    end
end
