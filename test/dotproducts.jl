include("general.jl")


@testset "dot products" begin
    
    for nside = [64,65]
        g = FFTgrid(Float64,Flat{1,nside})

        f0, f1 = (FlatS0Map(randn(nside,nside),1) for i=1:2)
        @test f0⋅f1 ≈ Fourier(f0)⋅Fourier(f1) ≈ f0⋅Fourier(f1) ≈ f0.Tx⋅f1.Tx * g.Δx^2 
        
        f0, f1 = (FlatS2QUMap(randn(nside,nside),randn(nside,nside),1) for i=1:2)
        @test f0⋅f1 ≈ EBFourier(f0)⋅EBFourier(f1) ≈ QUFourier(f0)⋅QUFourier(f1) ≈ f0⋅EBFourier(f1) ≈ f0⋅QUFourier(f1)
        if iseven(nside)
            @test_broken f0⋅f1 ≈ EBMap(f0)⋅EBMap(f1) # EB-QU nonexact conversion
        else
            @test f0⋅f1 ≈ EBMap(f0)⋅EBMap(f1)
        end
    end

end
