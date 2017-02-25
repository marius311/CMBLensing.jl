push!(LOAD_PATH, pwd()*"/src")
using Base.Test
using CMBFields

@testset "dot products" begin

    nside = 64
    g = FFTgrid(Float64,Flat{1,nside})

    ##
    f0, f1 = (FlatS0Map(randn(nside,nside),1) for i=1:2)
    @test f0⋅f1 ≈ Fourier(f0)⋅Fourier(f1) ≈ f0⋅Fourier(f1) ≈ f0.Tx⋅f1.Tx * g.Δx^2 
    ##
    f0, f1 = (FlatS2QUMap(randn(nside,nside),randn(nside,nside),1) for i=1:2)
    @test f0⋅f1 ≈ EBFourier(f0)⋅EBFourier(f1) ≈ QUFourier(f0)⋅QUFourier(f1) ≈ f0⋅EBFourier(f1) ≈ f0⋅QUFourier(f1)
    @test_broken f0⋅f0 ≈ EBMap(f0)⋅EBMap(f0)
    ##

end
