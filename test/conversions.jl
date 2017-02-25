push!(LOAD_PATH, pwd()*"/src")
using Base.Test
using CMBFields

# test bases conversions

nside = 64
T = Float64
P = Flat{1,nside}

@testset "FlatS2 basis conversion" begin
    f0 = FlatS2QUMap{T,P}((rand(T,nside,nside) for i=1:2)...)
    bases = [QUMap, EBMap, QUFourier, EBFourier] 
    for B1=bases, B2=bases
        if B1!=B2
            f = B1(f0)
            @test B1(B2(f))[:] ≈ f[:]
        end
    end
end
