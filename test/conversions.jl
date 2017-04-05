push!(LOAD_PATH, pwd()*"/src")
using Base.Test
import Base: ≈
using CMBFields
##

# test bases conversions

nside = 65
T = Float64
P = Flat{1,nside}

≈(a::Field,b::Field) = all(broadcast((a,b)->(≈(a,b; atol=1e-6)),a[:],b[:]))

@testset "FlatS2 basis conversion" begin
    bases = [QUMap, EBMap, QUFourier, EBFourier] 
    for B1=bases, B2=bases
        if B1!=B2
            @eval @test begin 
                f = FlatS2QUMap{T,P}((rand(T,nside,nside) for i=1:2)...)
                $B1($B2($B1(f))) ≈ $B1(f)
            end
        end
    end
end

##
# g = FFTgrid(Float64,Flat{1,4})
# k = g.k
# ϕ = angle.(k' .+ im*k)
# Qx, Ux = randn(g.nside,g.nside), randn(g.nside,g.nside)
# Ql, Ul = fft(Qx), fft(Ux)
# Ql[3,:] = Ul[3,:] = Ql[:,3] = Ul[:,3] = 0
# El = - Ql .* cos(2ϕ) - Ul .* sin(2ϕ)
# Bl =   Ql .* sin(2ϕ) - Ul .* cos(2ϕ)
# ifft(Bl)
##
