push!(LOAD_PATH, pwd()*"/src")
using Base.Test
using CMBFields

# test bases conversions

nside = 64
T = Float64
P = Flat{1,nside}

.≈(a,b) = broadcast((a,b)->(≈(a,b; atol=1e-6)),a,b)

@testset "FlatS2 basis conversion" begin
    bases = [QUMap, EBMap, QUFourier, EBFourier] 
    for B1=bases, B2=bases
        if B1!=B2
            @eval @test begin 
                f = FlatS2QUMap{T,P}((rand(T,nside,nside) for i=1:2)...)
                all($B1($B2($B1(f)))[:] .≈ $B1(f)[:])
            end
        end
    end
end

##
using PyPlot
g = FFTgrid(Float64,Flat{1,256})
k = g.k
ϕ = angle.(k' .+ im*k)
Qx, Ux = rand(g.nside,g.nside), rand(g.nside,g.nside)/100
Ql, Ul = fft(Qx), fft(Ux)
El = - Ql .* cos(2ϕ) - Ul .* sin(2ϕ)
Bl =   Ql .* sin(2ϕ) - Ul .* cos(2ϕ)
real(ifft(El)) |> matshow
colorbar()
imag(ifft(El)) |> matshow
colorbar()
El[1:g.nside÷2+1,:]
##
