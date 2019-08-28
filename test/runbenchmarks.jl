using CMBLensing

##

using Test
using BenchmarkTools: @belapsed, @btime

# check that the speed of various things is the same (or better) to within
# `rtol` relative tolerance of just operating on the arrays directly. 5% is
# reasonable to acount for random noise, although things could fail if e.g.
# other things start running in the background while these tests are being
# performed.
rtol = 0.05

≲(a,b; rtol=rtol) = (a < (1+rtol)*b)

N = 256

##

# broadcasting with Fields should be exactly as fast working with the arrays
# directly, no overhead allowed

@testset "Algebra" begin 

# spin 0 
f = FlatMap(rand(N,N))
g = FlatMap(rand(N,N))
Ðf = Ð(f)
@test @belapsed(@.     f + f) ≲ @belapsed(@.        f.Ix + f.Ix)
@test @belapsed(@. g = f + f) ≲ @belapsed(@. g.Ix = f.Ix + f.Ix)
@test @belapsed(∇[1]*Ðf)      ≲ @belapsed($(broadcastable(typeof(Ðf), ∇[1].diag)) .* Ðf.Il)

# spin 2
ft = FlatQUMap(rand(N,N),rand(N,N))
gt = FlatQUMap(rand(N,N),rand(N,N))
@test @belapsed(@.      ft + ft) ≲ @belapsed(@. (        ft.Qx + ft.Qx,         ft.Ux + ft.Ux))
@test @belapsed(@. gt = ft + ft) ≲ @belapsed(@. (gt.Qx = ft.Qx + ft.Qx, gt.Ux = ft.Ux + ft.Ux))

end

##

# these represent benchmarks on my laptop and may not necessarily pass on other
# systems, but its a way to catch performance regressions 

@testset "Lensing" begin
    
local f, ϕ, Lϕ, Cℓ

Cℓ = camb().unlensed_total

# spin 0
f = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S0, Cℓ.TT))
ϕ = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S0, Cℓ.ϕϕ))
Lϕ = cache(LenseFlow(ϕ),f)
@test @belapsed(Lϕ*f) ≲ 13e-3
@test @belapsed(Lϕ'*f) ≲ 13e-3
@test @belapsed(δf̃ϕ_δfϕ(Lϕ,f,f)'*FΦTuple(f,ϕ)) ≲ 80e-3

# spin 2
f = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S2, Cℓ.EE, Cℓ.BB))
ϕ = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S0, Cℓ.ϕϕ))
Lϕ = cache(LenseFlow(ϕ),f)
@test @belapsed(Lϕ*f) ≲ 30e-3
@test @belapsed(Lϕ'*f) ≲ 30e-3
@test @belapsed(δf̃ϕ_δfϕ(Lϕ,f,f)'*FΦTuple(f,ϕ)) ≲ 140e-3

end
