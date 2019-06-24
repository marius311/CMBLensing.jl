using Test
using BenchmarkTools: @belapsed

# check that the speed of various things is the same (or better) to within
# `rtol` relative tolerance of just operating on the arrays directly. 5% is
# reasonable to acount for random noise, although things could fail if e.g.
# other things start running in the background while these tests are being
# performed.
rtol = 0.05

≲(a,b; rtol=rtol) = (a < (1+rtol)*b)

N = 512

##

@testset "Algebra" begin 

# spin 0 
f = FlatMap(rand(N,N))
g = FlatMap(rand(N,N))
Ðf = Ð(f)

@test @belapsed(@.     f + f) ≲ @belapsed(@.        f.Ix + f.Ix)
@test @belapsed(@. g = f + f) ≲ @belapsed(@. g.Ix = f.Ix + f.Ix)
@test @belapsed(∇₀*Ðf)        ≲ @belapsed($(broadcastable(typeof(Ðf), ∇₀.diag)) .* Ðf.Il)

# spin 2
ft = FlatQUMap(rand(N,N),rand(N,N))
gt = FlatQUMap(rand(N,N),rand(N,N))

@test @belapsed(@.      ft + ft) ≲ @belapsed(@. (        ft.Qx + ft.Qx,         ft.Ux + ft.Ux))
@test @belapsed(@. gt = ft + ft) ≲ @belapsed(@. (gt.Qx = ft.Qx + ft.Qx, gt.Ux = ft.Ux + ft.Ux))

end

##
