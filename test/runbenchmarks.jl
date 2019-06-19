using Test
using BenchmarkTools: @belapsed

# check that the speed of various things is the same (or better) to within
# `rtol` relative tolerance of just operating on the arrays directly. 5% is
# reasonable to acount for random noise, although things could fail if e.g.
# other things start running in the background while these tests are being
# performed.
rtol = 0.05

≲(a,b; rtol=rtol) = (a < (1+rtol)*b)

##

@testset "Algebra" begin 

# spin 0 
N = 512
f = FlatMap(rand(N,N))
g = FlatMap(rand(N,N))

add(f)   = (f .+ f)
add!(g,f) = (g .= f .+ f)

@test @belapsed(add(f))    ≲ @belapsed(add(f.Ix))
@test @belapsed(add!(g,f)) ≲ @belapsed(add!(g.Ix,f.Ix))

# spin 2
ft = FlatQUMap(rand(N,N),rand(N,N))
gt = FlatQUMap(rand(N,N),rand(N,N))

@test @belapsed(add(ft))     ≲ @belapsed((add(ft.Qx);add(ft.Ux)))
@test @belapsed(add!(gt,ft)) ≲ @belapsed((add!(gt.Qx,ft.Qx);add!(gt.Ux,ft.Ux)))

end

##
