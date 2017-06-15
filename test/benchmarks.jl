using CMBLensing
using CMBLensing: ode4
using BenchmarkTools

##
T = Float32
nside = 256
P = Flat{3,nside}
ϕ = FlatS0Map{T,P}(randn(nside,nside))/1e7
f = FlatIQUMap{T,P}(@repeated(randn(nside,nside),3)...)
L = LenseFlow{ode4{7}}
##
myshow(s) = (println("== "*s*" =="); t->(show(STDOUT,MIME("text/plain"),t); println()))
##
(@benchmark $(L(ϕ)) * $f) |> myshow("LenseFlow")
(@benchmark $f * $(L(ϕ))) |> myshow("TransposeFlow")
