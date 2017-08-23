using CMBLensing
using CMBLensing: jrk4, cache
using BenchmarkTools

##
T = Float32
nside = 256
P = Flat{3,nside}
ϕ = FlatS0Map{T,P}(randn(nside,nside))/1e7
f = FlatIQUMap{T,P}(@repeated(randn(nside,nside),3)...)
L = LenseFlow{jrk4{7}}
##
myshow(s) = (println("== "*s*" =="); t->(show(STDOUT,MIME("text/plain"),t); println()))
##
(@benchmark $(cache(L(ϕ))) * $f) |> myshow("LenseFlow")
(@benchmark $f * $(cache(L(ϕ)))) |> myshow("TransposeFlow")
