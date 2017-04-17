push!(LOAD_PATH, pwd()*"/src")
using CMBLensing: FlatS0Map
using BenchmarkTools
##
m = rand(2000,2000)
f = FlatS0Map(m)
##
myshow(s) = (println(s); t->(show(STDOUT,MIME("text/plain"),t); println()))
(@benchmark @. 3 * $f^2 + 4 * $f^3) |> myshow("==Broadcasted fields==")
(@benchmark    3 * $f^2 + 4 * $f^3) |> myshow("==Non-broadcasted fields==")
(@benchmark @. 3 * $m^2 + 4 * $m^3) |> myshow("==Matrices==")
