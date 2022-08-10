
using Pkg
Pkg.precompile()

t_pkgload = @time @elapsed using CMBLensing, LinearAlgebra

t_TTFL = @time @elapsed begin
    ϕ = FlatMap(randn(10,10))/1e6
    f = FlatMap(rand(10,10))
    LenseFlow(ϕ) * f
end
t_TTFG = @time @elapsed gradient((f, ϕ) -> norm(LenseFlow(ϕ) * f), f, ϕ)

using DelimitedFiles
writedlm("test_startup_$(VERSION).txt", [t_pkgload, t_TTFL, t_TTFG])