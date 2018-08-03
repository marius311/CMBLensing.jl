using CMBLensing
using CMBLensing: jrk4, cache, @dictpack
using BenchmarkTools

##
T = Float32
nside = 256
P = Flat{3,nside}
ϕ = FlatS0Map{T,P}(randn(nside,nside))/1e7
f = FlatS2QUMap{T,P}(@repeated(randn(nside,nside),2)...)
L = LenseFlow{jrk4{7}}(ϕ)
cL = cache(L)
J = δf̃ϕ_δfϕ(L,L*f,f)
cJ = δf̃ϕ_δfϕ(cL,cL*f,f)
fϕ = FieldTuple(f,ϕ)
# some fake data and covariances to time the likelihood gradient
# d = f
# Cn,Cf,Cf̃ = @repeated(FullDiagOp(FlatS2QUMap{T,P}(@repeated(randn(nside,nside),2)...)),3)
# Cϕ = FullDiagOp(FlatS0Map{T,P}(randn(nside,nside))/1e7)
# ds = DataSet(;@dictpack(d,Cn,Cf,Cf̃,Cϕ)...)
##
myshow(s) = (print_with_color(:light_green,"=== "*s*" ===\n"); t->(show(STDOUT,MIME("text/plain"),t); println()))
##
@benchmark($L * $f)   |> myshow("Lense")
@benchmark($f * $L)   |> myshow("Transpose Lense")
@benchmark($J * $fϕ)  |> myshow("Jacobian")
@benchmark($fϕ * $J)  |> myshow("Transpose Jacobian")
##
@benchmark($cL * $f)  |> myshow("Cached Lense")
@benchmark($f * $cL)  |> myshow("Cached Transpose Lense")
@benchmark($cJ * $fϕ) |> myshow("Cached Jacobian")
@benchmark($fϕ * $cJ) |> myshow("Cached Transpose Jacobian")
##
@benchmark(δlnP_δfϕₜ(:mix, $f, $ϕ, $ds, $cL)) |> myshow("Cached mixed posterior gradient")
