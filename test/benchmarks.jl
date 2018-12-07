using CMBLensing
using CMBLensing: jrk4, cache, @dictpack, fourier∂
using BenchmarkTools

##
T = Float32
nside = 256
P = Flat{3,nside,fourier∂}
ϕ = FlatS0Map{T,P}(randn(nside,nside))/1e7
f = FlatS0Map{T,P}(@repeated(randn(nside,nside),1)...)
L = cache(LenseFlow{jrk4{7}}(ϕ),f)
J = δf̃ϕ_δfϕ(L,L*f,f)
fϕ = FieldTuple(f,ϕ)
# some fake data and covariances to time the likelihood gradient
d = f
Cn,Cf,Cf̃ = @repeated(FullDiagOp(FlatS0Map{T,P}(@repeated(rand(nside,nside),1)...)),3)
Cϕ = FullDiagOp(FlatS0Map{T,P}(rand(nside,nside))/1e7)
ds = DataSet(;@dictpack(d,Cn,Cf,Cf̃,Cϕ)...)
##
myshow(s) = (printstyled("=== "*s*" ===\n",color=:light_green); t->(show(stdout,MIME("text/plain"),t); println()))
##
@benchmark(cache(LenseFlow{jrk4{7}}($ϕ),$f)) |> myshow("Caching")
@benchmark($L    * $f)                       |> myshow("Lense")
@benchmark($(L') * $f)                       |> myshow("Transpose Lense")
@benchmark($(J') * $fϕ)                      |> myshow("Transpose Jacobian")
@benchmark(δlnP_δfϕₜ(:mix, $f, $ϕ, $ds, $L)) |> myshow("Mixed posterior gradient")
