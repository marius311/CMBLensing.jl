using CMBFields
using PyPlot

## calc Cℓs and store in Main since I reload CMBFields alot during development
cls = isdefined(Main,:cls) ? Main.cls : @eval Main cls=$(class(lmax=6000,r=0.05))
## set up the types of maps
Θpix, nside, T = 3, 75, Float64
P = Flat{Θpix,nside}
## covariances 
Cf    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],   cls[:te],   cls[:ee],   cls[:bb])
Cϕ    = Cℓ_to_cov(T,P,S0,   cls[:ℓ],cls[:ϕϕ])
##
f,f1,f2 = @repeated(Ł(simulate(Cf)),3)
ϕ,ϕ1,ϕ2 = @repeated(Ł(simulate(Cϕ)),3)
δf = simulate(Cf) |> Ł
δϕ = simulate(Cϕ) |> Ł
##
ϵ = 1e-5
L = ϕ->PowerLens(ϕ,order=2)
##

plotdiff(x) = plot([x[1],x[2],x[1]-x[2]])

## check gradients
[1/(2ϵ)*(L(ϕ+ϵ*δϕ)*f - L(ϕ-ϵ*δϕ)*f) , δf̃_δϕ(L(ϕ),f)*δϕ] |> plotdiff
##
[1/(2ϵ)*(L(ϕ)*(f+ϵ*δf) - L(ϕ)*(f-ϵ*δf)) , δf̃_δf(L(ϕ),f)*δf] |> plotdiff
## check transposes
f1 ⋅ (δf̃_δϕ(L(ϕ),f)*ϕ2), (f1*δf̃_δϕ(L(ϕ),f)) ⋅ ϕ2
##
f1 ⋅ (δf̃_δf(L(ϕ),f)*f2), (f1*δf̃_δf(L(ϕ),f)) ⋅ f2
##

## check second derivatives
[1/(2ϵ)*(f1*δf̃_δϕ(L(ϕ+ϵ*δϕ),f) - f1*δf̃_δϕ(L(ϕ-ϵ*δϕ),f)) , δ²f̃_δϕ²(L(ϕ),f,f1,δϕ)] |> plotdiff
##
[1/(2ϵ)*(f1*δf̃_δϕ(L(ϕ),f+ϵ*δf) - f1*δf̃_δϕ(L(ϕ),f-ϵ*δf)) , δ²f̃_δϕδf(L(ϕ),f,f1,δf)] |> plotdiff
##
[1/(2ϵ)*(δf̃_δϕ(L(ϕ),f+ϵ*δf)*ϕ1 - δf̃_δϕ(L(ϕ),f-ϵ*δf)*ϕ1) , δ²f̃_δfδϕ(L(ϕ),f,ϕ1,δf)] |> plotdiff
