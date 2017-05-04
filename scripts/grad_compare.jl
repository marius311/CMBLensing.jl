using CMBLensing
using PyPlot

## calc Cℓs and store in Main since I reload CMBLensing alot during development
cls = isdefined(Main,:cls) ? Main.cls : @eval Main cls=$(class(lmax=6000,r=0.05))
## set up the types of maps
Θpix, nside, T = 3, 64, Float64
P = Flat{Θpix,nside}
## covariances 
# Cf    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],   cls[:te],   cls[:ee],   cls[:bb])
# Cf̃    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:ln_tt],cls[:ln_te],cls[:ln_ee],cls[:ln_bb])
Cf    = Cℓ_to_cov(T,P,S2,cls[:ℓ], cls[:ee],   cls[:bb])
Cf̃    = Cℓ_to_cov(T,P,S2,cls[:ℓ], cls[:ln_ee],cls[:ln_bb])
Cϕ    = Cℓ_to_cov(T,P,S0,   cls[:ℓ],cls[:ϕϕ])
μKarcminT = 0.001
Ωpix = deg2rad(Θpix/60)^2
CN  = FullDiagOp(FlatS2QUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),2)...))
CN̂  = FullDiagOp(FlatS2EBFourier{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside÷2+1,nside)),2)...))
# CN  = FullDiagOp(FlatIQUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),3)...))
# CN̂ = Cℓ_to_cov(T,P,S0,S2, 0:6000, repeated(μKarcminT^2 * Ωpix * ones(6001),4)...)
## masks
ℓmax_mask, Δℓ_taper = 3000, 500
Ml = [ones(ℓmax_mask); (cos(linspace(0,π,Δℓ_taper))+1)/2]
# Mf = FullDiagOp(FlatS2QUFourier{T,P}(@repeated(Cℓ_to_cov(T,P,S0,1:(ℓmax_mask+Δℓ_taper),Ml).f[:Tl],2)...)) * Squash
Mϕ = Mf = Squash
## generate simulated datasets
ϕ₀ = simulate(Cϕ)
f₀ = simulate(Cf)
n₀ = simulate(CN)
L_lf = LenseFlow(ϕ₀)
L_pl = PowerLens(ϕ₀)
d_lf = L_lf*f₀ + n₀
d_pl = L_pl*f₀ + n₀
ds_pl = DataSet(d_pl,CN,Cf,Cϕ,Mf,Mϕ);
ds_lf = DataSet(d_lf,CN,Cf,Cϕ,Mf,Mϕ);


#### check accuracy of likelihood and derivatives for the two algorithms
using Base.Test
ϵ = 1e-8
δϕ = simulate(Cϕ)
δf = simulate(Cf)
## likelihoood evaluated with PowerLens at t=0 and with LenseFlow at t=0 and t=1
((@inferred lnP(0,f₀,ϕ₀,ds_pl,PowerLens)), 
 (@inferred lnP(0,f₀,ϕ₀,ds_lf,LenseFlow)),
 (@inferred lnP(1,L_lf*f₀,ϕ₀,ds_lf,LenseFlow)))
## PowerLens gradient at t=0
(@inferred δlnP_δfϕₜ(0,f₀,ϕ₀,ds_pl,PowerLens)⋅FieldTuple(δf,δϕ)), (lnP(0,f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_pl,PowerLens) - lnP(0,f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_pl,PowerLens))/(2ϵ)
## LenseFlow gradient at t=0
(@inferred δlnP_δfϕₜ(0,f₀,ϕ₀,ds_lf,LenseFlow)⋅FieldTuple(δf,δϕ)), (lnP(0,f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_lf,LenseFlow) - lnP(0,f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_lf,LenseFlow))/(2ϵ)
## LenseFlow gradient at t=1
(@inferred δlnP_δfϕₜ(1,L_lf*f₀,ϕ₀,ds_lf,LenseFlow)⋅FieldTuple(δf,δϕ)), (lnP(1,L_lf*f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_lf,LenseFlow) - lnP(1,L_lf*f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_lf,LenseFlow))/(2ϵ)
##

### 
using Optim
using Optim: x_trace
##
(fstart,ϕstart) = fϕ_start = Ł(FieldTuple(𝕎(Cf̃,CN̂)*d_lf,0ϕ₀))
FΦ = typeof(fϕ_start)
Hinv = FullDiagOp(FieldTuple(Mf*(@. (CN̂^-1 + Cf^-1)^-1).f, Mϕ*Cϕ.f))
Δx² = FFTgrid(T,P).Δx^2

##
import Base.LinAlg.A_ldiv_B!
struct foo end
A_ldiv_B!(s,::foo,q) = (s.=FΦ(Hinv*q[~fϕ_start])[:])
##
res = optimize(
    x->(println(1); -lnP(1,x[~fϕ_start]...,ds_lf)),
    (x,∇lnP)->(println(2); ∇lnP .= -Δx²*FΦ(FieldTuple(δlnP_δfϕₜ(1,x[~fϕ_start]...,ds_lf)...))[:]),
    fϕ_start[:],
    LBFGS(P=foo()),
    Optim.Options(time_limit = 60.0, store_trace=true, show_trace=true))
##
fϕ_start = res.minimizer

res = optimize(
    x->(println(1); -lnP(0,x[~fϕ_start]...,ds_lf)),
    (x,∇lnP)->(println(2); ∇lnP .= -Δx²*FΦ(FieldTuple(δlnP_δfₜϕ(0,x[~fϕ_start]...,ds_lf)...))[:]),
    fϕ_start[:],
    LBFGS(P=foo()),
    Optim.Options(time_limit = 60.0, store_trace=true, show_trace=true))
##
x_trace(res)
[fϕ_start.f2,res.minimizer[~fϕ_start].f2,ϕ₀]' |> plot

f = 𝕎(Cf,CN̂)*(LenseFlow(res.minimizer[~fϕ_start].f2)\res.minimizer[~fϕ_start].f1)
f = 𝕎(Cf,CN̂)*(LenseFlow(res.minimizer[~fϕ_start].f2)\fϕ_start.f1)
[fϕ_start.f1,f,f₀] |> plot
plot(res.minimizer[~fϕ_start].f2/Map(ϕ₀)-1,vmin=-0.3,vmax=0.3)

[ϕ₀ fstart[2]] |> plot
