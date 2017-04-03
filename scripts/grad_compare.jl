push!(LOAD_PATH, pwd()*"/src")
using CMBFields
using PyPlot

## calc Cℓs and store in Main since I reload CMBFields alot during development
cls = isdefined(Main,:cls) ? Main.cls : @eval Main cls=$(class(lmax=6000,r=1e-3))
## set up the types of maps
Θpix, nside, T = 3, 128, Float32
P = Flat{Θpix,nside}
## covariances 
Cf    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],   cls[:te],   cls[:ee],   cls[:bb])
Cf̃    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:ln_tt],cls[:ln_te],cls[:ln_ee],cls[:ln_bb])
Cϕ    = Cℓ_to_cov(T,P,S0,   cls[:ℓ],cls[:ϕϕ])
μKarcminT = 0.1
Ωpix = deg2rad(Θpix/60)^2
CN  = FullDiagOp(FlatIQUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),3)...))
CN̂ = Cℓ_to_cov(T,P,S0,S2, 0:6000, repeated(μKarcminT^2 * Ωpix * ones(6001),4)...)
## masks
ℓmax_mask = 2000
Mf    = Cℓ_to_cov(T,P,S0,S2,1:ℓmax_mask,repeated(ones(ℓmax_mask),4)...) * Squash
Mϕ    = Cℓ_to_cov(T,P,S0,1:ℓmax_mask,ones(ℓmax_mask)) * Squash
## generate simulated datasets
ϕ₀ = simulate(Cϕ)
f₀ = simulate(Cf)
n₀ = simulate(CN)
L_lf = LenseFlowOp(ϕ₀)
L_pl = PowerLens(ϕ₀)
d_lf = L_lf*f₀ + n₀
d_pl = L_pl*f₀ + n₀
ds_pl = DataSet(d_pl,CN,Cf,Cϕ,Mf,Mϕ);
ds_lf = DataSet(d_lf,CN,Cf,Cϕ,Mf,Mϕ);
##

#### check accuracy of likelihood and derivatives for the two algorithms
using Base.Test
ϵ = 1e-3
δϕ = simulate(Cϕ)
δf = simulate(Cf)
## likelihoood evaluated with PowerLens at t=0 and with LenseFlow at t=0 and t=1
((@inferred lnP(0,f₀,ϕ₀,ds_pl,PowerLens)), 
 (@inferred lnP(0,f₀,ϕ₀,ds_lf,LenseFlowOp)),
 (@inferred lnP(1,L_lf*f₀,ϕ₀,ds_lf,LenseFlowOp)))
## PowerLens gradient at t=0
(@inferred δlnP_δfₜϕ(0,f₀,ϕ₀,ds_pl,PowerLens)⋅(δf,δϕ)), (lnP(0,f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_pl,PowerLens) - lnP(0,f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_pl,PowerLens))/(2ϵ)
## LenseFlow gradient at t=0
(@inferred δlnP_δfₜϕ(0,f₀,ϕ₀,ds_lf,LenseFlowOp)⋅(δf,δϕ)), (lnP(0,f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_lf,LenseFlowOp) - lnP(0,f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_lf,LenseFlowOp))/(2ϵ)
## LenseFlow gradient at t=1
(@inferred δlnP_δfₜϕ(1,L_lf*f₀,ϕ₀,ds_lf,LenseFlowOp)⋅(δf,δϕ)), (lnP(1,L_lf*f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_lf,LenseFlowOp) - lnP(1,L_lf*f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_lf,LenseFlowOp))/(2ϵ)
##

### 
using Optim
using Optim: x_trace
##
fϕ_start = Ł(FieldTuple(𝕎(Cf̃,CN̂)*d_lf,0ϕ₀))
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
    (x,∇lnP)->(println(2); ∇lnP .= -Δx²*FΦ(FieldTuple(δlnP_δfₜϕ(1,x[~fϕ_start]...,ds_lf)...))[:]),
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

f = 𝕎(Cf,CN̂)*(LenseFlowOp(res.minimizer[~fϕ_start].f2)\res.minimizer[~fϕ_start].f1)
f = 𝕎(Cf,CN̂)*(LenseFlowOp(res.minimizer[~fϕ_start].f2)\fϕ_start.f1)
[fϕ_start.f1,f,f₀] |> plot
plot(res.minimizer[~fϕ_start].f2/Map(ϕ₀)-1,vmin=-0.3,vmax=0.3)

[ϕ₀ fstart[2]] |> plot
