push!(LOAD_PATH, pwd()*"/src")
using CMBFields

## calc Cℓs and store in Main since I reload CMBFields alot during development
cls = isdefined(Main,:cls) ? Main.cls : @eval Main cls=$(class(lmax=6000,r=1e-3))
## set up the types of maps
Θpix, nside, T = 3, 256, Float32
P = Flat{Θpix,nside}
## covariances 
Cf    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],cls[:te],cls[:ee],cls[:bb])
Cϕ    = Cℓ_to_cov(T,P,S0,cls[:ℓ],cls[:ϕϕ])
μKarcminT = 1e-3
Ωpix = deg2rad(Θpix/60)^2
CN  = FullDiagOp(FlatIQUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),3)...))
CÑ = Cℓ_to_cov(T,P,S0,S2, 0:6000, repeated(μKarcminT^2 * Ωpix * ones(6001),4)...)
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
((@inferred lnP(f₀,0,ϕ₀,ds_pl,PowerLens)), 
 (@inferred lnP(f₀,0,ϕ₀,ds_lf,LenseFlowOp)),
 (@inferred lnP(L_lf*f₀,1,ϕ₀,ds_lf,LenseFlowOp)))
## PowerLens gradient at t=0
(@inferred δlnP_δfₜϕ(f₀,0,ϕ₀,ds_pl,PowerLens)⋅(δf,δϕ)), (lnP(f₀+ϵ*δf,0,ϕ₀+ϵ*δϕ,ds_pl,PowerLens) - lnP(f₀-ϵ*δf,0,ϕ₀-ϵ*δϕ,ds_pl,PowerLens))/(2ϵ)
## LenseFlow gradient at t=0
(@inferred δlnP_δfₜϕ(f₀,0,ϕ₀,ds_lf,LenseFlowOp)⋅(δf,δϕ)), (lnP(f₀+ϵ*δf,0,ϕ₀+ϵ*δϕ,ds_lf,LenseFlowOp) - lnP(f₀-ϵ*δf,0,ϕ₀-ϵ*δϕ,ds_lf,LenseFlowOp))/(2ϵ)
## LenseFlow gradient at t=1
(@inferred δlnP_δfₜϕ(L_lf*f₀,1,ϕ₀,ds_lf,LenseFlowOp)⋅(δf,δϕ)), (lnP(L_lf*f₀+ϵ*δf,1,ϕ₀+ϵ*δϕ,ds_lf,LenseFlowOp) - lnP(L_lf*f₀-ϵ*δf,1,ϕ₀-ϵ*δϕ,ds_lf,LenseFlowOp))/(2ϵ)
##

using PyPlot
fstart = 𝕎(Cf,CÑ)*d_lf

##
semilogy(get_Cℓ(fstart.f2)...)
semilogy(get_Cℓ(n₀.f2)...)
semilogy(get_Cℓ(d_lf.f2)...)
##
gf,gϕ = δlnP_δfₜϕ(fstart,1,0ϕ₀,ds_lf,LenseFlowOp);
[L_lf*f₀, fstart, 1e-6Cf*gf] |> plot
[ϕ₀, Cϕ*gϕ] |> plot

semilogy(get_Cℓ(ϕ₀)...)
semilogy(get_Cℓ(1e-6Cϕ*gϕ)...)

##
gf,gϕ = δlnP_δfₜϕ(fstart,0,0ϕ₀,ds_lf,LenseFlowOp);
Cϕ*δlnP_δfₜϕ(fstart,1,0ϕ₀,ds_lf,LenseFlowOp)[2] |> plot


Cf^(-1)

## older stuff below here which I still need to get working again....
using Optim
f₀ = simulate(Cf) |> LenseBasis
ϕ₀ = simulate(Cϕ) |> LenseBasis
L = PowerLens(ϕ₀)
df̃ = L*Ł(f₀) + simulate(CN)
##
fstart = [Ł(𝕎(Cf,CÑ)*df̃), zero(FlatS0Map{T,P})]
[f₀,fstart[1],df̃] |> plot
##
import Base.LinAlg.A_ldiv_B!
struct foo end
A_ldiv_B!(s,::foo,q) = ((f,ϕ) = q[~(f₀,ϕ₀)]; s.=[Ł(CÑ*f),Ł(Cϕ*ϕ)][:])
##
res = optimize(
    x->lnL̃(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp),
    (x,∇f)->(∇f .= dlnL̃_df̃ϕ(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp)[:]),
    fstart[:],
    LBFGS(P=foo()),
    Optim.Options(time_limit = 600.0, store_trace=true, show_trace=true))
##
fstart = res.minimizer[~(f₀,ϕ₀)]
fstart[1] = LenseFlowOp(fstart[2])\fstart[1]
##
res2 = optimize(
    x->lnL(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp),
    (x,∇f)->(∇f .= dlnL_dfϕ(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp)[:]),
    fstart[:],
    LBFGS(P=foo()),
    Optim.Options(time_limit = 60.0, store_trace=true, show_trace=true))
##
[f₀,fstart[1]-f₀,res.minimizer[~(f₀,ϕ₀)][1]-f₀] |> plot
[ϕ₀,fstart[2]-ϕ₀,res.minimizer[~(f₀,ϕ₀)][2]-ϕ₀] |> plot
[fstart[2] res.minimizer[~(f₀,ϕ₀)][2] ϕ₀] |> plot
plot([res.minimizer[~(f₀,ϕ₀)][2] ϕ₀]; vmin=-6e-6, vmax=6e-6)
norm(f₀[:],1)
maximum(abs(f₀[:]))
##
[f₀,dlnL_dfϕ(fstart...,df̃,LenseFlowOp)[1]] |> plot
[f₀, dlnL_dfϕ(f₀,ϕ₀,df̃,LenseFlowOp)[1]] |> plot
[-Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0ϕ₀,df̃,LenseFlowOp)[2] -Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0.9ϕ₀,df̃,LenseFlowOp)[2] ϕ₀;
 -Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0ϕ₀,df̃,PowerLens)[2] -Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0.9ϕ₀,df̃,PowerLens)[2] ϕ₀] |> plot
plot([Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,4ϕ₀,df̃,LenseFlowOp)[2] Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,4ϕ₀,df̃,PowerLens)[2]])#; vmin=-2e16, vmax=2e16)

[-dlnL_dfϕ(0𝕎(Cf,CÑ)*df̃,0ϕ₀,df̃,PowerLens)[1] f₀] |> plot
##
fstart = [Ł(𝕎(Cf,CÑ)*df̃), zero(FlatS0Map{T,P})]
∇L = dlnL_dfϕ(fstart...,df̃,PowerLens)
iP_∇L = [(CÑ^-1+Cf^-1)^-1*∇L[1], Cϕ*∇L[2]]
l = lnL(fstart...,df̃,PowerLens)
close("all")
α=logspace(-10,-12,100)
loglog(α,[-(lnL((fstart - α*iP_∇L)...,df̃,PowerLens)-l) for α=α])
# yscale("symlog")
##
fstart = [Ł(𝕎(Cf,CÑ)*df̃), zero(FlatS0Map{T,P})]
##
∇L = dlnL̃_df̃ϕ(fstart...,df̃,LenseFlowOp)
iP_∇L = [CÑ*∇L[1], Cϕ*∇L[2]]
l = lnL̃(fstart...,df̃,LenseFlowOp)
close("all")
α=logspace(log10(0.4),-3,100)
semilogx(α,[(l-lnL̃((fstart - α*iP_∇L)...,df̃,LenseFlowOp)) for α=α])
##
ylim(-1000,1000)
lnL̃((fstart - 0*iP_∇L)...,df̃,LenseFlowOp)
lnL̃((fstart - 0.17*iP_∇L)...,df̃,LenseFlowOp)

fstart = (fstart - 0.17*iP_∇L)


[ϕ₀ fstart[2]] |> plot
