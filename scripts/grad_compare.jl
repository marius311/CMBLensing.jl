push!(LOAD_PATH, pwd()*"/src")
using CMBFields


## calc Cℓs and store in Main since I reload CMBFields alot during development
cls = isdefined(Main,:cls) ? Main.cls : @eval Main cls=$(class(lmax=4000))
## set up the types of maps
Θpix, nside, T = 1, 256, Float32
P = Flat{Θpix,nside}
## covariances 
Cf    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],cls[:te],cls[:ee],cls[:bb])
Cϕ    = Cℓ_to_cov(T,P,S0,cls[:ℓ],cls[:ϕϕ])
μKarcminT = 0.001
Ωpix = deg2rad(Θpix/60)^2
CN  = FullDiagOp(FlatIQUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),3)...))
## masks
Mf    = Cℓ_to_cov(T,P,S0,S2,1:2000,repeated(ones(2000),4)...)
Mϕ    = Cℓ_to_cov(T,P,S0,1:2000,ones(2000))
## generate simulated datasets
ϕ₀ = simulate(Cϕ)
f₀ = simulate(Cf)
n₀ = simulate(CN)
df̃_lf = LenseFlowOp(ϕ₀)*f₀ + n₀
df̃_pl = PowerLens(ϕ₀)*f₀ + n₀
ds_pl = DataSet(df̃_pl,CN,Cf,Cϕ,Mf,Mϕ);
ds_lf = DataSet(df̃_lf,CN,Cf,Cϕ,Mf,Mϕ);
##

## check accuracy of likelihood and derivatives for the two algorithms
using Base.Test
ϵ = 1e-5
δϕ = simulate(Cϕ)
δf = simulate(Cf)
##
(@inferred lnP(f₀,ϕ₀,ds_pl,0,PowerLens)), (@inferred lnP(f₀,ϕ₀,ds_lf,0.,LenseFlowOp))
##
(@inferred δlnP_δfϕ(f₀,ϕ₀,ds_pl,0,PowerLens)⋅(δf,δϕ)), (lnP(f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_pl,0,PowerLens) - lnP(f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_pl,0,PowerLens))/(2ϵ)
##
(@inferred δlnP_δfϕ(f₀,ϕ₀,ds_lf,0,LenseFlowOp)⋅(δf,δϕ)), (lnP(f₀+ϵ*δf,ϕ₀+ϵ*δϕ,ds_lf,0,LenseFlowOp) - lnP(f₀-ϵ*δf,ϕ₀-ϵ*δϕ,ds_lf,0,LenseFlowOp))/(2ϵ)
##


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
